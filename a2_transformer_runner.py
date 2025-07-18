import time
from tqdm import tqdm  # Not required, but you're welcome to use it.
import argparse
from typing import Callable, Sequence

import torch

import a2_utils
import a2_dataloader
from a2_transformer_model import TransformerEncoderDecoder
from a2_bleu_score import BLEU_score

try:
    import wandb
except ImportError:
    pass


class TransformerRunner:
    """
    Interface between model and training and inference operations
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        src_vocab_size: int,
        tgt_vocab_size: int,
        padding_idx: int = 2,
    ):
        self.opts = opts

        # dataset info
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.padding_idx = padding_idx

        # model hyper-params
        self.num_layers = opts.encoder_num_hidden_layers
        self.d_model = opts.word_embedding_size
        self.d_ff = opts.transformer_ff_size
        self.num_heads = opts.heads  # default = 4, same used in mh_attn
        self.dropout = opts.encoder_dropout
        self.atten_dropout = opts.attention_dropout
        self.is_pre_layer_norm = not self.opts.with_post_layer_norm

        self.BLEU_score = BLEU_score

        self.model = TransformerEncoderDecoder(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.padding_idx,
            self.num_layers,
            self.d_model,
            self.d_ff,
            self.num_heads,
            self.dropout,
            self.atten_dropout,
            self.is_pre_layer_norm,
            no_src_pos=opts.no_source_pos,
            no_tgt_pos=opts.no_target_pos,
        )

        for p in self.model.parameters():  # parameter initialization
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        # training and eval hyper-params
        self.accum_iter = getattr(self.opts, "gradient_accumulation", 1)
        self.label_smoothing = 0.0  # 0.1
        self.lr = 1.0
        self.warmup = 300  # 3000 // self.accum_iter
        self.factor = 1.0
        self._opt_d = 512  # 512,  # self.d_model

        if getattr(self.opts, "patience", None):
            self.max_epochs = float("inf")
            self.patience = self.opts.patience
        else:
            self.max_epochs = getattr(self.opts, "epochs", 7)
            self.patience = float("inf")

    def load_model(self):
        state_dict = torch.load(self.opts.model_path)
        self.model.load_state_dict(state_dict)
        del state_dict

    def save_model(self):
        device = next(self.model.parameters()).device
        self.model.cpu()
        with a2_utils.smart_open(self.opts.model_path, "wb") as model_file:
            torch.save(self.model.state_dict(), model_file)
        self.model.to(device)

    def init_visualization(self):
        writer = None
        if self.opts.viz_wandb:
            # View at: https://wandb.ai/<opts.viz_wandb>/csc401-w24-a2
            wandb.init(
                name=f"Train-{type(self.model.decoder).__name__}",
                project="csc401-w24-a2",
                entity=self.opts.viz_wandb,
                sync_tensorboard=(self.opts.viz_tensorboard is not None),
            )
            wandb.config = {
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "batch_size": self.opts.batch_size,
                "source_vocab_size": self.src_vocab_size,
                "target_vocab_size": self.tgt_vocab_size,
            }
            wandb.watch(self.model)
        if self.opts.viz_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(comment="a2_run")  # outputs to ./runs/
        return writer

    def update_visualization(self, writer, cur_epoch, loss, bleu_list, log_str):
        if self.opts.viz_tensorboard:
            writer.add_scalar("Loss/train", loss, cur_epoch)
            for n, bleu in bleu_list:
                writer.add_scalar(f"BLEU{n}/train", bleu, cur_epoch)
            writer.add_text("", log_str, cur_epoch)
        if self.opts.viz_wandb:
            # wandb.log({"loss": loss, "global_step": epoch})
            d = {"loss": loss}
            for n, bleu in bleu_list:
                d[f"BLEU-{n}"] = bleu
            wandb.log(d)

    def finalize_visualization(self, writer):
        if self.opts.viz_tensorboard:
            writer.flush()  # Ensure all pending events have been written to disk
            writer.close()
        if self.opts.viz_wandb:
            wandb.finish()

    def train(
        self,
        train_dataloader: a2_dataloader.HansardDataLoader,
        dev_dataloader: a2_dataloader.HansardDataLoader,
        device: torch.device = torch.device("cpu"),
        n_gram_levels: tuple[int, ...] = (4, 3),
    ):
        """
        Train all epochs
        """
        # optimizer,
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: a2_utils.schedule_rate(
                step, self._opt_d, factor=self.factor, warmup=self.warmup
            ),
        )
        # loss function
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        # put on device
        self.model.to(device)
        criterion.to(device)

        writer = self.init_visualization()

        # current state
        cur_step = 0
        cur_epoch = 1

        best_bleu = 0.0
        num_poor = 0

        start = time.time()
        while cur_epoch <= self.max_epochs and num_poor < self.patience:
            self.model.train()
            print(
                f"[Device:{self.opts.device}] Epoch {cur_epoch} Training ====",
                flush=True,
            )
            loss, num_steps = self.train_for_epoch(
                train_dataloader,
                optimizer,
                scheduler,
                criterion,
                self.opts.device,
                self.accum_iter,
            )
            cur_step += num_steps

            torch.cuda.empty_cache()

            print(
                f"[Device:{self.opts.device}] Epoch {cur_epoch} Validation ====",
                flush=True,
            )
            self.model.eval()
            bleu = None
            if (
                cur_epoch > self.opts.skip_eval
            ):  # skip BLEU computation for the first few epochs until model converges
                with torch.no_grad():
                    bleu = self.compute_average_bleu_over_dataset(
                        self.BLEU_score,
                        dev_dataloader,
                        device=self.opts.device,
                        use_greedy_decoding=self.opts.greedy,  # TODO change
                    )

            t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            if bleu is not None:
                bleu_str = " ".join(
                    [f"BLEU-{n}: {b:.4f}" for n, b in zip(n_gram_levels, bleu)]
                )
            else:
                bleu_str = f"BLEU: skipped until epoch {self.opts.skip_eval + 1}"
                bleu = [0.0] * len(n_gram_levels)  # for visualizer
            log_str = f"Epoch {cur_epoch}: loss={loss}, {bleu_str}, time={t}"
            self.update_visualization(
                writer, cur_epoch, loss, zip(n_gram_levels, bleu), log_str
            )
            print(log_str)

            if bleu[0] < best_bleu:  # use first n-gram level for early stopping
                num_poor += 1
            else:
                num_poor = 0
                best_bleu = bleu[0]
            cur_epoch += 1
        if cur_epoch > self.max_epochs:
            print(f"Finished {self.max_epochs} epochs")
        else:
            print(f"BLEU did not improve after {self.patience} epochs. Done.")

        self.finalize_visualization(writer)

    @staticmethod
    def train_input_target_split(
        target_tokens: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Split target tokens into input and target for maximum likelihood training (teacher forcing)

        target_tokens: torch.Tensor Long, [batch_size, seq_len]
        return: the model inputs [batch_size, seq_len - 1],
            and the training targets [batch_size * (seq_len - 1)] as a flat, contiguous tensor
        """
        input_tokens = target_tokens[:, :-1]
        target_tokens_flat = target_tokens[:, 1:]
        target_tokens_flat = target_tokens_flat.contiguous().view(-1)
        return input_tokens, target_tokens_flat

    @staticmethod
    def train_step_optimizer_and_scheduler(
        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR
    ) -> None:
        """
        Step the optimizer, zero out the gradient, and step scheduler
        """
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        return

    def train_for_epoch(
        self,
        dataloader: a2_dataloader.HansardDataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        criterion: torch.nn.CrossEntropyLoss,
        device: torch.device = torch.device("cpu"),
        accum_iter: int = 20,
    ):
        """
        Train a single epoch

        Transformers generally perform better with larger batch sizes,
        so we use gradient accumulation (accum_iter) to simulate larger batch sizes.
        This means that we only update the model parameters every accum_iter batches.
        Remember to scale the loss correctly when backpropagating with gradient accumulation.

           1. For every iteration of the `dataloader`
            which yields triples `source, source_lens, targets`, where
                source, [batch_size, src_seq_len] and padded with sos, eos, and padding tokens
                source_lens, [batch_size]
                targets  [batch_size, tgt_seq_len] and padded with sos, eos, and padding tokens

           2. Sends these to the appropriate device via `.to(device)` etc.
           3. Splits the targets into model (`self.model`) input and criterion targets
           4. Gets the logits via the model
           5. Gets the loss via the criterion
           6. Backpropagate gradients through the model
           7. Updates the model parameters every `accum_iter` iterations:
            a) step the optimizer
            b) zero the gradients
            c) step the scheduler
           8. Returns the average loss over sequences (as a float not a tensor), num iterations

        return: float, int
        """
        total_loss = 0.0
        num_iters = 0
        total_sequences = 0.0
        
        
        for source, source_lens, targets in dataloader:
            #many batches of source tensors and respective targets tensors. first dim is number of samples per batch
            num_iters += 1 #number of mini-batches in an epoch
            source = source.to(device)
            source_lens = source_lens.to(device) #irrelevent
            targets = targets.to(device)

            # Split targets into model input and criterion targets
            input_tokens, target_tokens_flat = self.train_input_target_split(targets)
            logits = self.model(source, input_tokens) #[batch_size, tgt_seq_len, tgt_vocab_size]
            logits = logits.view(-1, logits.size(-1)) #[batch_size * tgt_seq_len, tgt_vocab_size]
            loss = criterion(logits, target_tokens_flat)
            loss = loss / accum_iter
            loss.backward()
            total_loss += loss.item() * accum_iter
            total_sequences += source.shape[0]
            
            
            if num_iters % accum_iter == 0:
                self.train_step_optimizer_and_scheduler(optimizer, scheduler)
                
        if num_iters % accum_iter != 0:
            self.train_step_optimizer_and_scheduler(optimizer, scheduler)
        
        avg_loss = total_loss / total_sequences  if total_sequences > 0 else 0.0
        
        return avg_loss, num_iters
            
            
            

    def test(self, dataloader, n_gram_levels: tuple[int, ...] = (4, 3)):
        self.load_model()
        self.model.to(self.opts.device)
        self.model.eval()
        with torch.no_grad():
            bleu = self.compute_average_bleu_over_dataset(
                self.BLEU_score,
                dataloader,
                device=self.opts.device,
                use_greedy_decoding=self.opts.greedy,
            )
            bleu_str = " ".join(
                [f"BLEU-{n}: {b:.4f}" for n, b in zip(n_gram_levels, bleu)]
            )
        print(f"The average BLEU score over the test set was {bleu_str}")

    def translate(self, input_sentence): #5.1
        """
        Translate a single sentence

        This method translates the input sentence from the model's source
        language to the target language.
        1. Tokenize the input sentence.
        2. Convert tokens into ordinal IDs.
        3. Feed the tokenized sentence into the model.
        4. Decode the output of the sentence into a string.

        Consult :class:`HansardEmptyDataset` for a description of parameters
        and attributes.
          self.dataset.tokenize(input_sentence)
              This method tokenizes the input sentence.  For example:
              >>> self.dataset.tokenize('This is a sentence.')
              ['this', 'is', 'a', 'sentence']
          self.dataset.source_word2id
              A dictionary that maps tokens to ids for the source language.
              For example: `self.dataset.source_word2id['francophone'] -> 5127`
          self.dataset.source_unk_id
              The speical token for unknown input tokens.  Any token in the
              input sentence that isn't present in the source vocabulary should
              be converted to this special token.
          self.dataset.target_id2word
              A dictionary that maps ids to tokens for the target language.
              For example: `self.dataset.target_id2word[6123] -> 'anglophone'`

        return: str
        """
        sos_idx, eos_idx, pad_idx = (
            self.dataset.target_sos_id,
            self.dataset.target_eos_id,
            self.padding_idx,
        )

        # Step 1: Tokenize the input sentence.
        # Step 2: Convert tokens into ordinal IDs.
        tokenized = self.dataset.tokenize(input_sentence)
        source_tokens = []
        for tok in tokenized:
            id = self.dataset.source_word2id.get(tok)
            if not id:
                id = self.dataset.source_unk_id
            source_tokens.append(id)
        #covert source_tokens: torch.Tensor, [1, len(source_tokens)]
        source_tokens_tensor = torch.tensor(source_tokens).unsqueeze(0)
        # === ============== === #
        # Step 3. Feed the tokenized sentence into the model.
        if self.opts.greedy:
            hypotheses = self.model.greedy_decode(source_tokens_tensor, sos_idx, eos_idx)
        else:
            hypotheses = self.model.beam_search_decode(
                source_tokens_tensor, sos_idx, eos_idx, k=self.opts.beam_width
            )
        #torch.Tensor, [batch_size, max_seq_len]
        # === Your Code Here === #
        # Step 4. Decode the output of the sentence into a string.
        readable = ""
        for i, hyp in enumerate(hypotheses[0]):
            if i == 0: readable = self.dataset.target_id2word[hyp.item()]
            else:   readable += " " + self.dataset.target_id2word[hyp.item()]
        return readable
        # === ============== === #

    @staticmethod
    def compute_batch_total_bleu(
        bleu_score_func: Callable[[Sequence[str], Sequence[str], int], float],
        target_y_ref: torch.LongTensor,
        target_y_cand: torch.LongTensor,
        sos_idx: int,
        eos_idx: int,
        pad_idx: int,
        n_gram_levels: tuple[int, ...] = (4, 3),
    ) -> tuple[tuple[float, ...], int]:
        """
        Compute the total BLEU score for each n_gram_level in n_gram_levels over elements in a batch.
        Clean up the sequences by removing ALL special tokens (sos_idx, eos_idx, pad_idx).

        Assume that the candidate sequences have been padded after the eos token.

        bleu_score_func: Callable[[Sequence[str], Sequence[str], int], float] from BLEU_score.py
        target_y_ref : torch.LongTensor [batch_size, max_ref_seq_len]
        target_y_cand : torch.LongTensor [batch_size, max_cand_seq_len]
        sos_idx : int start of sentence special token
        eos_idx : int end of sentence special token
        pad_idx : int padding special token
        n_gram_levels : tuple[int] n-gram levels to compute BLEU score at i.e the precisions

        return: list summed BLEU score at each level for batch, batch_size
        """
        batch_size = target_y_ref.size(0)
        total_bleu = [0.0] * len(n_gram_levels)

        for i in range(batch_size):
            ref_seq = target_y_ref[i].tolist()
            cand_seq = target_y_cand[i].tolist()
            ref_seq_cleaned = [token for token in ref_seq if token not in [sos_idx, eos_idx, pad_idx]]
            cand_seq_cleaned = [token for token in cand_seq if token not in [sos_idx, eos_idx, pad_idx]]
            # print("ref_seq_cleaned: ", ref_seq_cleaned)
            # print("cand_seq_cleaned: ", cand_seq_cleaned)
            for j, n in enumerate(n_gram_levels):
                bleu_score = bleu_score_func(ref_seq_cleaned, cand_seq_cleaned, n)
                total_bleu[j] += bleu_score

        return total_bleu, batch_size


    def compute_average_bleu_over_dataset(
        self,
        bleu_score_func: Callable[[Sequence[str], Sequence[str], int], float],
        dataloader: a2_dataloader.HansardDataLoader,
        device: torch.device = torch.device("cpu"),
        max_len: int = 100,
        use_greedy_decoding: bool = True,
        n_gram_levels: tuple[int, ...] = (4, 3),
    ) -> tuple[float, ...]:
        """
        Determine the average BLEU score across sequences

        This function computes the average BLEU score across all sequences in
        a single loop through the `dataloader`.

        Returns avg_bleu : float
            The total BLEU score summed over all sequences divided by the number of
            sequences
        """

        assert not self.model.training
        sos_idx, eos_idx, pad_idx = (
            dataloader.target_sos_id,
            dataloader.target_eos_id,
            self.padding_idx,
        )

        total_bleu = [0.0] * len(n_gram_levels)
        num = 0.0
        for i, (source_tokens, _, target_tokens) in enumerate(dataloader):
            source_tokens = source_tokens.to(device)
            # this is `cheating` as it uses the target info,
            # but it makes it faster to compute the bleu score
            max_len_i = min(target_tokens.shape[1] + 5, max_len)
            # print(f"max_len_i: {max_len_i} for batch {i}/ {len(dataloader)}", flush=True)

            if use_greedy_decoding:
                candidates = self.model.greedy_decode(
                    source_tokens, sos_idx, eos_idx, max_len_i
                )
            else:
                candidates = self.model.beam_search_decode(
                    source_tokens, sos_idx, eos_idx, max_len_i, k=self.opts.beam_width
                )
            batch_bleu, batch_size = self.compute_batch_total_bleu(
                bleu_score_func, target_tokens, candidates, sos_idx, eos_idx, pad_idx
            )
            for j in range(len(batch_bleu)):
                total_bleu[j] += batch_bleu[j]
            num += batch_size
        return tuple(b / num for b in total_bleu)
