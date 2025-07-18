import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x): #1.1
        """
        Compute layer normalization
            y = gamma * (x - mu) / (sigma + eps) + beta where mu and sigma are computed over the feature dimension

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        mu =  x.mean(-1, keepdim=True)
        sigma = x.std(-1, keepdim=True)
        y = self.gamma * (x - mu) / (sigma + self.eps) + self.beta
        return y


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for both self-attention and cross-attention
    """

    def __init__(
        self,
        num_heads,
        d_model,
        dropout=0.0,
        atten_dropout=0.0,
        store_attention_scores=False,
    ):
        """
        num_heads: int, the number of heads
        d_model: int, the dimension of the model
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        store_attention_scores: bool, whether to store the attention scores for visualization
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        # Assume values and keys are the same size
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.atten_dropout = nn.Dropout(p=atten_dropout)  # applied after softmax

        # applied at the end
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # used for visualization
        self.store_attention_scores = store_attention_scores
        self.attention_scores = None  # set by set_attention_scores

    def set_attention_scores(self, scores):
        """
        The attention scores should be given after masking but before the softmax.
        scores: torch.Tensor, shape [batch_size, num_heads, query_seq_len, key_seq_len]
        return: None
        """
        if scores is None:  # for clean up
            self.attention_scores = None
        if self.store_attention_scores and not self.training:
            self.attention_scores = scores.cpu().detach().numpy()

    def attention(self, query, key, value, mask=None): #1.3
        """
        Scaled dot product attention
        The mask is applied before the softmax.
        Attention dropout `self.atten_dropout` is applied to the attention weights after the softmax.

        query: torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        key: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        value: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        mask:  torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True, where masked or None

        return torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) # [batch_size, num_heads, query_seq_len, key_seq_len]
        d_k = torch.tensor(self.d_head, dtype=scores.dtype) # Scalar
        attention_scores = scores / math.sqrt(d_k) # [batch_size, num_heads, query_seq_len, key_seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1) # [batch_size, 1, query_seq_len, key_seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # [batch_size, num_heads, query_seq_len, key_seq_len]
        self.set_attention_scores(attention_scores)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1) # [batch_size, num_heads, query_seq_len, key_seq_len]
        attention_weights = self.atten_dropout(attention_weights)  # [batch_size, num_heads, query_seq_len, key_seq_len]
        context = torch.matmul(attention_weights, value)  # [batch_size, num_heads, query_seq_len, key_seq_len]

        return context

    def forward(self, query, key=None, value=None, mask=None): #1.4
        """
        If the key and values are None, assume self-attention is being applied.  Otherwise, assume cross-attention.

        Note we only need one mask, which will work for either causal self-attention or cross-attention as long as
        the mask is set up properly beforehand.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        query: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        key: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        value: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        mask: torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True where masked or None

        return: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        """
        batch_size = query.shape[0]
        Q = self.q_linear(query) # [batch_size, query_seq_len, d_model]
        
        if (key == None or value == None):
            key = query
            value = query
            
        K = self.k_linear(key) # [batch_size, key_seq_len, d_model]
        V = self.v_linear(value) # [batch_size, key_seq_len, d_model]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3) # [batch_size, num_heads, query_seq_len, d_head]
        K = K.view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3) # [batch_size, num_heads, key_seq_len, d_head]
        V = V.view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3) #  [batch_size, num_heads, key_seq_len, d_head]


        context = self.attention(Q, K, V, mask) # [batch_size, num_heads, query_seq_len, d_head]
        context = context.permute(0, 2, 1, 3) # [batch_size, query_seq_len, num_heads, d_head]
        context = context.contiguous().view(batch_size, -1, self.num_heads * self.d_head) # [batch_size, query_seq_len, num_heads * d_head]
        output = self.out_linear(context) # [batch_size, query_seq_len, d_model]
        output = self.dropout(output) # [batch_size, query_seq_len, d_model]

        return output


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.f = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x): #1.2
        """
        Compute the feedforward sublayer.
        Dropout is applied after the activation function and after the second linear layer

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        x = self.w_1(x)
        x = self.f(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        x = self.dropout2(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """

    Idea if we can give this init done, then the students can fill in the decoder init in the same way but add in cross attention


    Performs multi-head self attention and FFN with the desired pre- or post-layer norm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        """
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        """
        super(TransformerEncoderLayer, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_heads

        self.ln1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout
        )

        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)

    def pre_layer_norm_forward(self, x, mask): #2.1.1
        """
        x: torch.Tensor, the input to the layer
        mask: torch.Tensor, the mask to apply to the attention
        """
        input = x
        x = self.ln1(x)
        x = input + self.self_attn(x, None, None, mask)
        x = x + self.ff(self.ln2(x))
        return x

    def post_layer_norm_forward(self, x, mask): #2.1.2
        attn = self.self_attn(x, None, None, mask)
        x = x + attn
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x

    def forward(self, x, mask):
        if self.is_pre_layer_norm:
            return self.pre_layer_norm_forward(x, mask)
        else:
            return self.post_layer_norm_forward(x, mask)


class TransformerEncoder(nn.Module):
    """
    Stacks num_layers of TransformerEncoderLayer and applies layer norm at the correct place.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        super(TransformerEncoder, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        self.layers = torch.nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    d_model, d_ff, num_heads, dropout, atten_dropout, is_pre_layer_norm
                )
            )
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        """
        x: torch.Tensor, the input to the encoder
        mask: torch.Tensor, the mask to apply to the attention
        """
        if not self.is_pre_layer_norm:
            x = self.norm(x)
        for layer in self.layers:
            x = layer(x, mask)
        if self.is_pre_layer_norm:
            x = self.norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Performs multi-head self attention, multi-head cross attention, and FFN,
    with the desired pre- or post-layer norm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        """
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        is_pre_layer_norm: bool, whether to apply layer norm before or after each sublayer
        """
        #2.2.3
        super(TransformerDecoderLayer, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)

    def pre_layer_norm_forward(self, x, mask, src_x, src_mask): #2.2.1
        input = x
        x = self.ln1(x)
        x = input + self.self_attn(x, x, x, mask)
        input = x
        x = self.ln2(x)
        x = input + self.cross_attn(x, src_x, src_x, mask=src_mask)
        x = x + self.ff(self.ln3(x))
        return x

    def post_layer_norm_forward(self, x, mask, src_x, src_mask): #2.2.2
        x = x + self.self_attn(x, None, None, mask)
        x = self.ln1(x)
        x = x + self.cross_attn(x, src_x, src_x, mask=src_mask)
        x = self.ln2(x)
        x = x + self.ff(x)
        x = self.ln3(x)
        return x

    def forward(self, x, mask, src_x, src_mask):
        if self.is_pre_layer_norm:
            return self.pre_layer_norm_forward(x, mask, src_x, src_mask)
        else:
            return self.post_layer_norm_forward(x, mask, src_x, src_mask)

    def store_attention_scores(self, should_store=True):
        self.self_attn.store_attention_scores = should_store
        self.cross_attn.store_attention_scores = should_store

    def get_attention_scores(self):
        return self.self_attn.attention_scores, self.cross_attn.attention_scores


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        super(TransformerDecoder, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        self.layers = torch.nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    d_model, d_ff, num_heads, dropout, atten_dropout, is_pre_layer_norm
                )
            )
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)  # logit projection

    def forward(self, x, mask, src_x, src_mask, normalize_logits: bool = False): #2.3
        """
        x: torch.Tensor, the input to the decoder
        mask: torch.Tensor, the mask to apply to the attention
        src_x: torch.Tensor, the output of the encoder
        src_mask: torch.Tensor, the mask to apply to the attention
        normalize_logits: bool, whether to apply log_softmax to the logits

        Returns the logits or log probabilities if normalize_logits is True

        """
        if not self.is_pre_layer_norm:
            x = self.norm(x)

        for layer in self.layers:
            x = layer(x, mask, src_x, src_mask)

        if self.is_pre_layer_norm:
            x = self.norm(x)

        x = self.proj(x)
        if normalize_logits:
            x = nn.functional.log_softmax(x, dim=-1)
        
        return x

    def store_attention_scores(self, should_store=True):
        for layer in self.layers:
            layer.store_attention_scores(should_store)

    def get_attention_scores(self):
        """
        Return the attention scores (self-attention, cross-attention) from all layers
        """
        scores = []
        for layer in self.layers:
            scores.append(layer.get_attention_scores())
        return scores


class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TransformerEmbeddings, self).__init__()
        self.lookup = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        x: torch.Tensor, shape [batch_size, seq_len] of int64 in range [0, vocab_size)
        return torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.lookup(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].requires_grad_(False)
        x = x + pos  # Add the position encoding to original vector x
        return self.dropout(x)


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        padding_idx: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
        no_src_pos: bool = False,
        no_tgt_pos: bool = False,
    ):
        super(TransformerEncoderDecoder, self).__init__()
        """
        src_vocab_size: int, the size of the source vocabulary
        tgt_vocab_size: int, the size of the target vocabulary
        padding_idx: int, the index of the pad token
        num_layers: int, the number of layers
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        is_pre_layer_norm: bool, whether to apply layer norm before or after each sublayer
        no_src_pos: bool, whether to skip positional encoding for the source
        no_tgt_pos: bool, whether to skip positional encoding for the target
        """
        self.temporary = tgt_vocab_size
        self.src_embed = TransformerEmbeddings(src_vocab_size, d_model)
        if no_src_pos:
            self.src_pe = None
            print("Warning: no positional encoding for the source")
        else:
            self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_embed = TransformerEmbeddings(tgt_vocab_size, d_model)
        if no_tgt_pos:
            self.tgt_pe = None
            print("Warning: no positional encoding for the target")
        else:
            self.tgt_pe = PositionalEncoding(d_model, dropout)

        self.encoder = TransformerEncoder(
            num_layers,
            d_model,
            d_ff,
            num_heads,
            dropout,
            atten_dropout,
            is_pre_layer_norm,
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            num_layers,
            d_model,
            d_ff,
            num_heads,
            dropout,
            atten_dropout,
            is_pre_layer_norm,
        )

        self.padding_idx = padding_idx

    def create_pad_mask(self, tokens): #2.4.1
        """
        Create a padding mask using pad_idx (an attribute of the class)

        tokens: torch.Tensor, [batch_size, seq_len]
        return: torch.Tensor, [batch_size, 1, seq_len] where True means to mask, and on the same device as tokens
        """
        pad_mask = (tokens == self.padding_idx)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask.to(tokens.device)
        return pad_mask

    @staticmethod
    def create_causal_mask(tokens): #2.4.2
        """
        Create a causal (upper) triangular mask

        tokens: torch.Tensor, [batch_size, seq_len]
        pad_idx: int, the index of the pad token
        return: torch.Tensor, [1, seq_len, seq_len] where True means to mask, and on the same device as tokens
            If seq_len = 5, then the mask should look like:
                tensor([[[False,  True,  True,  True,  True],
                         [False, False,  True,  True,  True],
                         [False, False, False,  True,  True],
                         [False, False, False, False,  True],
                         [False, False, False, False, False]]])
            and make sure to set the correct dtype and device
        """
        seq_len = tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        causal_mask = causal_mask.bool()
        causal_mask = causal_mask.to(tokens.device)
        causal_mask = causal_mask.unsqueeze(0)

        return causal_mask

    def get_src_embeddings(self, src):
        """
        Get the non-contextualized source embeddings
        src: torch.Tensor, [batch_size, src_seq_len]
        return: torch.Tensor, [batch_size, src_seq_len, d_model]
        """
        src_x = self.src_embed(src)
        if self.src_pe is not None:
            src_x = self.src_pe(src_x)
        return src_x

    def get_tgt_embeddings(self, tgt):
        """
        Get the non-contextualized target embeddings
        tgt: torch.Tensor, [batch_size, tgt_seq_len]
        return: torch.Tensor, [batch_size, tgt_seq_len, d_model]
        """
        tgt_x = self.tgt_embed(tgt)
        if self.tgt_pe is not None:
            tgt_x = self.tgt_pe(tgt_x)
        return tgt_x

    def forward(self, src, tgt, normalize_logits: bool = False): #2.5
        """
        src: torch.Tensor, [batch_size, src_seq_len]
        tgt: torch.Tensor, [batch_size, tgt_seq_len]
        normalize_logits: bool, whether to apply log_softmax to the logits
        return: torch.Tensor, [batch_size, tgt_seq_len, tgt_vocab_size] of logits or log probabilities
        """
        # add create_pad_mask
        #add create_causal_mask
        src_mask = self.create_pad_mask(src)
        tgt_mask = self.create_causal_mask(tgt)

        src_embed = self.get_src_embeddings(src)
        tgt_embed = self.get_tgt_embeddings(tgt)

        encoder_output = self.encoder(src_embed, src_mask)
        decoder_output = self.decoder(tgt_embed, tgt_mask, encoder_output, src_mask, normalize_logits=normalize_logits)
        return decoder_output

    # decoding methods
    @staticmethod
    def all_finished(current_generation, eos_token, max_len=100):
        """
        Check if all the current generation is finished
        current_generation: torch.Tensor, [batch_size, seq_len]
        eos_token: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: bool, True if all the current generation is finished
        """
        return (
            torch.all(torch.any(current_generation == eos_token, dim=-1), dim=0).item()
            or current_generation.shape[-1] >= max_len
        )

    @staticmethod
    def initialize_generation_sequence(src, target_sos):
        """
        Initialize the generation by returning the initial input for the decoder.
        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        return: torch.Tensor, [batch_size, 1] on the same device as src and of time int64 filled with target_sos
        """
        return (
            torch.zeros(src.shape[0], 1).fill_(target_sos).type_as(src).to(src.device)
        )

    @staticmethod
    def concatenate_generation_sequence(tgt_generation, next_token): #3.1.2
        """
        Concatenate the next token to the current generation
        tgt_generation: torch.Tensor, [batch_size, seq_len]
        next_token: torch.Tensor, [batch_size, 1]
        return: torch.Tensor, [batch_size, seq_len + 1]
        """
        concatenated_sequence = torch.cat([tgt_generation, next_token], dim=-1)
        return concatenated_sequence

    def pad_generation_sequence(self, tgt_generation, target_eos):
        """
        Replace the generation past the end of sequence token with the end of sequence token.

        This is useful for:
            1) finalizing the generation for greedy decoding
            2) helping with intermediate steps in beam search

        tgt_generation: torch.Tensor, [batch_size, seq_len] or [batch_size, k * k, seq_len]
        target_eos: int, the end of sequence token
        return: torch.Tensor, [batch_size, seq_len] or [batch_size, k * k, seq_len]
        """
        lengths = (
            tgt_generation == target_eos
        )  # [batch_size, seq_len] or [batch_size, k * k, seq_len]
        # deal with case where eos was not found  [batch_size, seq_len + 1] or [batch_size, k * k, seq_len + 1]

        lengths = torch.cat(
            [lengths, torch.ones(*lengths.shape[:-1], 1).bool().to(lengths.device)],
            dim=-1,
        )
        # find the first eos token
        a = (
            torch.arange(lengths.shape[-1], device=tgt_generation.device)
            .unsqueeze(0)
            .int()
        )  # [1, seq_len + 1]
        if len(lengths.shape) == 3:
            a = a.unsqueeze(1)
        lengths = (
            torch.where(lengths, lengths * a, torch.ones_like(lengths) * torch.inf)
        ).min(dim=-1)[0]
        # replace tokens past the eos token with the padding token
        mask = a[..., :-1] > lengths.unsqueeze(
            -1
        )  # [batch_size, seq_len] or [batch_size, k * k, seq_len]
        return tgt_generation.masked_fill(mask, self.padding_idx)

    def greedy_decode(self, src, target_sos, target_eos, max_len=100):#3.1
        """
        Do not call the encoder more than once, or you will lose marks.
        The model calls must be batched and the only loop should be over the sequence length, or you will lose marks.
        It will also make evaluating the debugging the model difficult if you do not follow these instructions.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, seq_len]
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
        """
        curr_gen = self.initialize_generation_sequence(src, target_sos)
        src_mask = self.create_pad_mask(src)
        src_embed = self.get_src_embeddings(src)
        encoder_output = self.encoder(src_embed, src_mask)
        
        for i in range(max_len):
            tgt_mask = self.create_causal_mask(curr_gen)
            tgt_embed = self.get_tgt_embeddings(curr_gen)
            decoder_output = self.decoder(tgt_embed, tgt_mask, encoder_output, src_mask)
            tok = decoder_output[:, -1:, :].argmax(dim=-1, keepdim=True).squeeze(1)
            curr_gen = self.concatenate_generation_sequence(curr_gen, tok)
            if self.all_finished(curr_gen, target_eos, max_len):
                break
        curr_gen = self.pad_generation_sequence(curr_gen, target_eos)

        return curr_gen

    @staticmethod
    def expand_encoder_for_beam_search(src_x, src_mask, k): #3.2.2
        """
        Beamsearch will process `batches` of size `batch_size * k` so we need to expand the encoder outputs
        so that we can process the beams in parallel.

        Expand the encoder outputs for beam search to be of size [batch_size * k, ...]
        src_x: torch.Tensor, [batch_size, src_seq_len, d_model]
        src_mask: torch.Tensor, [batch_size, 1, src_seq_len]
        k: int, the beam size
        return: torch.Tensor, [batch_size * k, src_seq_len, d_model], [batch_size * k, 1, src_seq_len]
        """
        batch_size, src_seq_len, d_model = src_x.size()
        expanded_encoder_output = src_x.unsqueeze(1).repeat(1, k, 1, 1)
        expanded_encoder_output = expanded_encoder_output.view(batch_size * k, src_seq_len, d_model)
        
        expanded_encoder_output_mask = src_mask.repeat(1, k, 1)
        expanded_encoder_output_mask = expanded_encoder_output_mask.view(batch_size * k, 1, src_seq_len)
        
        return expanded_encoder_output, expanded_encoder_output_mask

    @staticmethod
    def repeat_and_reshape_for_beam_search(t, k, expan, batch_size): #3.2.3
        """
        Repeat the tensor for beam search expan times to be of size [batch_size * k, expan, cur_len]
        and then reshape to [batch_size, k * expan, cur_len]

        t: torch.Tensor, [batch_size * k, cur_len]
        k: int, the beam size
        expan: int, the expansion size
        batch_size: int, the batch size
        return: torch.Tensor, [batch_size, k * expan, cur_len]
        """
        cur_len = t.shape[1]
        t = t.unsqueeze(1)  # Add a singleton dimension at position 1
        t_expanded = t.expand(-1, expan, -1)  # Expand the tensor along dimension 1
        t_new = t_expanded.reshape(batch_size, k * expan, cur_len)  # Reshape the tensor
        return t_new

    def initialize_beams_for_beam_search(
        self, src, target_sos, target_eos, max_len=100, k=5
    ):#3.2.1
        """
        This function will initialize the beam search by taking the first decoder step and 
        using the top-k outputs to initialize the beams.

        Here we want to end up with a tensor of shape [batch_size * k, 2] for the input token
        sequence and a tensor of shape [batch_size * k, 2] of log probabilities of the sequences.
        2 is the sequence dimension and is 2 because of the sos token and the first real token.

        This involves the following steps:

            1) Initializes the input sequence with the start of sequence token  with shape 
            [batch_size, 1]
            2) Takes the first step of the decoder and the get the log probabilities, 
            [batch_size, 1, vocab_size]
            3) Please ensure that the end of sequence token is not predicted in the first step.
            4) Gets the top-k predictions, [batch_size, k]
            5) Initializes the log probabilities of the sequences, [batch_size * k, 1]
            6) Creates the beam tensors with the top-k predictions and log probabilities, 
            [batch_size  * k, 2]
               (i.e., two tensors with this shape).
            7) Expands the encoder outputs for beam search to be of size [batch_size * k, ...]

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size * k, 2], the token sequences
                torch.Tensor, [batch_size * k, 2], the log probabilities of the sequences
                torch.Tensor, [batch_size * k, src_seq_len, d_model], the expanded encoder outputs
                torch.Tensor, [batch_size * k, 1, src_seq_len], the expanded encoder mask
        """        
        
        tgt_generation = self.initialize_generation_sequence(src, target_sos) #[batch_size, 1]
        src_mask = self.create_pad_mask(src)
        src_embed = self.get_src_embeddings(src)
        encoder_output = self.encoder(src_embed, src_mask)
        tgt_mask = self.create_causal_mask(tgt_generation)
        tgt_embed = self.get_tgt_embeddings(tgt_generation)
        probabilities = self.decoder(tgt_embed, tgt_mask, encoder_output, src_mask) #[batch_size, tgt_seq_len, tgt_vocab_size]
        probabilities = probabilities[:, -1, :]
        probabilities[:, target_eos] = float('-inf')
        topk_logits, topk_toks = torch.topk(probabilities, k, dim=-1) #[batch_size, k]
        batch_size = src.shape[0]
        
        topk_toks = topk_toks.view(-1, 1) #reshape [batch_size * k, 1]
        tgt_generation = torch.full((batch_size * k, 1), target_sos).to(src.device)
        tgt_generation = torch.cat((tgt_generation, topk_toks), dim=1)
        
        seq_prob = torch.zeros(batch_size * k, 1).to(src.device)
        topk_logits = topk_logits.view(-1, 1) #reshape [batch_size * k, 1]
        seq_prob = torch.cat((seq_prob, topk_logits), dim=1)#add new logits topk_logits
        
        
        expanded_encoder_output, expanded_encoder_output_mask = self.expand_encoder_for_beam_search(encoder_output, src_mask, k)
        
        return tgt_generation, seq_prob, expanded_encoder_output, expanded_encoder_output_mask

    def pad_and_score_sequence_for_beam_search(
        self, tgt_generation, seq_log_probs, target_eos
    ): #3.2.4
        """
        This function will pad the sequences with eos and seq_log_probs with corresponding zeros.
        It will then get the score of each sequence by summing the log probabilities.

        Note assume that we want this to work for generic shapes of the input where the last dimension is the sequence.

        tgt_generation: torch.Tensor, [..., seq_len]
        seq_log_probs: torch.Tensor, [..., seq_len]
        target_eos: int, the end of sequence token
        return: torch.Tensor, [..., seq_len], the padded token sequences
                torch.Tensor, [...., seq_len], the log probabilities of the sequences
                torch.Tensor, [...], the summed scores of the sequences
        """
        
        # tgt_generation = self.pad_generation_sequence(tgt_generation, target_eos)
        # mask = tgt_generation == self.padding_idx
        # tgt_generation = tgt_generation.masked_fill(mask, target_eos)
        # seq_log_probs = seq_log_probs.masked_fill(mask, 0)
        scores = seq_log_probs.sum(dim=-1)
        return tgt_generation, seq_log_probs, scores

    def finalize_beams_for_beam_search(self, top_beams, device):
        """
        This function will take a list of top beams of length batch_size, where each element is a tensor of some length
        and return a padded tensor of the top beams.  Use self.padding_idx for the padding.

        top_beams: list of torch.Tensor, each of shape [seq_len_i]
        device: torch.device, the device to put the tensor on
        return: torch.Tensor, [batch_size, max_seq_len]
        """
        max_seq_len = max(len(t) for t in top_beams)
        batch_size = len(top_beams)
        all_tensors_list = []

        for t in top_beams:
            seq_len_i = len(t)
            num_values = max_seq_len - seq_len_i

            if seq_len_i < max_seq_len:
                padding_tensor = torch.full((num_values,), self.padding_idx, device=device)
                t = t.to(device)
                final_tensor = torch.cat((t, padding_tensor), dim=0)
            else:
                final_tensor = t
            all_tensors_list.append(final_tensor)

        # Stack the tensors along the 0th dimension to create the final tensor
        concatenated_tensor = torch.stack(all_tensors_list, dim=0)
        return concatenated_tensor


    def beam_search_decode(self, src, target_sos, target_eos, max_len=100, k=5):
        """
        This processes the batch in parallel.  Finished beams are removed from the batch and not processed further.

        This needs to keep track of a list of finished candidates along with currently alive beams to prevent
        the search from stopping prematurely with duplicate beams.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, max_seq_len] of the mostly likely beam
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
        """

        # get the initial first predictions and initialize the beams with them
        batch_size = src.shape[0]  # original batch size
        (
            tgt_generation,
            seq_log_probs,
            src_x,
            src_mask,
        ) = self.initialize_beams_for_beam_search(
            src, target_sos, target_eos, max_len, k
        )

        finished_sequences = [[] for _ in range(batch_size)]
        finished_scores = [
            [] for _ in range(batch_size)
        ]  # averaged log probs i.e score
        is_finished = [False] * batch_size

        # use to keep track of which sequence in batch are still being processed
        cur_sentence_ids = torch.arange(batch_size, device=src.device)  # [batch_size]
        cur_batch_size = batch_size

        # expansion size for each beam, 2 so that we can possibly get k finished sequences and k alive
        # this is used in place of vocab_size for efficiency
        expan = 2 * k
        while not all(is_finished) and tgt_generation.shape[-1] < max_len:
            tgt_mask = self.create_causal_mask(tgt_generation)
            tgt_x = self.get_tgt_embeddings(tgt_generation)
            
            log_probs = self.decoder(
                tgt_x, tgt_mask, src_x, src_mask, normalize_logits=True
            )
            log_probs, pred = torch.topk(
                log_probs[:, -1, :], expan, dim=1
            )  # [batch_size * k, expan]

            # reshape to cur to [batch_size, k * expan, cur_len] and new to [batch_size, k * expan, 1]
            tgt_generation = self.repeat_and_reshape_for_beam_search(
                tgt_generation, k, expan, cur_batch_size
            )
            seq_log_probs = self.repeat_and_reshape_for_beam_search(
                seq_log_probs, k, expan, cur_batch_size
            )
            log_probs = log_probs.reshape(cur_batch_size, k * expan, 1)
            pred = pred.reshape(cur_batch_size, k * expan, 1)

            # expansion to [batch_size, k * expan, cur_len + 1]
            tgt_generation = self.concatenate_generation_sequence(tgt_generation, pred)
            seq_log_probs = torch.cat(
                [seq_log_probs, log_probs], dim=-1
            )  # or concatenate_generation_sequence

            #  masking to length and score by summing
            (
                tgt_generation,
                seq_log_probs,
                scores,
            ) = self.pad_and_score_sequence_for_beam_search(
                tgt_generation, seq_log_probs, target_eos
            )

            # sort and get top k of the current expan candidates
            scores, topk = torch.topk(scores, expan, dim=1)  # [batch_size, k * expan]

            # gather the top expan candidates sorted by score, [batch_size, expan, cur_len + 1]
            tgt_generation = torch.gather(
                tgt_generation,
                1,
                topk.unsqueeze(-1).expand(-1, -1, tgt_generation.shape[-1]),
            )
            seq_log_probs = torch.gather(
                seq_log_probs,
                1,
                topk.unsqueeze(-1).expand(-1, -1, seq_log_probs.shape[-1]),
            )

            # split into finished and alive
            has_eos = (tgt_generation == target_eos).any(dim=-1)  # [batch_size, expan]
            alive_idxs = []
            has_not_finished_idxs = []  # otherwise remove from batch for efficiency
            for b in range(cur_batch_size):
                original_b = cur_sentence_ids[b].item()  # original sentence id
                alive_idxs_b = []
                for beam in range(expan):
                    if has_eos[b, beam]:
                        finished_sequences[original_b].append(
                            tgt_generation[b, beam, ...].cpu()
                        )
                        finished_scores[original_b].append(scores[b, beam].item())
                    else:
                        alive_idxs_b.append(beam)

                if len(finished_sequences[original_b]) > 1:  # sort finished sequences
                    z = list(
                        zip(finished_scores[original_b], finished_sequences[original_b])
                    )
                    z = sorted(z, key=lambda x: x[0], reverse=True)
                    finished_scores[original_b] = [
                        x[0] for x in z[:k]
                    ]  # cut to k if more than k finished
                    finished_sequences[original_b] = [x[1] for x in z[:k]]

                # these conditions only work assuming an monotonic increase in score (length normalization breaks it)
                has_finished = False
                if len(
                    finished_sequences[original_b]
                ):  # if the most probable finished is more probable than top alive
                    best_finished = max(finished_scores[original_b])
                    if best_finished > scores[b, alive_idxs_b[0]]:
                        has_finished = True
                elif (
                    len(finished_sequences[original_b]) == k
                ):
                    worst_finished = min(finished_scores[original_b])
                    if worst_finished > scores[b, alive_idxs_b[0]]:
                        has_finished = True
                if not has_finished:
                    has_not_finished_idxs.append(b)
                    alive_idxs.append(alive_idxs_b[:k])

            if len(has_not_finished_idxs) == 0:  # all finished
                break

            # gather sequences that still need decoding, this is done for efficiency
            has_not_finished_idxs = torch.tensor(
                has_not_finished_idxs, dtype=torch.long, device=src.device
            )
            tgt_generation = torch.index_select(
                tgt_generation, 0, has_not_finished_idxs
            )
            seq_log_probs = torch.index_select(seq_log_probs, 0, has_not_finished_idxs)
            src_shape = src_x.shape
            src_x = src_x.reshape(cur_batch_size, k, src_shape[-2], src_shape[-1])
            src_mask = src_mask.reshape(cur_batch_size, k, -1)
            src_x = torch.index_select(src_x, 0, has_not_finished_idxs)
            src_mask = torch.index_select(src_mask, 0, has_not_finished_idxs)
            cur_batch_size = len(has_not_finished_idxs)
            src_x = src_x.reshape(cur_batch_size * k, src_shape[-2], src_shape[-1])
            src_mask = src_mask.reshape(cur_batch_size * k, 1, src_shape[-2])
            cur_sentence_ids = torch.index_select(
                cur_sentence_ids, 0, has_not_finished_idxs
            )

            # gather the top k alive beams
            alive_idxs = torch.tensor(
                alive_idxs, dtype=torch.long, device=tgt_generation.device
            )

            tgt_generation = torch.gather(
                tgt_generation,
                1,
                alive_idxs.unsqueeze(-1).expand(-1, -1, tgt_generation.shape[-1]),
            )
            seq_log_probs = torch.gather(
                seq_log_probs,
                1,
                alive_idxs.unsqueeze(-1).expand(-1, -1, seq_log_probs.shape[-1]),
            )

            # reshape to [batch_size * k, cur_len]
            tgt_generation = tgt_generation.reshape(cur_batch_size * k, -1)
            seq_log_probs = seq_log_probs.reshape(cur_batch_size * k, -1)

        if not all(is_finished):  # take top alive if no finished
            tgt_generation = tgt_generation.reshape(cur_batch_size, k, -1)
            for b in range(cur_batch_size):
                original_b = cur_sentence_ids[b].item()
                if len(finished_sequences[original_b]) == 0:  # just take top alive
                    g = tgt_generation[b, 0, ...]
                    finished_sequences[original_b].append(g.cpu())

        return self.finalize_beams_for_beam_search(
            [x[0] for x in finished_sequences], src_x.device
        )

    def beam_search_decode_slow(self, src, target_sos, target_eos, max_len=100, k=5):
        """
        Slow version which does not batch the beam search. This is useful for confirming debugging and understanding.
        Consider the summed log probability of the sequence (including eos) when scoring and sorting the candidates.

        This needs to keep track of a list of finished candidates along with currently alive beams to prevent
        the search from stopping prematurely with duplicate beams.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, max_seq_len] of the mostly likely beam
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
        """

        # print('Warning: this is a slow version of beam search and should only be run for debugging.')
        # get the initial first predictions and initialize the beams with them
        batch_size = src.shape[0]  # original batch size

        (
            tgt_generation,
            seq_log_probs,
            src_x,
            src_mask,
        ) = self.initialize_beams_for_beam_search(
            src, target_sos, target_eos, max_len, k
        )

        # reshape so we can loop over the batch dimension
        tgt_generation = tgt_generation.reshape(batch_size, k, -1)
        seq_log_probs = seq_log_probs.reshape(batch_size, k, -1)
        src_x = src_x.reshape(batch_size, k, src_x.shape[-2], src_x.shape[-1])
        src_mask = src_mask.reshape(
            batch_size, k, 1, -1
        )  # note the extra dim for the mask

        top_beams = []
        expan = 2 * k  # expansion size, 2 * k for k finished and k alive,
        # this is used in place of vocab_size for efficiency
        for b in range(batch_size):
            # slice out the current sequence from the batch
            tgt_generation_b = tgt_generation[b, ...]
            seq_log_probs_b = seq_log_probs[b, ...]
            src_x_b = src_x[b, ...]
            src_mask_b = src_mask[b, ...]

            finished_sequences_b = []
            finished_scores_b = []

            for i in range(1, max_len):
                tgt_mask = self.create_causal_mask(tgt_generation_b)
                tgt_x = self.get_tgt_embeddings(tgt_generation_b)
                log_probs = self.decoder(
                    tgt_x, tgt_mask, src_x_b, src_mask_b, normalize_logits=True
                )
                log_probs, pred = torch.topk(log_probs[:, -1, :], expan, dim=1)

                # reshape to cur to [k * expan, cur_len] and new to [k * expan, 1]
                tgt_generation_b = self.repeat_and_reshape_for_beam_search(
                    tgt_generation_b, k, expan, 1
                ).squeeze(0)
                seq_log_probs_b = self.repeat_and_reshape_for_beam_search(
                    seq_log_probs_b, k, expan, 1
                ).squeeze(0)
                log_probs = log_probs.reshape(k * expan, 1)
                pred = pred.reshape(k * expan, 1)

                # expansion to [batch_size, k * expan, cur_len + 1]
                tgt_generation_b = self.concatenate_generation_sequence(
                    tgt_generation_b, pred
                )
                seq_log_probs_b = torch.cat([seq_log_probs_b, log_probs], dim=-1)

                # masking to length and score by summing
                (
                    tgt_generation_b,
                    seq_log_probs_b,
                    scores,
                ) = self.pad_and_score_sequence_for_beam_search(
                    tgt_generation_b, seq_log_probs_b, target_eos
                )

                # sort and get top k of the current expan candidates
                scores, topk = torch.topk(scores, expan, dim=0)  # [k * expan]

                # gather the top expan candidates sorted by score, [expan, cur_len + 1]
                tgt_generation_b = torch.index_select(tgt_generation_b, 0, topk)
                seq_log_probs_b = torch.index_select(seq_log_probs_b, 0, topk)

                # split into finished and alive and determine if we are finished
                has_eos = (tgt_generation_b == target_eos).any(dim=-1)  # [expan]
                alive_idxs_b = []
                for beam in range(expan):
                    if has_eos[beam]:
                        finished_sequences_b.append(tgt_generation_b[beam, ...].cpu())
                        finished_scores_b.append(scores[beam].item())
                    else:
                        alive_idxs_b.append(beam)
                if len(finished_sequences_b) > 1:  # sort finished sequences
                    z = list(zip(finished_scores_b, finished_sequences_b))
                    z = sorted(z, key=lambda x: x[0], reverse=True)
                    finished_scores_b = [
                        x[0] for x in z[:k]
                    ]  # cut to k if more than k finished
                    finished_sequences_b = [x[1] for x in z[:k]]

                # these conditions only work assuming an monotonic increase in score (length normalization breaks it)
                if len(
                    finished_sequences_b
                ):  # finished if most probable finished is more probable than top alive
                    best_finished = max(finished_scores_b)
                    if best_finished > scores[alive_idxs_b[0]]:
                        break
                elif (
                    len(finished_sequences_b) == k
                ):  # finished if all finished more probable than top alive
                    worst_finished = min(finished_scores_b)
                    if worst_finished > scores[alive_idxs_b[0]]:
                        break

                # gather the top k alive beams
                alive_idxs_b = torch.tensor(
                    alive_idxs_b[:k], dtype=torch.long, device=tgt_generation.device
                )
                tgt_generation_b = torch.index_select(tgt_generation_b, 0, alive_idxs_b)
                seq_log_probs_b = torch.index_select(seq_log_probs_b, 0, alive_idxs_b)

            if len(finished_sequences_b) == 0:  # take top alive if no finished
                finished_sequences_b.append(tgt_generation_b[0, ...].cpu())
            top_beams.append(finished_sequences_b[0])

        return self.finalize_beams_for_beam_search(top_beams, src_x.device)

    def store_attention_scores(self, should_store=True):
        self.decoder.store_attention_scores(should_store)

    def get_attention_scores(self):
        return self.decoder.get_attention_scores()