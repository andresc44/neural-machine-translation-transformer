from math import exp  # exp(x) gives e^x
from collections.abc import Sequence


def grouper(seq: Sequence[str], n: int) -> list:
    """
    Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    #0, 1, 2, 3, 4
    #n == 3
    ngrams = []
    l = len(seq) #5
    if l >= n:
        last_n_start_idx = l-n #2
        for i in range(last_n_start_idx + 1):
            sub_seq = seq[i:i+n]
            ngrams.append(sub_seq)
    else:
        ngrams = list(seq)
    return ngrams


def n_gram_precision(
    reference: Sequence[str], candidate: Sequence[str], n: int
) -> float:
    """
    Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """
    ref_ngrams = grouper(reference, n)
    cand_ngrams = grouper(candidate, n)
    N_denom = len(cand_ngrams)
    C = 0
    if  len(candidate) < n:
        return 0.0
        
    if N_denom:
        for gram in cand_ngrams:
            if gram in ref_ngrams:
                C += 1
        prec = float(C) / float(N_denom)
    else:
        prec = 0
    return prec



def brevity_penalty(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """
    Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """
    c = len(candidate)
    r = len(reference)
    if c: brev = r / c
    else: return 0
    if brev < 1:
        BP = 1
    else:
        BP = exp(1-brev)
    return BP
    


def BLEU_score(reference: Sequence[str], candidate: Sequence[str], n) -> float:
    """
    Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    """
    mult_prob = 1
    for i in range(1, n+1):
        prob = n_gram_precision(reference, candidate, i)
        mult_prob *= prob
    exponent = 1.0 / float(n)
    BP_c = brevity_penalty(reference, candidate)
    bleu = 100 * BP_c * mult_prob ** exponent
    return bleu
