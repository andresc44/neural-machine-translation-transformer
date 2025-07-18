import pytest
import numpy as np
import a2_bleu_score


@pytest.mark.parametrize("ids", [True, False])
def test_bleu(ids):
    reference = """\
it is a guide to action that ensures that the military will always heed
party commands""".strip().split()
    candidate = """\
it is a guide to action which ensures that the military always obeys the
commands of the party""".strip().split()
    if ids:
        # should work with token ids (ints) as well
        reference = [hash(word) for word in reference]
        candidate = [hash(word) for word in candidate]

    ## Unigram precision
    p1_hat = a2_bleu_score.n_gram_precision(reference, candidate, 1)
    p1 = 15 / 18  # w/o capping
    assert np.isclose(p1_hat, p1)

    ## Bi-gram precision
    p2_hat = a2_bleu_score.n_gram_precision(reference, candidate, 2)
    p2 = 8 / 17
    assert np.isclose(p2_hat, p2)

    ## BP
    BP_hat = a2_bleu_score.brevity_penalty(reference, candidate)
    BP = 1.0
    assert np.isclose(BP_hat, BP)

    ## BLEU Score
    bleu_score_hat = a2_bleu_score.BLEU_score(reference, candidate, 2)
    bleu_score = (BP * (p1 * p2) ** (1 / 2)) * 100
    assert np.isclose(bleu_score_hat, bleu_score)
