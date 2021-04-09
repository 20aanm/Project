# -*- coding: utf-8 -*-

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf


def bleu(reference, predict):
    """Compute sentence-level bleu score.

    Args:
        reference (list[str])
        predict (list[str])
    """

    if len(predict) == 0:
        if len(reference) == 0:
            return 1.0
        else:
            return 0.0

    # TODO(kelvin): is this quite right?
    # use a maximum of 4-grams. If 4-grams aren't present, use only lower n-grams.
    n = min(4, len(reference), len(predict))
    weights = tuple([1. / n] * n)  # uniform weight on n-gram precisions
    return sentence_bleu([reference], predict, weights) 


def chrf(reference, predict):
    """Compute sentence-level bleu score.

    Args:
        reference (list[str])
        predict (list[str])
    """

    if len(predict) == 0:
        if len(reference) == 0:
            return 1.0
        else:
            return 0.0

    # TODO(kelvin): is this quite right?
    # use a maximum of 4-grams. If 4-grams aren't present, use only lower n-grams.
    return sentence_chrf(reference,predict,1,4) 


print("\n____________________________________________\n")
score_blue_GT=bleu("ref_file(GT)","predict_file(GT)")
score_chrf_GT=chrf("ref_file(GT)","predict_file(GT)")
print("GraphTransformer blue score",score_blue_GT)
print("GraphTransformer chrf score",score_chrf_GT)
print("\n____________________________________________\n")

score_blue_DCGCN=bleu("Reference(DCGCN)","Prediction(DCGCN)")
score_chrf_DCGCN=chrf("Reference(DCGCN)","Prediction(DCGCN)")
print("DCGCN blue score",score_blue_DCGCN)
print("DCGCN chrf score",score_chrf_DCGCN)
print("\n____________________________________________\n")
score_blue_LDGCN=bleu("Reference2(LDGCN)","Prediction2(LDGCN)")
score_chrf_LDGCN=chrf("Reference2(LDGCN)","Prediction2(LDGCN)")
print("LDGCN blue score",score_blue_LDGCN)
print("LDGCN chrf score",score_chrf_LDGCN)
print("\n____________________________________________\n")
