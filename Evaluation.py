# -*- coding: utf-8 -*-

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate import bleu_score
import meteor_score
import tensorflow as tf


def eval_score(file_ref,file_pred, bleu_baseline=False):
  """Read the dataset"""
  print("Reading raw data .. ")
  print("data path: %s" % file_ref)
  sentence_ref = []
  sentence_pred=[]
  with tf.gfile.GFile(file_ref, mode="r") as ref:
      for line in ref:
          sentence_ref.append(line.lower().rstrip("\n"))
  print("data path: %s" % file_pred)
  with tf.gfile.GFile(file_pred, mode="r") as pred:
        for line in pred:
          sentence_pred.append(line.lower().rstrip("\n"))

  #print(sentence_ref)  
  #print(sentence_pred)  
  if(bleu_baseline):
      print("calculating scores ... ")
      hypothesis = [s for s in sentence_pred]
      references = [s for s in sentence_ref]
      #print(hypothesis)
      #print(references)
      bleu = corpus_bleu(
            references, hypothesis,
            smoothing_function=bleu_score.SmoothingFunction().method1) * 100
      print("bleu: %.2f" % (bleu*100))
      print("chrf: %.2f"%(chrf(references, hypothesis)*100))
      print("meteor: %.2f"%(meteor_score.meteor_score(references,str(hypothesis))*100))


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
    return sentence_chrf(reference,predict)




print("\n____________________________________________\n")
eval_score("results/ref_file(GT)","results/predict_file(GT)",True)
print("\n____________________________________________\n")


print("\n--------------------16 gcn hidden layers-------------------------\n")
eval_score("results/Reference(DCGCN)","results/Prediction(DCGCN)",True)
print("\n____________________________________________\n")
eval_score("results/Reference2(LDGCN)","results/Prediction2(LDGCN)",True)
print("\n____________________________________________\n")



#128 hidden layers
#number of para. is equal to 
print("\n--------------------128 gcn hidden layers-------------------------\n")
eval_score("results/Ref(LDGCN)128","results/Pred(LDGCN)128",True)
print("\n____________________________________________\n")

#number of para. is equal to 26488858
eval_score("results/Ref(DCGCN)128","results/Pred(DCGCN)128",True)

print("\n____________________________________________\n")



print("\n____________________________________________\n")
print("DCGCN: 1000 epochs")
print("16 gcn hidden layers")
eval_score("results/Reference(DCGCN)_1000","results/Prediction(DCGCN)_1000",True)


print("\n____________________________________________\n")
print("LDGC: 1000 epochs")
print("16 gcn hidden layers")
eval_score("results/Reference(LDGCN)_1000","results/Prediction(LDGCN)_1000",True)

print("\n____________________________________________\n")
print("GraphTransformer: 1000 epochs")
eval_score("results/ref_file(GT)_1000","results/predict_file(GT)_1000",True)
print("\n____________________________________________\n")



print("\n--------------------256 gcn hidden layers-------------------------\n")
eval_score("results/Reference2(DCGCN)_256","results/Prediction2(DCGCN)_256",True)
print("\n____________________________________________\n")
eval_score("results/Reference2(LDGCN)_256","results/Prediction2(LDGCN)_256",True)
print("\n____________________________________________\n")


print("\n--------------------480 gcn hidden layers-------------------------\n")
eval_score("results/Reference2(DCGCN)_480","results/Prediction2(DCGCN)_480",True)
print("\n____________________________________________\n")
eval_score("results/Reference2(LDGCN)_480","results/Prediction2(LDGCN)_480",True)
print("\n____________________________________________\n")
