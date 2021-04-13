from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
#from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import shutil
import hashlib
from sys import platform
import ReadingData
from ReadingData import *
import argparse
import copy
import collections
from gensim.models import KeyedVectors
from DCGCN import DCGCN
import json
import warnings
#warnings.filterwarnings('ignore')
FLAGS = None
# tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="model/", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="output/", help="Out directory")
    parser.add_argument("--train_dir", type=str, default="model/graph/", help="Training directory")
    parser.add_argument("--gpu_device", type=str, default="0", help="which gpu to use")

    parser.add_argument("--train_data", type=str, default="training",
                        help="Training data path")

    parser.add_argument("--valid_data", type=str, default="dev",
                        help="Valid data path")

    parser.add_argument("--test_data", type=str, default="test",
                        help="Test data path")

    parser.add_argument("--from_vocab", type=str, default="data/vocab_15000",
                        help="from vocab path")
    parser.add_argument("--to_vocab", type=str, default="data/vocab_15000",
                        help="to vocab path")
    parser.add_argument("--label_vocab", type=str, default="data/evocab",
                        help="label vocab path")
    parser.add_argument("--output_dir", type=str, default="model/graph/")


    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit)")

    parser.add_argument("--from_vocab_size", type=int, default=15000, help="source vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=15000, help="target vocabulary size")
    parser.add_argument("--edge_vocab_size", type=int, default=150, help="edge label vocabulary size")
    parser.add_argument("--enc_layers", type=int, default=10, help="Number of layers in the encoder")
    parser.add_argument("--dec_layers", type=int, default=4, help="Number of layers in the decoder")
    parser.add_argument("--num_units", type=int, default=256, help="Size of each model layer")
    parser.add_argument("--num_heads", type=int, default=8, help="Size of each model layer")
    parser.add_argument("--gcn_encoding", type=int, default=True, help="Using GCN encoding")
    parser.add_argument("--gcn_num_hidden", type=int, default=480, help="Number of hidden layers of GCN")
    parser.add_argument("--gcn_num_layers", type=int, default=4, help="Number of GCN layers")
    parser.add_argument("--gcn_dropout", type=float, default=0.5, help="GCN dropout")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of word embedding")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=3.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--max_src_len", type=int, default=100, help="Maximum length of source ordering")
    parser.add_argument("--max_tgt_len", type=int, default=100, help="Maximum length of target ordering")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--use_copy", type=int, default=True, help="Whether use copy mechanism")
    parser.add_argument("--use_depth", type=int, default=False, help="Whether use depth embedding")
    parser.add_argument("--use_charlstm", type=int, default=False, help="Whether use character embedding")
    parser.add_argument("--input_keep_prob", type=float, default=1.0, help="Dropout input keep prob")
    parser.add_argument("--output_keep_prob", type=float, default=0.9, help="Dropout output keep prob")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epoch")
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda2", type=float, default=0.5)

def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        output_dir=flags.output_dir,

        # data params
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        edge_vocab_size=flags.edge_vocab_size,
        GO_ID=ReadingData.GO_ID,
        EOS_ID=ReadingData.EOS_ID,
        PAD_ID=ReadingData.PAD_ID,
        emb_dim=flags.emb_dim,
        max_train_data_size=flags.max_train_data_size,

        train_data=flags.train_data,
        valid_data=flags.valid_data,
        test_data=flags.test_data,

        from_vocab=flags.from_vocab,
        to_vocab=flags.to_vocab,
        label_vocab=flags.label_vocab,
        share_vocab=False,

        # model params
        use_copy=flags.use_copy,
        use_depth=flags.use_depth,
        use_charlstm=flags.use_charlstm,
        gcn_encoding=flags.gcn_encoding,
        gcn_num_hidden=flags.gcn_num_hidden,
        gcn_num_layers=flags.gcn_num_layers,
        gcn_dropout=flags.gcn_dropout,
        input_keep_prob=flags.input_keep_prob,
        output_keep_prob=flags.output_keep_prob,
        dropout_rate=flags.dropout_rate,
        init_weight=0.1,
        num_units=flags.num_units,
        num_heads=flags.num_heads,
        enc_layers=flags.enc_layers,
        dec_layers=flags.dec_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        max_src_len=flags.max_src_len,
        max_tgt_len=flags.max_tgt_len,
        max_seq_length=100,
        #train params
        epoch_num=flags.epoch_num,
        epoch_step=0,
        lambda1=flags.lambda1,
        lambda2=flags.lambda2
    )

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto


class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model"))):
  pass

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model"))):
  pass

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model"))):
  pass

def create_model(hparams, model, length=22):
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = model(hparams, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = model(hparams, tf.contrib.learn.ModeKeys.INFER)

    return TrainModel(graph=train_graph, model=train_model), EvalModel(graph=eval_graph, model=eval_model), InferModel(
        graph=infer_graph, model=infer_model)

def read_data(src_path, tgt_path, vocab):

    data_set = []
    with tf.gfile.GFile(src_path, mode="r") as src_file:
        with tf.gfile.GFile(tgt_path, mode="r") as tgt_file:
            src, tgt = src_file.readline(), tgt_file.readline()
            while src and tgt:
                src_ids = [int(x) for x in src.rstrip("\n").split(" ")]
                tgt_ids = [int(x) for x in tgt.rstrip("\n").split(" ")]

                pair = (src_ids, tgt_ids)
                data_set.append(pair)
                src, tgt = src_file.readline(), tgt_file.readline()
    return data_set

def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans
def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params
def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params
def train(hparams):

    embeddings = init_embedding(hparams)
    hparams.add_hparam(name="embeddings", value=embeddings)
    wvocab, wvocab_rev = initialize_vocabulary(hparams.from_vocab)
    evocab, evocab_rev = initialize_vocabulary(hparams.label_vocab)
    cvocab, cvocab_rev = initialize_vocabulary("data/cvocab")
    train_data, train_unks = read_data_graph("data/train.src", "data/train.edge",
                                             "data/train.tgt",
                                             wvocab, evocab, cvocab, hparams)
    valid_data, valid_unks = read_data_graph("data/valid.src", "data/valid.edge",
                                             "data/valid.tgt",
                                             wvocab, evocab, cvocab, hparams)


    train_model, eval_model, infer_model = create_model(hparams, DCGCN)
    config = get_config_proto(
        log_device_placement=False)
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)



    epoch = tf.train.get_checkpoint_state(hparams.train_dir)
    epoch_path = os.path.join(hparams.train_dir, "epoch")
    with train_model.graph.as_default():
        if epoch and tf.train.checkpoint_exists(epoch.model_checkpoint_path):
            print("Reading model parameters from %s" % epoch.model_checkpoint_path)
            train_model.model.saver.restore(train_sess, epoch.model_checkpoint_path)
            eval_model.model.saver.restore(eval_sess, epoch.model_checkpoint_path)
            infer_model.model.saver.restore(infer_sess, epoch.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)

        else:
            train_sess.run(tf.global_variables_initializer())
            global_step = 0
            print("Number of trainable parameters: %d" % count_number_trainable_params())
    step_loss_train, step_time_train, total_predict_count_train, total_loss_train, total_time_train, avg_loss_train, avg_time_train = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    step_loss_val, step_time_val, total_predict_count_val, total_loss_val, total_time_val, avg_loss_val, avg_time_val = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ppl_val = 0
    Early_stop = 10
    epoch_count = 0
    epoch_step = int((len(train_data) - 1) / hparams.batch_size) + 1
    now_step = 0
    random.shuffle(train_data)
    while global_step <= 1000:
        start_time = time.time()

        step_loss_train, global_step, predict_count_train = train_model.model.train_step(train_sess, train_data,
                                                                                         no_random=True,
                                                                                         id=now_step * hparams.batch_size)
        now_step += 1
        total_loss_train += step_loss_train
        total_time_train += (time.time() - start_time)
        total_predict_count_train += predict_count_train
        if global_step >= 1:
            ppl = safe_exp(total_loss_train / total_predict_count_train)
            avg_loss_train = total_loss_train / global_step
            avg_time_train = total_time_train / global_step
            total_loss_train, total_predict_count_train, total_time_train = 0.0, 0.0, 0.0
            print(
                "Epoch(%d)   step-time %.2fs  loss %.3f ppl %.2f" % (global_step, avg_time_train, avg_loss_train, ppl))
            train_ppl = ppl
            with open('train_epochs.txt', 'a', encoding='utf-8') as f:
                f.write("Epoch(%d)   step-time %.2fs  loss %.3f ppl %.2f" % (
                global_step, avg_time_train, avg_loss_train, ppl) + "\n")
            with open('train_loss.txt', 'a', encoding='utf-8') as f:
                f.write(str(avg_loss_train) + "\n")
            with open('train_ppl.txt', 'a', encoding='utf-8') as f:
                f.write(str(ppl) + "\n")
            total_loss_train, total_predict_count_train, total_time_train = 0.0, 0.0, 0.0
        if now_step == epoch_step:
            now_step = 0
            random.shuffle(train_data)
            train_model.model.saver.save(train_sess, epoch_path, global_step=global_step)
            epoch = tf.train.get_checkpoint_state(hparams.train_dir)
            if epoch and tf.train.checkpoint_exists(epoch.model_checkpoint_path):
                eval_model.model.saver.restore(eval_sess, epoch.model_checkpoint_path)
                print("loading model.........")
            else:
                raise ValueError("epoch file not found.")
            val_time = time.time()
            step_loss_val, predict_count_val = eval_model.model.eval_step(eval_sess, valid_data.copy(), no_random=True,
                                                                          id=now_step * hparams.batch_size)
            total_loss_val += step_loss_val
            total_predict_count_val += predict_count_val
            total_time_val += (time.time() - val_time)
            avg_loss_val = total_loss_val / global_step
            avg_time_val = total_time_val / global_step
            ppl = safe_exp(total_loss_val / total_predict_count_val)

            print("Validation: Epoch(%d) step-time %.2fs  loss %.3f ppl %.2f" % (
            global_step, avg_time_val, avg_loss_val, ppl))

            with open('val_epochs.txt', 'a', encoding='utf-8') as f:
                f.write("Epoch(%d)   step-time %.2fs  loss %.3f ppl %.2f" % (
                global_step, avg_time_val, avg_loss_val, ppl) + "\n")
            with open('val_loss.txt', 'a', encoding='utf-8') as f:
                f.write(str(avg_loss_val) + "\n")
            with open('val_ppl.txt', 'a', encoding='utf-8') as f:
                f.write(str(ppl) + "\n")
            if global_step == 1:
                ppl_val = ppl
                print(ppl_val)
            if ppl_val < ppl:
                epoch_count += 1
                ppl_val=ppl
                print("Earlystop count:",epoch_count)
            else:
                epoch_count=0
                ppl_val=ppl
            if epoch_count > Early_stop:
                print("Earlystop at epoch(" + str(global_step)+")")
                with open('train_epochs.txt', 'a', encoding='utf-8') as f:
                    f.write("Earlystop at epoch(" + str(global_step), +")" + "\n")
                with open('val_epochs.txt', 'a', encoding='utf-8') as f:
                    f.write("Earlystop at epoch(" + str(global_step), +")"+"\n")
                break
            print("\n___________________________________________\n")
            total_loss_val, total_predict_count_val, total_time_val = 0.0, 0.0, 0.0

def init_embedding(hparams):
    f = open(hparams.from_vocab, "r", encoding="utf-8")
    vocab = []
    for line in f:
        vocab.append(line.rstrip("\n"))

    word_vectors = KeyedVectors.load_word2vec_format("data/amr_vector.txt")

    emb = []
    num = 0
    for i in range(0, len(vocab)):
        word = vocab[i]
        if word in word_vectors:
            num += 1
            emb.append(word_vectors[word])
        else:
            emb.append((0.1 * np.random.random([hparams.emb_dim]) - 0.05).astype(np.float32))

    print(" init embedding finished")
    emb = np.array(emb)
    print(num)
    print(emb.shape)
    return emb

def main(_):

    hparams = create_hparams(FLAGS)
    print(hparams)
    train(hparams)

if __name__ == "__main__":
    print(FLAGS)
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    print(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run()
