from tensorflow.python.platform import gfile
import tensorflow as tf
import json
from gensim.models import KeyedVectors
import numpy as np
from tensorflow.python.layers import core as layers_core
import random
from tensorflow.python.ops import array_ops
from tensorflow.keras import Input, Model
from tensorboard.plugins.hparams import api as hp
from tf2_gnn.layers.gnn import GNN, GNNInput
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.math import segment_mean
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.optimizers import Adam, RMSprop,Nadam, Adagrad, Adamax,Adadelta
import networkx as nx
import math
import warnings
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import sent_tokenize
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

UNK_ID = 3
max_src_len=100
max_tgt_len=100


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def read_data_graph(src_path, edge_path, ref_path, wvocab, evocab, cvocab):
    data_set = []
    unks = []
    ct = 0
    lct = 0
    with tf.io.gfile.GFile(src_path, mode="r") as src_file:
        with tf.io.gfile.GFile(edge_path, mode="r") as edge_file:
            with tf.io.gfile.GFile(ref_path, mode="r") as ref_file:
                src, edges, ref = src_file.readline(), edge_file.readline(), ref_file.readline()

                while src and ref:
                    ct += 1

                    src_seq = src.lower().rstrip("\n").split(" ")
                    tgt = ref.lower().rstrip("\n").split(" ")
                    ref = ref.rstrip("\n")
                    graph = json.loads(edges.rstrip("\n"))

                    src_ids = []
                    tgt_ids = []
                    char_ids = []
                    unk = []
                    edges = []
                    reen = {}
                    ct_re = 0
                    i = 0
                    depth = []
                    for w in src_seq:
                        if w == " " or len(w) < 1:
                            continue
                        char_id = []
                        for cc in range(0, len(w)):
                            if w[cc] not in cvocab:
                                char_id.append(76)
                            else:
                                char_id.append(cvocab[w[cc]])
                        char_ids.append(char_id)
                        if w in wvocab:
                            src_ids.append(wvocab[w])
                            unk.append(w)
                        else:
                            src_ids.append(UNK_ID)
                            unk.append(w)
                        depth.append(0)

                        i += 1
                    depth[0] = 1

                    for w in tgt:
                        if w in wvocab:
                            tgt_ids.append(wvocab[w])
                        else:
                            tgt_ids.append(UNK_ID)

                    for l in graph:
                        id1 = int(l)
                        for pair in graph[l]:
                            edge, id2 = pair[0], pair[1]
                            if edge in evocab:
                                edge = evocab[edge]
                            else:
                                edge = UNK_ID

                            if depth[id1] == 0:
                                print("depth_id")
                            if depth[int(id2)] == 0:
                                depth[int(id2)] = depth[id1] + 1
                                if depth[int(id2)] > ct_re:
                                    ct_re = depth[int(id2)]
                            edges.append([edge, id1, int(id2)])
                    data_set.append([src_ids, tgt_ids, edges, char_ids, depth, ref.rstrip("\n")])
                    unks.append(unk)

                    if not(len(src_ids) < max_src_len - 1 and len(edges) < max_src_len - 1 and len(
                            tgt_ids) < max_tgt_len - 1):
                        lct += 1
                    src, edges, ref = src_file.readline(), edge_file.readline(), ref_file.readline()
    return tqdm(src,np.array(edges),ref)


from_vocab='data/vocab_15000'
to_vocab='data/vocab_15000'
wvocab, wvocab_rev = initialize_vocabulary(from_vocab)
evocab, evocab_rev = initialize_vocabulary("data/evocab")
cvocab, cvocab_rev = initialize_vocabulary("data/cvocab")

train_data = read_data_graph("data/train.src", "data/train.edge",
                                             "data/train.tgt",
                                             wvocab, evocab, cvocab)
valid_data= read_data_graph("data/valid.src", "data/valid.edge",
                                             "data/valid.tgt",
                                             wvocab, evocab, cvocab)
test_data = read_data_graph("data/test.src", "data/test.edge",
                                             "data/train.tgt",
                                             wvocab, evocab, cvocab)



print(train_data)

max_vocab = 100
max_len = 100

#embeddings = init_embedding(from_vocab)
# build vocabulary from training set
all_nodes = [s[0] for s in train_data]
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(all_nodes)


def prepare_single_batch(samples):
    sample_nodes = [s[0] for s in samples]
    print(sample_nodes)
    sample_nodes = tokenizer.texts_to_sequences(sample_nodes)
    
    sample_nodes = pad_sequences(sample_nodes, padding='post')
    max_nodes_len = np.shape(sample_nodes)[1]
    edges = [s[1]+i*max_nodes_len for i,s in enumerate(samples)]
    edges = [e for e in edges if len(e) > 0]
    node_to_graph = [[i]*max_nodes_len for i in range(len(samples))]
    all_nodes = np.reshape(sample_nodes, -1)
    all_edges = np.concatenate(edges)
    print(all_edges)
    node_to_graph = np.reshape(node_to_graph, -1)
    return {
        'data': all_nodes,
        'edges': all_edges,
        'node2grah': node_to_graph,
    }, np.array([s[2] for s in samples])



def gen_batch(dataset, batch_size=16, repeat=False, shuffle=True):
    while True:
        dataset = list(dataset)
        if shuffle:
            random.shuffle(dataset)
        l = len(dataset)
        for ndx in range(0, l, batch_size):
            batch_samples = dataset[ndx:min(ndx + batch_size, l)]
            yield prepare_single_batch(batch_samples)
        if not repeat:
            break


#hparams.add_hparam(name="embeddings", value=embeddings)
data = keras.Input(batch_shape=(None,))
edge = keras.Input(batch_shape=(None, 2), dtype=tf.int32)
node2graph = keras.Input(batch_shape=(None,), dtype=tf.int32)
embeded = Embedding(tokenizer.num_words, 20)(data)
num_graph = tf.reduce_max(node2graph)+1


# Build Graph Convolution 

gnn_input = GNNInput(
    node_features=embeded,
    adjacency_lists=(edge,),
    node_to_graph_map=node2graph, 
    num_graphs=num_graph,
)


"""
"message_calculation_class" configures the message passing style. This chooses the tf2_gnn.layers.message_passing.* layer used in each step.

RGCN: Relational Graph Convolutional Networks 
"""

params = GNN.get_default_hyperparameters()
#setting the message_calculationclass
params['message_calculation_class'] = 'rgcn'
params['use_inter_layer_layernorm']=True
params['hidden_dim'] = 12
params["num_aggr_MLP_hidden_layers"]=4
params["num_heads"]=4 
params["initial_node_representation_activation"]="relu"
params['dense_intermediate_layer_activation']='relu'
params['global_exchange_weighting_fun']='sigmoid'
#params['global_exchange_every_num_layers']=True
params['global_exchange_mode']='gru' # rnn layer 
gnn_layer = GNN(params)
gnn_out =gnn_layer(gnn_input)#[None,32]

avg = segment_mean(
    data=gnn_out,
    segment_ids=node2graph
)
print('mean:', avg)

pred = Dense(1, activation='sigmoid')(avg)
print('pred:', pred)

model = Model(
    inputs={
        'data': data,
        'edges': edge,
        'node2grah': node2graph,
    },
    outputs=pred
)
model.summary()

model.compile(
    optimizer=Adamax(),
    loss='sparse_categorical_crossentropy',
    metrics=['AUC']
)



batch_size = 16
num_batchs = math.ceil(len(train_data) / batch_size)
num_batchs_validation = math.ceil(len(valid_data) / batch_size)

history=model.fit(
    gen_batch(
        train_data, batch_size=batch_size, repeat=True
    ),
    steps_per_epoch=num_batchs,
    epochs=20,
    validation_data=gen_batch(
        valid_data, batch_size=16, repeat=True
    ),
    validation_steps=num_batchs_validation,
     callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, )
    ],
)

y_pred = model.predict(
    gen_batch(test_data, batch_size=16, shuffle=False)
)



"""
queries=[]    
keys=[]    
Q = tf.layers.dense(queries,64, activation=None, use_bias=False, name="q")  # (N, T_q, C)
K = tf.layers.dense(keys,64,activation=None, use_bias=False, name="k")  # (N, T_k, C)
V = tf.layers.dense(keys,64, activation=None, use_bias=False, name="v")  # (N, T_k, C)


# Split and concat
Q_ = tf.concat(tf.split(Q, 2, axis=2), axis=0)  # (h*N, T_q, C/h)
K_ = tf.concat(tf.split(K, 2, axis=2), axis=0)  # (h*N, T_k, C/h)
"""
