# -*- coding: utf-8 -*-

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


glove_file = datapath('C:/Users/Ahmed Harby/Documents/GitHub/GraphTransformer/data/amr_vector.txt')
tmp_file = get_tmpfile("C:/Users/Ahmed Harby/Documents/GitHub/GraphTransformer/data/amr.txt")

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)
model.save("amr.model")