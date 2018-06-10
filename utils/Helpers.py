# -*- coding: utf-8 -*-
import numpy as np
from config import *
import gensim
EMBED_DIM = 128
def show_predicted_vs_ground_truth(probs, a, cand, inv_dict):
    i = 0
    id = list(np.argmax(probs, axis=1))
    pre = cand[i,id[0]]
    predicted_ans = map(lambda i:inv_dict[i], list(pre))
    true_ans = map(lambda i:inv_dict[i], cand[i,list(a)[0]])
    return predicted_ans, true_ans

def count_candidates(probs, c, m_c):
    hits = 0
    predicted_ans = list(np.argmax(probs, axis=1))
    for i, x in enumerate(predicted_ans):
        for j, y in enumerate(c[i,:]):
            if x == y and m_c[i,j] > 0:
                hits += 1
                break
    return hits

def show_question(d, q, a, m_d, m_q, c, m_c, inv_dict):
    i = 0
    inv_vocab = lambda x:inv_dict[x]
    pa =  map(inv_vocab, list(d[i,m_d[i]>0,0]))
    pb =  map(inv_vocab, list(q[i,m_q[i]>0,0]))
    cc = list(c[i, m_c[i] > 0])
    cc2= list(c[i, m_c[i] > 0,0])
    # pc =  map(inv_vocab, list(c[i,m_c[i]>0]))
    pd =  inv_vocab(a[i])
    return pa,pb,pd

def load_word2vec_embeddings(dictionary, vocab_embed_file):
    if vocab_embed_file is None: return None, EMBED_DIM

    fp = open(vocab_embed_file)

    info = fp.readline().split()
    embed_dim = int(info[1])

    vocab_embed = {}
    for line in fp:
        line = line.split()
        vocab_embed[line[0]] = np.array(map(float, line[1:]), dtype='float32')
    fp.close()

    vocab_size = len(dictionary)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for w, i in dictionary.iteritems():
        if w in vocab_embed:
            W[i,:] = vocab_embed[w]
            n += 1
    print "%d/%d vocabs are initialized with word2vec embeddings." % (n, vocab_size)
    return W, embed_dim

def load_sub_embeddings(dictionary, vocab_embed_file):
    if vocab_embed_file is None or vocab_embed_file =="": return None, 100

    # fp = open(vocab_embed_file)
    #
    # info = fp.readline().split()
    # embed_dim = int(info[1])
    #
    # vocab_embed = {}
    # for line in fp:
    #     line = line.split()
    #     vocab_embed[line[0]] = np.array(map(float, line[1:]), dtype='float32')
    # fp.close()
    #
    # vocab_size = len(dictionary)
    # W = np.random.randn(vocab_size, embed_dim).astype('float32')
    # n = 0
    # for w, i in dictionary.iteritems():
    #     if w in vocab_embed:
    #         W[i,:] = vocab_embed[w]
    #         n += 1
    # print "%d/%d vocabs are initialized with sub embeddings." % (n, vocab_size)
    # return W, embed_dim

