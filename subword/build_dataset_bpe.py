# -*- coding: utf-8 -*-
import codecs
from datetime import datetime
import sys
import re
reload(sys)
sys.setdefaultencoding('utf-8')
from sys import argv
import string
raw,ans_file,output = "sample/cloze.valid.doc_query","sample/cloze.valid.answer","output/bpe_corpus.txt"

def read_answer(file):
    f = codecs.open(file, "r")
    answer_dic = {}
    for line in f:
        line = line.strip().split(' ||| ')
        qid = line[0]
        answer = line[1]
        answer_dic[qid] = answer
    return answer_dic

answer_doc = read_answer(ans_file)
text = []
doc = []
f = open(raw, "r")
vocab_file = output
vocab_fp = open(vocab_file, "w")
for line in f:
    line = line.strip().split(' ||| ')
    if re.search('^<qid', line[0]):
        qid = line[0]
        answer = answer_doc[qid]
        for sent in doc:
            if "XXXXX" in sent:
                sent = sent.replace("XXXXX", answer)
            if len(sent) > 1:
                vocab_fp.write(sent+"\n")
        doc = []
    else:
        sent = line[1]
        doc.append(sent)
# @placehoder, @begin and @end are included in the vocabulary list
f.close()
vocab_fp.close()

