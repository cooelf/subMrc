# -*- coding: utf-8 -*-
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import numpy as np
import glob
import os
import sys
import re
import pickle
from config import MAX_WORD_LEN
from datetime import datetime
from collections import Counter
SYMB_BEGIN = "@begin"
SYMB_END = "@end"
UNK = "@unk"

class Data:

    def __init__(self, dictionary, num_entities, training, validation, test):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1])
        self.num_entities = num_entities
        self.inv_dictionary = {v:k for k,v in dictionary[0].items()}

class DataPreprocessor:

    def preprocess(self, question_dir, no_training_set=False, use_subs=False, proportion = 0.9,subdict='data/cmrc/vocab_bpe_1k.txt'):
        """
                preprocess all data into a standalone Data object.
                the training set will be left out (to save debugging time) when no_training_set is True.
                """
        data_f = os.path.join(question_dir, "data.pkl")

        if os.path.exists(data_f):
            print "data exists, loading",datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data_file = file(data_f, 'rb')
            data = pickle.load(data_file)
            print "training data = %d" % len(data.training)
            data_file.close()
        else:
            print "data not exists, building",datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            vocab_f = os.path.join(question_dir,"vocab.txt")
            word_dictionary, char_dictionary,sub_dic, num_entities = \
                    self.make_dictionary(question_dir, vocab_file=vocab_f ,proportion = proportion,subdict=subdict)
            dictionary = (word_dictionary, char_dictionary,sub_dic)
            if no_training_set:
                training = None
            else:
                print "preparing training data ...",datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                training = self.parse_all_files(question_dir + "/CMRC2017_train", dictionary, use_subs)
                print "training data = %d" % len(training)
            print "preparing validation data ...",datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            validation = self.parse_all_files(question_dir + "/CMRC2017_cloze_valid_v2", dictionary, use_subs)
            print "validation data = %d" % len(validation)
            print "preparing test data ...",datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            test = self.parse_all_files(question_dir + "/CMRC2017_test", dictionary, use_subs)
            print "test data = %d" % len(test)
            data = Data(dictionary, num_entities, training, validation, test)
            # data_file = file(data_f, 'wb')
            # pickle.dump(data, data_file)
            # f1 = file('training.pkl', 'wb')
            # f2 = file('validation.pkl', 'wb')
            # f3 = file('test.pkl', 'wb')
            # pickle.dump(training, f1)
            # pickle.dump(validation, f2)
            # pickle.dump(test, f3)
            # data_file.close()
            # f1.close()
            # f2.close()
            # f3.close()
        return data

    def make_dictionary(self, question_dir, vocab_file, proportion,subdict):
        if os.path.exists(vocab_file):
            print "loading vocabularies from " + vocab_file + " ..."
            word = map(lambda x: x.strip(), open(vocab_file).readlines())
            print vocab_file
        else:
            print "no " + vocab_file + " found, constructing the vocabulary list ...",datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            fnames = []
            fnames += glob.glob(question_dir + "/CMRC2017_train/*.doc_query")
            fnames += glob.glob(question_dir + "/CMRC2017_train/*.answer")
            fnames += glob.glob(question_dir + "/CMRC2017_cloze_valid_v2/*.doc_query")
            fnames += glob.glob(question_dir + "/CMRC2017_cloze_valid_v2/*.answer")
            counter = Counter()
            n = 0.
            for fname in fnames:
                fp = open(fname)
                for line in fp:
                    line = line.strip().split(' ||| ')
                    text = line[1].split()
                    for token in text:
                        counter[token] += 1

                fp.close()

            word, _ = zip(*counter.most_common())
            print "writing vocabularies to " + vocab_file + " ...",datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            vocab_fp = open("vocab.txt", "w")
            vocab_fp.write('\n'.join(word))
            vocab_fp.close()


        word = (list)(word)
        known_word_count = int(proportion * len(word))
        word_split = word[:known_word_count]
        word_split.append(UNK)
        vocab_size = len(word_split)
        word_dictionary = {token: i for i, token in enumerate(word_split)}

        sub_dic = {}
        char_list = []
        for line in open(subdict):
            line_split = line.split("\t")
            # sub_dic[line_split[0].decode("utf-8")] = line_split[1].decode("utf-8").replace("\n","")
            sub_dic[line_split[0]] = line_split[1].replace("\n", "")
            sub_words = line_split[1].split()
            for sub in sub_words:
                char_list.append(sub)
        message = "load %d subwords" % len(sub_dic)
        print(message)
        # for line in open("data/cmrc/cmrc_av_vitebi.txt"):
        #     line_split = line.strip()          # sub_dic[line_split[0].decode("utf-8")] = line_split[1].decode("utf-8").replace("\n","")
        #     sub_dic[line_split.replace(" ", "")] = line_split
        #     sub_words = line_split.split()
        #     for sub in sub_words:
        #         char_list.append(sub)
        # message = "load %d subwords" % len(sub_dic)
        # print(message)

        char_set = set(char_list)
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in word_split if v.startswith('@entity')])
        print "vocab_size = %d" % vocab_size
        print "num characters = %d" % len(char_set)
        print "%d anonymoused entities" % num_entities
        print "%d other tokens (including @placeholder, %s and %s)" % (
                vocab_size-num_entities, SYMB_BEGIN, SYMB_END)

        return word_dictionary, char_dictionary,sub_dic, num_entities


    def parse_one_file(self, fname, answer_file, dictionary, use_subs):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """

        w_dict, s_dict,sub_dic = dictionary[0], dictionary[1],dictionary[2]
        UNK_ID = w_dict["@unk"]
        answer_doc = self.read_answer(answer_file)
        data = []
        doc = ""
        # num_lines = 0
        # qid_list = np.load("F:\cmrc\mut_split_7795\utils\qid.npy").tolist()
        f = open(fname, "r")
        for line in f:
                line = line.strip().split(' ||| ')
                if re.search('^<qid', line[0]):
                    qid = line[0]
                    # if qid in unk_qid:
                    if True:
                        query = line[1]
                        answer = answer_doc[qid]
                        doc_raw = doc.split()  # document
                        qry_raw = query.split()  # query
                        ans_raw =answer.strip()  # answer
                        candidates = list(set(doc_raw))
                        cand_raw = map(lambda x: x.strip().split(':')[0].split(), candidates)  # candidate answers

                        # wrap the query with special symbols
                        qry_raw.insert(0, SYMB_BEGIN)
                        qry_raw.append(SYMB_END)
                        try:
                            cloze = qry_raw.index('XXXXX')
                        except ValueError:
                            print 'XXXXX not found in qid', qid
                            # at = qry_raw.index('XXXXX')
                            # qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
                            # cloze = qry_raw.index('XXXXX')
                        # tokens/entities --> indexes
                        doc_words = map(lambda w: w_dict.get(w, UNK_ID), doc_raw)
                        qry_words = map(lambda w: w_dict.get(w, UNK_ID), qry_raw)
                        if use_subs:
                            doc_subs = map(lambda w: map(lambda s: s_dict.get(s, s_dict[' ']), sub_dic[w].split()[:MAX_WORD_LEN] if w in sub_dic else [sub for sub in w][:MAX_WORD_LEN]), doc_raw)
                            qry_subs = map(lambda w: map(lambda s: s_dict.get(s, s_dict[' ']), sub_dic[w].split()[:MAX_WORD_LEN] if w in sub_dic else [sub for sub in w][:MAX_WORD_LEN]), qry_raw)
                            # for word in doc_raw:
                            #     if word in sub_dic:
                            #         chars = sub_dic[word]
                            #         for char in chars.split():
                            #             c = c_dict[char]

                            # doc_chars = map(lambda w: map(lambda c: c_dict.get(c, c_dict[' ']),
                            #                               sub_dic[w].split()[:MAX_WORD_LEN] if w in sub_dic else [char for
                            #                                                                                       char in
                            #                                                                                       w][
                            #                                                                                      :MAX_WORD_LEN]),
                            #                 doc_raw)
                            # qry_chars = map(lambda w: map(lambda c: c_dict.get(c, c_dict[' ']),
                            #                               sub_dic[w].split()[:MAX_WORD_LEN] if w in sub_dic else [char for
                            #                                                                                       char in
                            #                                                                                       w][
                            #                                                                                      :MAX_WORD_LEN]),
                            #                 qry_raw)

                        else:
                            doc_subs, qry_subs = [], []
                        ans = map(lambda w: w_dict.get(w, UNK_ID), ans_raw.split())
                        cand = [map(lambda w: w_dict.get(w, UNK_ID), c) for c in cand_raw]
                        if len(cand) <1 or len(doc_words)<1 or len(qry_words) <1 or len(ans) <1:
                            print "item length < 1"
                        dataset = doc_words, qry_words, ans, cand, doc_subs, qry_subs, cloze, qid
                        data.append(dataset)
                    doc = ""
                else:
                    sent = line[1]
                    doc += sent + " "
        return data

    def parse_all_files(self, directory, dictionary, use_subs):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """
        query_file = glob.glob(directory + '/*.doc_query')[0]
        answer_file = glob.glob(directory + '/*.answer')[0]
        questions = self.parse_one_file(query_file, answer_file, dictionary, use_subs)
        return questions

    def read_answer(self, file):
        f = open(file, "r")
        answer_dic ={}
        for line in f:
            line = line.strip().split(' ||| ')
            qid = line[0]
            answer = line[1]
            answer_dic[qid] = answer
        return answer_dic

