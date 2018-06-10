# -*- coding: utf-8 -*-
import codecs
from BPE import *

codes = codecs.open("output/codes_bpe_1k.txt", encoding='utf-8')
bpe = BPE(codes, -1, ' ', None, None)

word = u"大花猫"
print (bpe.word2sub(word))

