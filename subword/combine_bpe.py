import sys
import re
reload(sys)
sys.setdefaultencoding('utf-8')
from sys import argv
origin_f = "sample/vocab.txt"
seg_f = "output/vocab_bpe_1k_raw.txt"
new_f = "output/vocab_bpe_1k.txt"
new = open(new_f,"w")
with open(origin_f) as origin, open(seg_f) as seg:
    origin_l = [word.replace("\n","") for word in origin]
    seg_l = [word.replace("\n","") for word in seg]
    for word,value in zip(origin_l,seg_l):
        new.write(word + "\t" + value.strip().replace("@@","") + '\n')
new.close()

