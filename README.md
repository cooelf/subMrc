Original implementation of the paper **Subword-augmented Embedding for Cloze Reading Comprehension**

## Environment

MRC Model:

- Python 2.7
- Theano 0.9.0dev
- Lasagne 0.2.dev1

BPE Segmentation:

- Python 2.7 and python 3.6 would be fine :)

### Instruction 
BPE segmentation can be referred to https://github.com/rsennrich/subword-nmt
An segmentation sample for a general use is shown in subword folder.
Firstly, learn a bpe segmentation and then use it!
build_dataset_bpe.py -> learn_bpe.py -> apply_bpe.py / word2sub.py

There are two ways of using subwords which could be adopted to various NLP tasks:
1) using the vocab->subwords mapping (as vocab_bpe_1k.txt) or
2) segment the words one by one in data processing (word2sub.py)

This implementation employs the first way. After obtaining the subword mapping file, we need to specify "subdic" directory in config.py.

If you use this sourse please cite our paper:

```
@inproceedings{zhang2018mrc,
    title = {Subword-augmented Embedding for Cloze Reading Comprehension},
    author = {Zhang, Zhuosheng and Huang, Yafang and Zhao, Hai},
    booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)},
    year = {2018}
}

```
