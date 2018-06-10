import train
import test
import argparse
import os
import numpy as np
import random

from config import get_params

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', dest='mode', type=int, default=2,
        help='run mode - (0-train+test, 1-train only, 2-test only, 3-val only)')
parser.add_argument('--nlayers', dest='nlayers', type=int, default=3,
        help='Number of reader layers')
parser.add_argument('--dataset', dest='dataset', type=str, default='cmrc',
        help='Dataset - cmrc')
parser.add_argument('--seed', dest='seed', type=int, default=36,
        help='Seed for different experiments with same settings')
parser.add_argument('--gating_fn', dest='gating_fn', type=str, default='T.mul',
        help='Gating function (T.mul || Tsum || Tconcat)')
args = parser.parse_args()
cmd = vars(args)
params = get_params(cmd['dataset'])
params.update(cmd)

np.random.seed(params['seed'])
random.seed(params['seed'])

# save directory
w2v_filename = params['word2vec'].split('/')[-1].split('.')[0] if params['word2vec'] else 'None'
save_path = ('experiments/mul_bpe_1k cmrc_nhid128_nlayers3_dropout0.5_wiki_chardim100_train1_seed36_use-feat1_gfT.mul')
if not os.path.exists(save_path): os.makedirs(save_path)

# train
if params['mode']<2:
    train.main(save_path, params)

# test
if params['mode']==0 or params['mode']==2:
    test.main(save_path, params)
elif params['mode']==3:
    test.main(save_path, params, mode='validation')
