# Minibatch Size
BATCH_SIZE = 64
# Gradient clip threshold
GRAD_CLIP = 10
# Learning rate
LEARNING_RATE = 0.001
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Number of epochs for training
NUM_EPOCHS = 10
# do validation every VALIDATION_FREQ iterations
VALIDATION_FREQ = 300
# maximum word length for character model
MAX_WORD_LEN = 5

# dataset params
def get_params(dataset):
    if dataset == 'cmrc':
        return cmrc_params
    else:
        raise ValueError("Dataset %s not found"%dataset)
cmrc_params={
        'nhidden'   :   128,
        'sub_dim'  :   100,
        'dropout'   :   0.5,
        'word2vec'  :   'data/wiki.txt',
        'sub2vec':     '',
        'subdic':     'data/cmrc/vocab_1k.txt',
        'train_emb' :   1,
        'use_feat'  :   1,
        'data'      :    "data/cmrc"
}
