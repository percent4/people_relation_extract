import os
from enum import Enum


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


file_path = os.path.dirname(os.path.dirname(__file__))

model_dir = os.path.join(file_path, 'chinese_L-12_H-768_A-12')
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
vocab_file = os.path.join(model_dir, 'vocab.txt')

# the maximum length of a sequence,Sequences larger than max_seq_len will be truncated on the left side. Thus, if you
# want to send long sequences to the model, please make sure the program can handle them correctly.
#max_seq_len = 5
xla = True
# list of int. this model has 12 layers, By default this program works on the second last layer. The last layer is too
# closed to the target functions,If you question about this argument and want to use the last hidden layer anyway, please
# feel free to set layer_indexes=[-1], so we use the second last layer
layer_indexes = [-2]
#pooling_strategy = PoolingStrategy.NONE
