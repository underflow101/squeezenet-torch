#####################################################################
# hyper.py
#
# Dev. Dongwon Paek
# Description: PyTorch model file of SqueezeNet
#####################################################################

CLASSES = ('others', 'phoneWithHand', 'writing', 'sleep')

EPOCHS = 1
BATCH_SIZE = 32
NUN_CLASSES = 4
IMG_SIZE = 224

LEARNING_RATE = 1e-3
DECAY = 5e-4
MOMENTUM = 0.9

NUM_WORKERS = 12

class CONFIG():
    # dataset info
    input_channels = 3
    class_count = 256

    # training settings
    lr = 4e-2
    momentum = 0.9
    weight_decay = 2e-4
    num_epochs = 1
    batch_size = 32
    pretrained_model = None
    num_classes = 4

    # misc
    mode = 'train'
    use_gpu = True
    use_tensorboard = False

    # dataset
    data_path = '/home/bearpaek/data/datasets/lplSmall/'
    #train_data_path = '/home/bearpaek/data/datasets/lplSmall/train/'
    train_data_path = 'train/train_data.hdf5'
    #test_data_path = '/home/bearpaek/data/datasets/lplSmall/validation/'
    test_data_path = 'validation/'
    train_x_key = 'train'
    train_y_key = 'train_y'
    test_x_key = 'test_x'
    test_y_key = 'test_y'

    # path
    log_path = './logs'
    model_save_path = './models'

    # epoch step size
    loss_log_step = 1
    model_save_step = 1
    train_eval_step = 1
    def __init__(self):
        super().__init__()
    '''
    def input_channels(self):
        return 3
    def class_count(self):
        return 256
    def lr(self):
        return 4e-2
    def momentum(self):
        return 0.9
    def weight_decay(self):
        return 2e-4
    def num_epochs(self):
        return 1
    def batch_size(self):
        return 32
    def pretrained_model(self):
        return None
    def mode(self):
        return 'train'
    def use_gpu(self):
        return True
    def use_tensorboard(self):
        return False
    def data_path(self):
        return '/home/bearpaek/data/datasets/lplSmall/'
    def train_data_path(self):
        return '/home/bearpaek/data/datasets/lplSmall/train/'
    def test_data_path(self):
        return '/home/bearpaek/data/datasets/lplSmall/validation/'
    def train_x_key(self):
        return 'train_x'
    def train_y_key(self):
        return 'train_y'
    def test_x_key(self):
        return 'test_x'
    def test_y_key(self):
        return 'test_y'
    def log_path(self):
        return './logs'
    def model_save_path(self):
        return './models'
    def loss_log_step(self):
        return 1
    def model_save_step(self):
        return 1
    def train_eval_step(self):
        return 1
    '''
    
# dataset info
input_channels = 3
class_count = 256

# training settings
lr = 4e-2
momentum = 0.9
weight_decay = 2e-4
num_epochs = 1
batch_size = 32
pretrained_model = None

# misc
mode = 'train'
use_gpu = True
use_tensorboard = False

# dataset
data_path = '/home/bearpaek/data/datasets/lplSmall/'
train_data_path = '/home/bearpaek/data/datasets/lplSmall/train/'
test_data_path = '/home/bearpaek/data/datasets/lplSmall/validation/'
train_x_key = 'train_x'
train_y_key = 'train_y'
test_x_key = 'test_x'
test_y_key = 'test_y'

# path
log_path = './logs'
model_save_path = './models'

# epoch step size
loss_log_step = 1
model_save_step = 1
train_eval_step = 1