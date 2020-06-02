import os

# data path and log path
if(os.name=='poxis'):
    ORIGINAL_IMAGES_PATH = '../data/original'
    TRAINING_IMAGES_PATH = '../data/train'
    VALIDATION_IMAGES_PATH = '../data/valid'
    CHECKPOINTS_PATH = '../checkpoints/cp.ckpt'
    SERIALIZE_PATH = './tmp.data'
elif(os.name=='nt'):
    ORIGINAL_IMAGES_PATH = '..\\data\\original'
    TRAINING_IMAGES_PATH = '..\\data\\train'
    VALIDATION_IMAGES_PATH = '..\\data\\valid'
    CHECKPOINTS_PATH = '..\\checkpoints\\cp.ckpt'
    SERIALIZE_PATH = '.\\tmp.data'
else:
    ORIGINAL_IMAGES_PATH = './data/original'
    TRAINING_IMAGES_PATH = './data/train'
    VALIDATION_IMAGES_PATH = './data/valid'
    CHECKPOINTS_PATH = './checkpoints/cp.ckpt'
    SERIALIZE_PATH = './tmp.data'
    
DEFAULT_MODEL = 'vgg_7'
BATCH_SIZE = 8
STEPS_PER_EPOCH = 5000





