import os

# data path and log path
if(os.name=='poxis'):
    ORIGINAL_IMAGES_PATH = '../data/original'
    TRAINING_IMAGES_PATH = '../data/train'
    VALIDATION_IMAGES_PATH = '../data/valid'
    CHECKPOINTS_PATH = '../checkpoints/cp.ckpt'
    SERIALIZE_PATH = './tmp.data'
    INPUT_IMAGE_PATH = './test_image/original.png'
    OUTPUT_IMAGE_PATH = './test_image/2x.png'
elif(os.name=='nt'):
    ORIGINAL_IMAGES_PATH = '..\\data\\original'
    TRAINING_IMAGES_PATH = '..\\data\\train'
    VALIDATION_IMAGES_PATH = '..\\data\\valid'
    CHECKPOINTS_PATH = '..\\checkpoints\\cp.ckpt'
    SERIALIZE_PATH = '.\\tmp.data'
    INPUT_IMAGE_PATH = '.\\test_image\\original.png'
    OUTPUT_IMAGE_PATH = '.\\test_image\\2x.png'
else:
    ORIGINAL_IMAGES_PATH = './data/original'
    TRAINING_IMAGES_PATH = './data/train'
    VALIDATION_IMAGES_PATH = './data/valid'
    CHECKPOINTS_PATH = './checkpoints/cp.ckpt'
    SERIALIZE_PATH = './tmp.data'
    INPUT_IMAGE_PATH = './test_image/original.png'
    OUTPUT_IMAGE_PATH = './test_image/2x.png'
    
DEFAULT_MODEL = 'vgg_7'
BATCH_SIZE = 8
STEPS_PER_EPOCH = 5000

OUTPUT_IMAGE_SIZE = 64
INPUT_IMAGE_SIZE = 32
SCALE = 2





