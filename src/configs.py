import os

# data path and log path
ORIGINAL_IMAGES_PATH = '../data/original'
TRAINING_IMAGES_PATH = '../data/train'
VALIDATION_IMAGES_PATH = '../data/valid'
CHECKPOINTS_PATH = '../checkpoints/cp.ckpt'
SERIALIZE_PATH = './tmp.data'
INPUT_IMAGE_PATH = './test_image/original.png'
OUTPUT_IMAGE_PATH = './test_image/2x.png'
MODEL_PATH = '../model/model.h5'


    
# train
DEFAULT_MODEL = 'vgg_11'     # 模型名
BATCH_SIZE = 8              # batch大小
DEFAULT_EPOCH = 1500        # 循环次数
STEPS_PER_EPOCH = 5000

# start
OUTPUT_IMAGE_SIZE = 64      # 输出图片大小
INPUT_IMAGE_SIZE = 32       # 输入图片大小
MODEL_SCALE = 2             # 模型放大倍率           


# generate
SCALE = 1                   # 输出图像的block倍率
BLOCK_SIZE = 64             # 输出图像大小





