import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
from configs import *

def main(file_hash):
    # 载入模型参数
    targrt_model = keras.models.load_model(MODEL_SAVE_PATH)

    # 读入测试图片
    input_low_image = cv2.imread(INPUT_IMAGE_PATH+file_hash)
    # 转为浮点数
    input_low_image = input_low_image / 255
    

    input_image_rows = input_low_image.shape[0]
    input_image_cols = input_low_image.shape[1]
    input_image_chns = input_low_image.shape[2]

    format_image_rows = math.ceil(input_image_rows / INPUT_IMAGE_SIZE) * INPUT_IMAGE_SIZE
    format_image_cols = math.ceil(input_image_cols / INPUT_IMAGE_SIZE) * INPUT_IMAGE_SIZE
    output_image_rows = format_image_rows * MODEL_SCALE
    output_image_cols = format_image_cols * MODEL_SCALE

    format_low_image = np.zeros(((format_image_rows, format_image_cols, input_image_chns)))
    output_high_image = np.zeros(((output_image_rows, output_image_cols, input_image_chns)))

    format_low_image[0: input_image_rows, 0: input_image_cols, ...] = input_low_image
    test_low_image = np.zeros((1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, input_image_chns))
    
    # 切分图像依次放大
    for y in range(0, input_image_rows, INPUT_IMAGE_SIZE):
        for x in range(0, input_image_cols, INPUT_IMAGE_SIZE):
            test_low_image[0, ...] = format_low_image[y : y + INPUT_IMAGE_SIZE, x : x + INPUT_IMAGE_SIZE, ...]
            test_high_image = targrt_model.predict(test_low_image)

            output_high_image[2*y : 2*y + OUTPUT_IMAGE_SIZE, 2*x : 2*x + OUTPUT_IMAGE_SIZE, ...] = test_high_image[0, ...]

    cv2.imwrite(OUTPUT_IMAGE_PATH+file_hash+".png",output_high_image*255)

    return 1

    