import os
import cv2
import numpy as np
import shutil
from configs import *

def make_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)


def main():
    
    make_dir(TRAINING_IMAGES_PATH)
    make_dir(VALIDATION_IMAGES_PATH)

    original_images_names = os.listdir(ORIGINAL_IMAGES_PATH)
    original_images_path = []
    for original_image_name in original_images_names:
        original_images_path.append(os.path.join(ORIGINAL_IMAGES_PATH, original_image_name))

    image_num = 0
    for image_path in original_images_path:
        print(image_path)
        image = cv2.imread(image_path)

        image_rows = image.shape[0]
        image_cols = image.shape[1]

        max_block = (image_cols // BLOCK_SIZE)*(image_rows // BLOCK_SIZE)
        loop_num = max_block * SCALE
        block_num = 0
        for i in range(0, loop_num):
            block_x = np.random.randint(0, image_cols - BLOCK_SIZE)
            block_y = np.random.randint(0, image_rows - BLOCK_SIZE)

          
            block_image = image[block_y : block_y + BLOCK_SIZE, block_x : block_x + BLOCK_SIZE, ...]

            output_name = str(image_num)+'_'+str(block_num)+'.png'

            if i % 100 == 0:
                cv2.imwrite(os.path.join(VALIDATION_IMAGES_PATH, output_name), block_image)
            else:
                cv2.imwrite(os.path.join(TRAINING_IMAGES_PATH, output_name), block_image)
            block_num += 1
        image_num += 1
        
    

if __name__ == '__main__':
    main()