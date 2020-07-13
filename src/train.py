import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import model
import cv2
import matplotlib.image as mpimg
from tensorflow import keras
import os
from configs import *
import pickle


def main():
  
    # 导入训练集
    train_high_images = high_images(TRAINING_IMAGES_PATH)
    train_high_images = train_high_images/255
    # 对训练集进行图像增强
    # train_high_images = image_preprocess(train_high_images)
    train_low_images = tf.image.resize(train_high_images,[32,32], method='bilinear')

    # 导入验证集
    valid_high_images = high_images(VALIDATION_IMAGES_PATH)
    valid_low_images = tf.image.resize(valid_high_images,[32,32], method='bilinear')
   
    # 选择模型
    target_model = model.choos_model(DEFAULT_MODEL)
    # 接受训练中的数据,batch_size=8
    history = compile_and_fit(target_model, train_low_images, train_high_images,valid_low_images,valid_high_images,epochs=DEFAULT_EPOCH,batch=BATCH_SIZE)
    history_dict = history.history

    # 存储模型
    target_model.save(MODEL_PATH)

    # 存储history信息
    serialize(history_dict)
    
    # 绘制信息图标
    draw_result(history_dict)
    

def high_images(image_path):
    '''
    返回训练集图像
    '''
    all_images_names = os.listdir(image_path)
    num = len(all_images_names)
    images = np.zeros((num, 64, 64, 3))
    i = 0
    for image_name in all_images_names:
            image = os.path.join(image_path, image_name)
            image = np.array(cv2.imread(image))
            images[i] = image
            i += 1
    return images

def image_preprocess(images):
    '''
    对图像做增强处理
    images=4-D tensor
    '''
    images = tf.image.random_brightness(images, max_delta=0.3)
    # images = tf.image.random_contrast(images, 0.8, 1.2)
    return images

def show_image(image):
    '''
    显示图片
    image= 3-D tensor
    '''
    plt.figure("Image") # 图像窗口名称
    plt.imshow(image)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    plt.show()   

def get_optimizer():
    '''
    获得优化的学习率曲线
    '''
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001,decay_steps=STEPS_PER_EPOCH*50,decay_rate=0.5,staircase=False)
    return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks():
    return [
    tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_PATH,save_weights_only=True,verbose=1,period=5)
    ]


def compile_and_fit(input_model,data_train,data_label, vaild_data,vaild_label, optimizer=None, epochs=500,batch=8):
    '''
    训练模型
    '''
    if optimizer is None:
        optimizer = get_optimizer()
    input_model.compile(optimizer=optimizer,
                loss='mse',
                metrics=[
                    'accuracy'])

    input_model.summary()

    history = input_model.fit(
    data_train,data_label,
    batch_size=batch,
    epochs=epochs,
    validation_data=(vaild_data,vaild_label),
    callbacks=get_callbacks(),
    verbose=1, 
    shuffle=True)
    return history


def serialize(data):
    '''
    序列化数据
    '''
    tmp = pickle.dumps(data)
    with open(SERIALIZE_PATH,'wb') as f:
        f.write(tmp)

def unserialize():
    '''
    反序列化数据
    '''
    with open(SERIALIZE_PATH,'rb') as f:
        tmp = f.read()
        data = pickle.loads(tmp)
        return data


def draw_result(history_dict):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()   

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()


