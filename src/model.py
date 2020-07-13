import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers

def choos_model(name):
    if(name=='vgg_16'):
        return vgg_16()
    elif(name=='vgg_7'):
        return vgg_7()
    elif(name=='vgg_11'):
        return vgg_11()



def vgg_16():
    model = models.Sequential()
    # 对输入进行卷积
    model.add(layers.Conv2D(16,(3,3),input_shape=(32,32,3),padding='SAME'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu',padding='SAME'))

    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME')) 
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME')) 

    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))

    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))

    model.add(layers.Conv2D(256, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu',padding='SAME'))

    model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='SAME'))
    # 最后进行反卷积实现放大效果
    model.add(layers.Conv2DTranspose(3, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))

    print(model.summary())
    return model

def vgg_7():
    model = models.Sequential()
    # 对输入进行卷积
    model.add(layers.Conv2D(16,(3,3),input_shape=(32,32,3),padding='SAME'))

    model.add(layers.Conv2D(16, (3, 3), activation='relu',padding='SAME'))

    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME')) 

    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))
    # 最后进行反卷积实现放大效果
    model.add(layers.Conv2DTranspose(3, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))

    print(model.summary())
    return model


def vgg_11():
    model = models.Sequential()
    model.add(layers.Conv2D(16,(3,3),input_shape=(32,32,3),padding='SAME'))

    model.add(layers.Conv2DTranspose(16, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))
    # 根据测试的结果，似乎先放大图像效果好些
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME')) 


    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))


    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))

    
    model.add(layers.Conv2D(256, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu',padding='SAME'))
    
    model.add(layers.Conv2D(3, (3, 3), activation='relu',padding='SAME'))
    print(model.summary())
    return model

# vgg_16()
# vgg_7()
# vgg_11()