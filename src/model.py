import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers

def choos_model(name):
    if(name=='vgg_16'):
        return vgg_16()
    elif(name=='vgg_7'):
        return vgg_7()
    elif(name=='vgg_max'):
        return vgg_max()



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


def vgg_max():
    model = models.Sequential()
    model.add(layers.Conv2D(16,(3,3),input_shape=(32,32,3),padding='SAME'))
    # 卷积输入层，指定了输入图像的大小

    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME'))
    # 64个3x3的卷积核，生成64*128*128的图像，激活函数为relu

    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME')) 
    # 再来一次卷积 生成64*128*128
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    # model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64,(3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64,(3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
   
   
  
    model.add(layers.Conv2DTranspose(3, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))
    # 返回3层，两倍的模型
    print(model.summary())
    return model

# vgg_16()
# vgg_7()
# vgg_max()