import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers

def choos_model(name):
    if(name=='vgg_16'):
        return vgg_16()
    elif(name=='vgg_7_dec'):
        return vgg_7_dec()
    elif(name=='vgg_7'):
        return vgg_7()
    elif(name=='vgg_max'):
        return vgg_max()




def srcnn_935():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (9, 9), activation='relu', input_shape=(32, 32, 3),padding='SAME'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(3, (5, 5), activation='relu',padding='SAME'))
    # model.summary()
    return model

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
    model.add(layers.Conv2D(16,(3,3),input_shape=(32,32,3),padding='SAME'))
    model.add(layers.Conv2DTranspose(16, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))
    # 卷积输入层，指定了输入图像的大小
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME'))
    # 64个3x3的卷积核，生成64*128*128的图像，激活函数为relu
    

    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME')) 
    # 再来一次卷积 生成64*128*128


    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))


    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))

    
    model.add(layers.Conv2D(256, (3, 3), activation='relu',padding='SAME'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu',padding='SAME'))
    
    model.add(layers.Conv2D(3, (3, 3), activation='relu',padding='SAME'))
    

    batch_size = 8
    rows = 16
    cols = 16
    channels = 3
    # model.add(layers.Conv2DTranspose(256, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))
    # model.add(layers.Conv2DTranspose(3, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))
    
    print(model.summary())
    # 返回3层，两倍的模型
    return model

def vgg_7_dec():
    model = models.Sequential()
    # 对输入进行卷积
    model.add(layers.Conv2D(16,(3,3),input_shape=(32,32,3),padding='SAME'))

    model.add(layers.Conv2D(16, (3, 3), activation='relu',padding='SAME'))

    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME')) 

    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))
    # 最后进行反卷积实现放大效果
    model.add(layers.Conv2DTranspose(3, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))

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
   
   
    

    
   
    
    # model.add(layers.ZeroPadding2D((1,1)))
    batch_size = 8
    rows = 16
    cols = 16
    channels = 3
    # model.add(layers.Conv2DTranspose(3, (3, 3),strides=(1, 1), padding='SAME', activation='relu'))
    # model.add(layers.Conv2DTranspose(128, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))
    model.add(layers.Conv2DTranspose(3, (3, 3),strides=(2, 2), padding='SAME', activation='relu'))
    # 返回3层，两倍的模型
    print(model.summary())
    return model
# image = tf.constant([ 
#  [1,0,0,0,0], 
#  [0,1,0,0,0], 
#  [0,0,1,0,0], 
#  [0,0,0,1,0], 
#  [0,0,0,0,1], 
# ]) 

# # # print(image)
# # image = image[tf.newaxis, ..., tf.newaxis] 
# image = tf.zeros([1,8,8,3])
# # print(image)
# upscaled_patches = tf.image.resize(image,(4,4))

# # img_raw = tf.io.read_file(img_path)
# # # # print(repr(img_raw)[:100]+"...")
# # img_tensor = tf.image.decode_image(img_raw)
# # upscaled_patches = tf.image.resize(img_tensor,(5,5))
# # print(upscaled_patches)

# # print(upscaled_patches.shape)
# # print(upscaled_patches.dtype)
# weights_shape = [4, 4, 3, 3]
# filters = tf.compat.v1.get_variable('weights', shape=weights_shape, collections=['weights', 'variables'])
# # tf.nn.conv2d(input=inputs, filters=filters, strides=[1, *stride, 1], padding=padding)
# print(tf.nn.conv2d(input=upscaled_patches, filters=filters, strides=[1, 3,3, 1], padding='VALID'))
# srcnn_935()
# vgg_7_dec()
# vgg_16()
# vgg_7()
# vgg_max()