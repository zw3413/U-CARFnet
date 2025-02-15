#import tensorflow as tf
#from tensorflow import keras
from keras import backend as K
from keras import  layers 
#(Activation, Add, Concatenate, Conv1D, Conv2D, Dense,GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda,Reshape, multiply)
import math


def channel_attention(input_feature, ratio=8, name=""):
	channel = K.int_shape(input_feature)[-1]
	
	shared_layer_one = layers.Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_one_"+str(name))
	shared_layer_two = layers.Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_two_"+str(name))
	
	avg_pool = layers.GlobalAveragePooling2D()(input_feature)    
	max_pool = layers.GlobalMaxPooling2D()(input_feature)

	avg_pool = layers.Reshape((1,1,channel))(avg_pool)
	max_pool = layers.Reshape((1,1,channel))(max_pool)

	avg_pool = shared_layer_one(avg_pool)
	max_pool = shared_layer_one(max_pool)

	avg_pool = shared_layer_two(avg_pool)
	max_pool = shared_layer_two(max_pool)
	
	cbam_feature = layers.Add()([avg_pool,max_pool])
	cbam_feature = layers.Activation('sigmoid')(cbam_feature)
	
	
	return layers.multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, name=""):
	kernel_size = 7

	cbam_feature = input_feature
	
	avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	concat = layers.Concatenate(axis=3)([avg_pool, max_pool])

	cbam_feature = layers.Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False,
					name = "spatial_attention_"+str(name))(concat)	
	cbam_feature = layers.Activation('sigmoid')(cbam_feature)
		
	return layers.multiply([input_feature, cbam_feature])

# Squeeze-and-Excitation Block 压缩激励模块，通过学习通道间的依赖关系来增强特征表示
def se_block(input_feature, ratio=16, name=""):
	channel = K.int_shape(input_feature)[-1]

	se_feature = layers.GlobalAveragePooling2D()(input_feature)
	se_feature = layers.Reshape((1, 1, channel))(se_feature)

	se_feature = layers.Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=False,
					   name = "se_block_one_"+str(name))(se_feature)
					   
	se_feature = layers.Dense(channel,
					   kernel_initializer='he_normal',
					   use_bias=False,
					   name = "se_block_two_"+str(name))(se_feature)
	se_feature = layers.Activation('sigmoid')(se_feature)

	se_feature = layers.multiply([input_feature, se_feature])
	return se_feature

# Convolutional Block Attention Module  卷积块注意力模块，结合了通道注意力和空间注意力
def cbam_block(cbam_feature, ratio=8, name=""):
	cbam_feature = channel_attention(cbam_feature, ratio, name=name)
	cbam_feature = spatial_attention(cbam_feature, name=name)
	return cbam_feature

# Efficient Channel Attention  高效通道注意力，通过轻量级的方式增强通道注意力
def eca_block(input_feature, b=1, gamma=2, name=""):
	channel = K.int_shape(input_feature)[-1]
	kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
	kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
	
	avg_pool = layers.GlobalAveragePooling2D()(input_feature)
	
	x = layers.Reshape((-1,1))(avg_pool)
	x = layers.Conv1D(1, kernel_size=kernel_size, padding="same", name = "eca_layer_"+str(name), use_bias=False,)(x)
	x = layers.Activation('sigmoid')(x)
	x = layers.Reshape((1, 1, -1))(x)

	output = layers.multiply([input_feature,x])
	return output
