# from tensorflow.python.keras.engine import keras_tensor
# from tensorflow.keras.layers import MultiHeadAttention
# import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Flatten, Dense
import numpy as np
from math import sqrt
import csv
import time
import matplotlib.pyplot as plt
# import tensorflow_addons as tfa

import tensorflow as tf
# from tensorflow.keras.layers import MultiHeadSelfAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Flatten, Dense

from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
from math import sqrt
import numpy as np




def Get_All_Data(TG,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
	#deal with inflow data 处理进站数据
	metro_enter = []
	with open('data/inflowdata/in_'+str(TG)+'min.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line=[int(x) for x in line]
			metro_enter.append(line)
            
        
	def get_train_data_enter(data,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
		data = np.array(data)
		data2 = np.zeros((data.shape[0], data.shape[1]))
		a = np.max(data)
		b = np.min(data)
		for i in range(len(data)):
			for j in range(len(data[0])):
				data2[i, j] = round((data[i, j]-b)/(a-b), 5)
		#不包括第一周和最后一周的数据
		#not include the first week and the last week among the five weeks
		X_train_1 = [[] for i in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		Y_train = []
		for index in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(7):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i,index: index + time_lag-1])
				X_train_1[index-TG_in_one_week].append(temp)
			Y_train.append(data2[:,index + time_lag-1])
		X_train_1,Y_train = np.array(X_train_1), np.array(Y_train)
		print(X_train_1.shape,Y_train.shape)
#        print(X_train_1.shape,Y_train.shape)

		X_test_1 = [[] for i in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1)]
		Y_test = []
		for index in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1):
			for i in range(7):
				temp = data2[i, index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i, index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i, index: index + time_lag-1])
				X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)].append(temp)
			Y_test.append(data2[:, index + time_lag-1])
		X_test_1,Y_test = np.array(X_test_1), np.array(Y_test)
		print(X_test_1.shape, Y_test.shape)

		Y_test_original = []
		for index in range(len(data[0]) - TG_in_one_day*forecast_day_number,len(data[0])-time_lag+1):
			Y_test_original.append(data[:, index + time_lag-1])
		Y_test_original = np.array(Y_test_original)

		print(Y_test_original.shape)

		return X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b

	#获取训练集和测试集，Y_test_original为没有scale之前的原始测试集，评估精度用，a,b分别为最大值和最小值
	#Get the training dataset and the test dataset, Y_test_original is the original test data before scaling, which can be used for evaluation.
	#a and b as the maximum and minimum values, respectively.
	X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b=get_train_data_enter(metro_enter,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week)
	print(a,b)

	#deal with outflow data. Similar with the inflow data while not including the testing data for outflow
	#处理出站数据
	metro_exit = []
	with open('data/outflowdata/out_'+str(TG)+'min.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line = [int(x) for x in line]
			metro_exit.append(line)

	def get_train_data_exit(data,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
		data = np.array(data)
		data2 = np.zeros((data.shape[0], data.shape[1]))
		a = np.max(data)
		b = np.min(data)
		for i in range(len(data)):
			for j in range(len(data[0])):
				data2[i, j]=round((data[i, j]-b)/(a-b), 5)
		X_train_1 = [[] for i in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		for index in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(7):
				temp=data2[i, index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i, index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i, index: index + time_lag-1])
				X_train_1[index-TG_in_one_week].append(temp)
		X_train_1 = np.array(X_train_1)
		print(X_train_1.shape)

		X_test_1 = [[] for i in range(len(data2[0]) - TG_in_one_day*forecast_day_number, len(data2[0])-time_lag+1)]
		for index in range(len(data2[0]) - TG_in_one_day*forecast_day_number, len(data2[0])-time_lag+1):
			for i in range(7):
				temp = data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i, index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i, index: index + time_lag-1])
				X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)].append(temp)
		X_test_1 = np.array(X_test_1)
		print(X_test_1.shape)
		return X_train_1, X_test_1

	X_train_2, X_test_2 = get_train_data_exit(metro_exit, time_lag, TG_in_one_day, forecast_day_number, TG_in_one_week)

	#deal with graph data. involve the adjacency matrix 处理graph图数据，邻接矩阵信息
	adjacency = []
	with open('adjacency.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line = [float(x) for x in line]
			adjacency.append(line)
	adjacency = np.array(adjacency)
	# use adjacency matrix to calculate D_hat**-1/2 * A_hat *D_hat**-1/2
	I = np.matrix(np.eye(7))
	A_hat = adjacency+I
	D_hat = np.array(np.sum(A_hat, axis=0))[0]
	D_hat_sqrt = [sqrt(x) for x in D_hat]
	D_hat_sqrt = np.array(np.diag(D_hat_sqrt))
	D_hat_sqrtm_inv = np.linalg.inv(D_hat_sqrt)# get the D_hat**-1/2 (开方后求逆即为矩阵的-1/2次方)
	#D_A_final = D_hat**-1/2 * A_hat *D_hat**-1/2
	D_A_final = np.dot(D_hat_sqrtm_inv, A_hat)
	D_A_final = np.dot(D_A_final, D_hat_sqrtm_inv)
	print(D_A_final.shape)
	def get_train_data_graph(data,D_A_final,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week,):
		data = np.array(data)
		data2 = np.zeros((data.shape[0], data.shape[1]))
		a = np.max(data)
		b = np.min(data)
		for i in range(len(data)):
			for j in range(len(data[0])):
				data2[i,j]=round((data[i,j]-b)/(a-b),5)
		X_train_1 = [[] for i in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		for index in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(7):
				temp=data2[i,index: index + time_lag-1]
				X_train_1[index-TG_in_one_week].append(temp)
			X_train_1[index-TG_in_one_week] = np.dot(D_A_final, X_train_1[index-TG_in_one_week])
		X_train_1= np.array(X_train_1)
		print(X_train_1.shape)

		X_test_1 = [[] for i in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1)]
		for index in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1):
			for i in range(7):
				temp = data2[i,index: index + time_lag-1]
				X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)].append(temp)
			X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)] = np.dot(D_A_final, X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)])
		X_test_1 = np.array(X_test_1)
		print(X_test_1.shape)

		return X_train_1,X_test_1

	X_train_3, X_test_3 = get_train_data_graph(metro_enter, D_A_final, time_lag, TG_in_one_day, forecast_day_number, TG_in_one_week)

	#deal with meteorology data including the weather and PM data 处理11个指标的天气数据

	Weather = []
	with open('data/meteorology/'+str(TG)+' min after normolization.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line = [float(x) for x in line]
			Weather.append(line)


	def get_train_data_weather_PM(data, time_lag, TG_in_one_day, forecast_day_number, TG_in_one_week,):
		data = np.array(data)
		#不包括第一周和最后一周
		X_train_1 = [[] for i in range(TG_in_one_week, len(data[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		for index in range(TG_in_one_week, len(data[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(len(data)):
				#For meteorology data，we only consider today's data, namely recent pattern. 天气数据只考虑当天的
				X_train_1[index-TG_in_one_week].append(data[i,index: index + time_lag-1])
		X_train_1 = np.array(X_train_1)
		print(X_train_1.shape)

		X_test_1 = [[] for i in range(len(data[0]) - TG_in_one_day*forecast_day_number, len(data[0])-time_lag+1)]
		for index in range(len(data[0]) - TG_in_one_day*forecast_day_number, len(data[0])-time_lag+1):
			for i in range(len(data)):
				X_test_1[index-(len(data[0]) - TG_in_one_day*forecast_day_number)].append(data[i, index: index + time_lag-1])
		X_test_1 = np.array(X_test_1)
		print(X_test_1.shape)
		return X_train_1,X_test_1
            

	X_train_4, X_test_4 = get_train_data_weather_PM(Weather, time_lag, TG_in_one_day, forecast_day_number, TG_in_one_week)

	other = []
	with open('data/other/other_'+str(TG)+'min.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line = [float(x) for x in line]
			other.append(line)

	def get_train_data_other(data, time_lag, TG_in_one_day, forecast_day_number, TG_in_one_week,):
		data = np.array(data)
		#不包括第一周和最后一周
		X_train_1 = [[] for i in range(TG_in_one_week, len(data[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		for index in range(TG_in_one_week, len(data[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(len(data)):
				#For meteorology data，we only consider today's data, namely recent pattern. 天气数据只考虑当天的
				X_train_1[index-TG_in_one_week].append(data[i,index: index + time_lag-1])
		X_train_1 = np.array(X_train_1)
		print(X_train_1.shape)

		X_test_1 = [[] for i in range(len(data[0]) - TG_in_one_day*forecast_day_number, len(data[0])-time_lag+1)]
		for index in range(len(data[0]) - TG_in_one_day*forecast_day_number, len(data[0])-time_lag+1):
			for i in range(len(data)):
				X_test_1[index-(len(data[0]) - TG_in_one_day*forecast_day_number)].append(data[i, index: index + time_lag-1])
		X_test_1 = np.array(X_test_1)
		print(X_test_1.shape)
		return X_train_1,X_test_1
    
	X_train_5, X_test_5 = get_train_data_other(other, time_lag, TG_in_one_day, forecast_day_number, TG_in_one_week)

	return X_train_1, Y_train, X_test_1, Y_test, Y_test_original, a, b, X_train_2, X_test_2, X_train_3, X_test_3, X_train_4, X_test_4, X_train_5, X_test_5


# def main():
#     Get_All_Data(TG=10,time_lag=10,TG_in_one_day=108,forecast_day_number=5,TG_in_one_week=540)


# if __name__ == "__main__":
#     main()




import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

import tensorflow as tf
# # pip install tensorflow-gpu
# # 启用内存增长
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# # 或者禁用内存增长
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, False)

# import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import os
np.random.seed(1)
tf.random.set_seed(2)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def Unit(x, filters, pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
    out = keras.layers.add([res, out])
    return out

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

#自注意力机制
class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        self.time_steps = input_shape[1]
        self.features = input_shape[2]

        self.W = self.add_weight(shape=(self.features, self.features),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='attention_weights')
        self.b = self.add_weight(shape=(self.features,),
                                 initializer='zeros',
                                 trainable=True,
                                 name='attention_bias')

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        attention_scores = K.dot(inputs, self.W)
        attention_scores = K.tanh(attention_scores + self.b)
        attention_weights = K.softmax(attention_scores, axis=1)
        attended_sequence = K.batch_dot(attention_weights, inputs, axes=[1, 1])

        return attended_sequence

    def compute_output_shape(self, input_shape):
        return input_shape

#序列注意力机制
def attention_3d_block(inputs, timesteps):
    a = Permute((2, 1))(inputs)
    a = Dense(timesteps, activation='linear')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

#局部注意力机制
def local_attention_2d(x, window_size):
    attention_weights = Conv2D(1, (window_size, window_size), padding='same', activation='softmax')(x)
    attended_sequence = Multiply()([x, attention_weights])
    return attended_sequence

# 时序注意力机制
def temporal_attention_1d(x, window_size):
    attention_weights = Conv1D(1, window_size, padding='same', activation='softmax')(x)
    attended_sequence = Multiply()([x, attention_weights])
    return attended_sequence

#空间注意力机制
def spatial_attention_2d(x):
    attention_weights = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    attended_features = Multiply()([x, attention_weights])
    return attended_features


def transformer_block(inputs, head_size, num_heads):
    attention = tfa.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    add_attention = tf.keras.layers.Add()([attention, inputs])
    attention_output = tf.keras.layers.LayerNormalization()(add_attention)
    feed_forward = tf.keras.layers.Dense(units=inputs.shape[-1], activation='relu')(attention_output)
    feed_forward_output = tf.keras.layers.Dense(units=inputs.shape[-1])(feed_forward)
    add_feed_forward = tf.keras.layers.Add()([feed_forward_output, attention_output])
    output = tf.keras.layers.LayerNormalization()(add_feed_forward)
    return output

    
def multi_input_model(time_lag):
    input1_ = Input(shape=(7, time_lag-1, 3), name='input1')
    input2_ = Input(shape=(7, time_lag-1, 3), name='input2')
    input3_ = Input(shape=(7, time_lag-1, 1), name='input3')
    input4_ = Input(shape=(13, time_lag-1, 1), name='input4')
    input5_ = Input(shape=(10, time_lag-1, 1), name='input5')
    
    x1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input1_)
    x1 = Unit(x1, 32)
    x1 = Unit(x1, 64, pool=True)
    x1 = Reshape((3, 4 * 64))(x1)
    print(x1.shape)
    x1 = temporal_attention_1d(x1, window_size=5)
    print(x1.shape)
    x1 = Reshape((4, 3, 64))(x1)
    print(x1.shape)
    x1 = Flatten()(x1)
    x1 = Dense(7)(x1)
    print(x1.shape)

    x2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input2_)
    x2 = Unit(x2, 32)
    x2 = Unit(x2, 64, pool=True)
    x2 = Reshape((3, 4 * 64))(x2)
    print(x2.shape)
    x2 = temporal_attention_1d(x2, window_size=5)
    print(x2.shape)
    x2 = Reshape((4, 3, 64))(x2)
    print(x2.shape)
    x2 = Flatten()(x2)
    x2 = Dense(7)(x2)
    print(x2.shape)

    x3 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input3_)
    x3 = spatial_attention_2d(x3)
    x3 = Unit(x3, 32)
    x3 = Unit(x3, 64, pool=True)
    x3 = spatial_attention_2d(x3)
    print(x3.shape)
    x3 = Flatten()(x3)
    x3 = Dense(7)(x3)
    print(x3.shape)

    x4 = Flatten()(input4_)
    x4 = Dense(7)(x4)
    x4 = Reshape(target_shape=(7, 1))(x4)
    x4 = LSTM(128, return_sequences=True, input_shape=(7, 1))(x4)
    print(x4.shape)
    # x4 = SelfAttention()(x4)
    print(x4.shape)
    x4 = LSTM(7, return_sequences=False)(x4)
    x4 = Dense(7)(x4)
    print(x4.shape)
    
    x5 = Flatten()(input5_)
    x5 = Dense(7)(x5)
    x5 = Reshape(target_shape=(7, 1))(x5)
    x5 = LSTM(128, return_sequences=True, input_shape=(7, 1))(x5)
    print(x5.shape)
    # x5 = SelfAttention()(x5)
    print(x5.shape)
    x5 = LSTM(7, return_sequences=False)(x5)
    x5 = Dense(7)(x5)
    print(x5.shape)

    out = keras.layers.add([x1, x2, x3, x4, x5])
    out = Reshape(target_shape=(7, 1))(out)
    out = LSTM(128, return_sequences=True,input_shape=(7, 1))(out)
    out = attention_3d_block(out, 7)
    out = Flatten()(out)
    out = Dense(7)(out)
    
    model = Model(inputs=[input1_, input2_, input3_,input4_,input5_], outputs=[out])
    return model

    model = multi_input_model(time_lag=6)
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    history = model.fit([X_train_1, X_train_2, X_train_3, X_train_4, X_train_5], [Y_train], validation_data=([X_test_1, X_test_2, X_test_3, X_test_4, X_test_5], [Y_test]), batch_size=128, epochs=100, verbose=1)

def build_model(X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, Y_train, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, Y_test, Y_test_original, batch_size, epochs, a, time_lag):
    X_train_1 = X_train_1.reshape(X_train_1.shape[0], 7, time_lag-1, 3)
    X_train_2 = X_train_2.reshape(X_train_2.shape[0], 7, time_lag-1, 3)
    X_train_3 = X_train_3.reshape(X_train_3.shape[0], 7, time_lag-1, 1)
    X_train_4 = X_train_4.reshape(X_train_4.shape[0], 13, time_lag-1, 1)
    X_train_5 = X_train_5.reshape(X_train_5.shape[0], 10, time_lag-1, 1)
    Y_train = Y_train.reshape(Y_train.shape[0], 7)

    X_test_1 = X_test_1.reshape(X_test_1.shape[0], 7, time_lag-1, 3)
    X_test_2 = X_test_2.reshape(X_test_2.shape[0], 7, time_lag-1, 3)
    X_test_3 = X_test_3.reshape(X_test_3.shape[0], 7, time_lag-1, 1)
    X_test_4 = X_test_4.reshape(X_test_4.shape[0], 13, time_lag-1, 1)
    X_test_5 = X_test_5.reshape(X_test_5.shape[0], 10, time_lag-1, 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 7)

    history = None
    
    train_loss = []
    val_loss = [] 
    if epochs == 150:
        model = multi_input_model(time_lag)
        model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
        history = model.fit([X_train_1, X_train_2, X_train_3, X_train_4, X_train_5], Y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=False, validation_data=([X_test_1, X_test_2, X_test_3, X_test_4, X_test_5], Y_test))
        model.fit([X_train_1, X_train_2, X_train_3, X_train_4, X_train_5], Y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=False)
        output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4, X_test_5], batch_size=batch_size)
        
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12  # 可以设置字体大小
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    else:
        model = load_model('testresult_MMM/'+str(epochs-10)+'-model-with-graph.h5')
        history = model.fit([X_train_1, X_train_2, X_train_3, X_train_4, X_train_5], Y_train, batch_size=batch_size, epochs=10, verbose=2, shuffle=False, validation_data=([X_test_1, X_test_2, X_test_3, X_test_4, X_test_5], Y_test))
        model.fit([X_train_1, X_train_2, X_train_3, X_train_4, X_train_5], Y_train, batch_size=batch_size, epochs=10, verbose=2, shuffle=False)
        output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4, X_test_5], batch_size=batch_size)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12  # 可以设置字体大小
        plt.figure(figsize=(5, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    np.savetxt("training_loss.csv", history.history['loss'], delimiter=",")
    if 'val_loss' in history.history:
        np.savetxt("validation_loss.csv", history.history['val_loss'], delimiter=",")

    predictions = np.zeros((output.shape[0], output.shape[1]))
    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            predictions[i, j] = round(output[i, j]*a, 0)
            if predictions[i, j] < 0:
                predictions[i, j] = 0
    RMSE, R2, MAE, WMAPE, accuracy = evaluate_performance(Y_test_original, predictions)
    plot_model(model, to_file='model.png', show_shapes=True)
    


def save_data(path, model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, accuracy,Run_epoch):
    print(Run_epoch)
    RMSE_ALL = []
    R2_ALL = []
    MAE_ALL = []
    WMAPE_ALL = []
    Average_train_time = []
    accuracy_ALL = []
    RMSE_ALL.append(RMSE)
    R2_ALL.append(R2)
    MAE_ALL.append(MAE)
    WMAPE_ALL.append(WMAPE)
    accuracy_ALL.append(accuracy) 
    model.save(os.path.join(path,str(Run_epoch) + '-model-with-graph.h5'))
    np.savetxt(os.path.join(path, str(Run_epoch) + '-RMSE_ALL.txt'), RMSE_ALL)
    np.savetxt(os.path.join(path, str(Run_epoch) + '-R2_ALL.txt'), R2_ALL)
    np.savetxt(os.path.join(path, str(Run_epoch) + '-WMAPE_ALL.txt'), WMAPE_ALL)
    np.savetxt(os.path.join(path, str(Run_epoch) + '-accuracy_ALL.txt'), accuracy_ALL)
    with open(os.path.join(path, str(Run_epoch) + '-predictions.csv'), 'w') as file:
        predictions = predictions.tolist()
        for i in range(len(predictions)):
            file.write(str(predictions[i]).replace("'", "").replace("[", "").replace("]", "") + "\n")
    with open(os.path.join(path, str(Run_epoch) + '-Y_test_original.csv'), 'w') as file:
        Y_test_original = Y_test_original.tolist()
        for i in range(len(Y_test_original)):
            file.write(str(Y_test_original[i]).replace("'", "").replace("[", "").replace("]", "") + "\n")
    duration_time = time.time() - global_start_time
    Average_train_time.append(duration_time)
    np.savetxt(os.path.join(path, str(Run_epoch) + '-Average_train_time.txt'), Average_train_time)
    print('total training time(s):', duration_time)


# def weighted_mean_absolute_percentage_error(Y_true, Y_pred):
#     total_sum = np.sum(Y_true)
#     average = []
#     for i in range(len(Y_true)):
#         for j in range(len(Y_true[0])):
#             if Y_true[i][j] > 0:
#                 temp = (Y_true[i][j] / total_sum) * np.abs((Y_true[i][j] - Y_pred[i][j]) / Y_true[i][j])
#                 average.append(temp)
#     return np.sum(average)

# def weighted_mean_absolute_percentage_error(Y_test_original, predictions):
#     total_sum = np.sum(Y_test_original)
#     average = []
#     for i in range(len(Y_test_original)):
#         for j in range(len(Y_test_original[0])):
#             if Y_test_original[i][j] > 0:
#                 temp = (Y_test_original[i][j] / total_sum) * np.abs((Y_test_original[i][j] - predictions[i][j]) / Y_true[i][j])
#                 average.append(temp)
#     return np.sum(average)

# 一维组
def weighted_mean_absolute_percentage_error(Y_test_original, predictions):
    # 计算总和作为权重
    total_sum = np.sum(Y_test_original)
    
    # 初始化平均误差列表
    average = []
    
    # 遍历数组中的每个元素
    for i in range(len(Y_test_original)):
        # 检查实际值是否大于0，避免除以0
        if Y_test_original[i] > 0:
            # 计算加权平均绝对百分比误差
            error = (Y_test_original[i] / total_sum) * np.abs((Y_test_original[i] - predictions[i]) / Y_test_original[i])
            average.append(error)
    
    # 计算所有误差的平均值
    WMAPE = np.mean(average) * 100  # 将结果转换为百分比形式
    return WMAPE


def mean_absolute_percentage_error(Y_test_original, predictions):
    MAPE = (np.abs((Y_test_original-predictions)/predictions)).mean() * 100

def calculate_accuracy(Y_test_original, predictions):
    deviation = np.abs(Y_test_original, predictions)
    correct_predictions = deviation <= 5
    accuracy = np.mean(correct_predictions.astype(float))
    return accuracy

# def evaluate_performance(Y_test_original, predictions):
#     RMSE = sqrt(mean_squared_error(Y_test_original, predictions))
#     print('RMSE is: ' + str(RMSE))
#     R2 = r2_score(Y_test_original, predictions)
#     print("R2 is: " + str(R2))
#     MAE = mean_absolute_error(Y_test_original, predictions)
#     print("MAE is: " + str(MAE))
#     WMAPE = weighted_mean_absolute_percentage_error(Y_test_original, predictions)
#     print("WMAPE is: " + str(WMAPE))
#     accuracy = calculate_accuracy(Y_test_original, predictions)
#     print("准确度为：" + str(accuracy))
#     return RMSE, R2, MAE, WMAPE, accuracy

# GYA
def evaluate_performance(Y_test_original, predictions):
    RMSE = sqrt(mean_squared_error(Y_test_original[:,0], predictions[:,0]))
    print('RMSE is: ' + str(RMSE))
    R2 = r2_score(Y_test_original[:,0], predictions[:,0])
    print("R2 is: " + str(R2))
    MAE = mean_absolute_error(Y_test_original[:,0], predictions[:,0])
    print("MAE is: " + str(MAE))
    WMAPE = weighted_mean_absolute_percentage_error(Y_test_original[:,0], predictions[:,0])
    print("WMAPE is: " + str(WMAPE))
    accuracy = calculate_accuracy(Y_test_original[:,0], predictions[:,0])
    print("准确度为：" + str(accuracy))
    # MAPE = mean_absolute_percentage_error(Y_test_original[:,0], predictions[:,0])
    # print("MAPE:" +str(MAPE))
    return RMSE, R2, MAE, WMAPE, accuracy

# YIN
# def evaluate_performance(Y_test_original, predictions):
#     RMSE = sqrt(mean_squared_error(Y_test_original[:,1], predictions[:,1]))
#     print('RMSE is: ' + str(RMSE))
#     R2 = r2_score(Y_test_original[:,1], predictions[:,1])
#     print("R2 is: " + str(R2))
#     MAE = mean_absolute_error(Y_test_original[:,1], predictions[:,1])
#     print("MAE is: " + str(MAE))
#     WMAPE = weighted_mean_absolute_percentage_error(Y_test_original[:,1], predictions[:,1])
#     print("WMAPE is: " + str(WMAPE))
#     accuracy = calculate_accuracy(Y_test_original[:,1], predictions[:,1])
#     print("准确度为：" + str(accuracy))
#     # MAPE = mean_absolute_percentage_error(Y_test_original[:,0], predictions[:,0])
#     # print("MAPE:" +str(MAPE))
#     return RMSE, R2, MAE, WMAPE, accuracy

# ATAGA
# def evaluate_performance(Y_test_original, predictions):
#     RMSE = sqrt(mean_squared_error(Y_test_original[:,2], predictions[:,2]))
#     print('RMSE is: ' + str(RMSE))
#     R2 = r2_score(Y_test_original[:,2], predictions[:,2])
#     print("R2 is: " + str(R2))
#     MAE = mean_absolute_error(Y_test_original[:,2], predictions[:,2])
#     print("MAE is: " + str(MAE))
#     WMAPE = weighted_mean_absolute_percentage_error(Y_test_original[:,2], predictions[:,2])
#     print("WMAPE is: " + str(WMAPE))
#     accuracy = calculate_accuracy(Y_test_original[:,2], predictions[:,2])
#     print("准确度为：" + str(accuracy))
#     # MAPE = mean_absolute_percentage_error(Y_test_original[:,0], predictions[:,0])
#     # print("MAPE:" +str(MAPE))
#     return RMSE, R2, MAE, WMAPE, accuracy

# IGONO
# def evaluate_performance(Y_test_original, predictions):
#     RMSE = sqrt(mean_squared_error(Y_test_original[:,3], predictions[:,3]))
#     print('RMSE is: ' + str(RMSE))
#     R2 = r2_score(Y_test_original[:,3], predictions[:,3])
#     print("R2 is: " + str(R2))
#     MAE = mean_absolute_error(Y_test_original[:,3], predictions[:,3])
#     print("MAE is: " + str(MAE))
#     WMAPE = weighted_mean_absolute_percentage_error(Y_test_original[:,3], predictions[:,3])
#     print("WMAPE is: " + str(WMAPE))
#     accuracy = calculate_accuracy(Y_test_original[:,3], predictions[:,3])
#     print("准确度为：" + str(accuracy))
#     # MAPE = mean_absolute_percentage_error(Y_test_original[:,0], predictions[:,0])
#     # print("MAPE:" +str(MAPE))
#     return RMSE, R2, MAE, WMAPE, accuracy

# LMN
# def evaluate_performance(Y_test_original, predictions):
#     RMSE = sqrt(mean_squared_error(Y_test_original[:,4], predictions[:,4]))
#     print('RMSE is: ' + str(RMSE))
#     R2 = r2_score(Y_test_original[:,4], predictions[:,4])
#     print("R2 is: " + str(R2))
#     MAE = mean_absolute_error(Y_test_original[:,4], predictions[:,4])
#     print("MAE is: " + str(MAE))
#     WMAPE = weighted_mean_absolute_percentage_error(Y_test_original[:,4], predictions[:,4])
#     print("WMAPE is: " + str(WMAPE))
#     accuracy = calculate_accuracy(Y_test_original[:,4], predictions[:,4])
#     print("准确度为：" + str(accuracy))
#     # MAPE = mean_absolute_percentage_error(Y_test_original[:,4], predictions[:,4])
#     # print("MAPE:" +str(MAPE))
#     return RMSE, R2, MAE, WMAPE, accuracy

# IDUMA
# def evaluate_performance(Y_test_original, predictions):
#     RMSE = sqrt(mean_squared_error(Y_test_original[:,5], predictions[:,5]))
#     print('RMSE is: ' + str(RMSE))
#     R2 = r2_score(Y_test_original[:,5], predictions[:,5])
#     print("R2 is: " + str(R2))
#     MAE = mean_absolute_error(Y_test_original[:,5], predictions[:,5])
#     print("MAE is: " + str(MAE))
#     WMAPE = weighted_mean_absolute_percentage_error(Y_test_original[:,5], predictions[:,5])
#     print("WMAPE is: " + str(WMAPE))
#     accuracy = calculate_accuracy(Y_test_original[:,5], predictions[:,5])
#     print("准确度为：" + str(accuracy))
#     # MAPE = mean_absolute_percentage_error(Y_test_original[:,0], predictions[:,0])
#     # print("MAPE:" +str(MAPE))
#     return RMSE, R2, MAE, WMAPE, accuracy

# VIBOS
# def evaluate_performance(Y_test_original, predictions):
#     RMSE = sqrt(mean_squared_error(Y_test_original[:,6], predictions[:,6]))
#     print('RMSE is: ' + str(RMSE))
#     R2 = r2_score(Y_test_original[:,6], predictions[:,6])
#     print("R2 is: " + str(R2))
#     MAE = mean_absolute_error(Y_test_original[:,6], predictions[:,6])
#     print("MAE is: " + str(MAE))
#     WMAPE = weighted_mean_absolute_percentage_error(Y_test_original[:,6], predictions[:,6])
#     print("WMAPE is: " + str(WMAPE))
#     accuracy = calculate_accuracy(Y_test_original[:,6], predictions[:,6])
#     print("准确度为：" + str(accuracy))
#     # MAPE = mean_absolute_percentage_error(Y_test_original[:,6], predictions[:,6])
#     # print("MAPE:" +str(MAPE))
#     return RMSE, R2, MAE, WMAPE, accuracy

X_train_1, Y_train, X_test_1, Y_test, Y_test_original, a, b, X_train_2, X_test_2, X_train_3, X_test_3, X_train_4, X_test_4, X_train_5, X_test_5 = \
    Get_All_Data(TG=60, time_lag=6, TG_in_one_day=24, forecast_day_number=90, TG_in_one_week=2160)

    # Get_All_Data(TG=120, time_lag=6, TG_in_one_day=12, forecast_day_number=7, TG_in_one_week=84)

    # Get_All_Data(TG=60, time_lag=6, TG_in_one_day=24, forecast_day_number=7, TG_in_one_week=168)
    # Get_All_Data(TG=180, time_lag=6, TG_in_one_day=8, forecast_day_number=7, TG_in_one_week=56)
    # Get_All_Data(TG=60, time_lag=6, TG_in_one_day=144, forecast_day_number=5, TG_in_one_week=1008)

#10min:10,6,108,5,540,eopch=200
#15min:15,6,72,5,360 eopch=140
#30min:30,6,36,5,180 eopch=200
#60min:60,6,18,5,90 eopch=235

#60min:60,6,24,7,168 eopch=235
#120min:120,6,12,7,84 eopch=235 
#180min:180,6,8,7,56 eopch=235 


# TG=180
# Run_epoch = 50  # first training 50 epoch, and then add 10 epoch every time 初始训练epoch，以后每次加10，运行15次
# for i in range(15):
# # for i in range(20):
#     model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, accuracy = build_model(
#     X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, Y_train, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, Y_test,
#     Y_test_original, batch_size=64, epochs=Run_epoch, a=a, time_lag=6)
#     global_start_time = time.time()
#     save_data("testresult_MM_180/", model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, accuracy, Run_epoch)
# # print('MM')
#     Run_epoch += 10
    # Run_epoch += 10
# Run_epoch = 50  # first training 50 epoch, and then add 10 epoch every time 初始训练epoch，以后每次加10，运行15次
# for i in range(15):
# # for i in range(20):
#     model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, accuracy = build_model(
#     X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, Y_train, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, Y_test,
#     Y_test_original, batch_size=64, epochs=Run_epoch, a=a, time_lag=6)
#     global_start_time = time.time()
#     save_data("testresult_MM/"+str(TG), model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, accuracy, Run_epoch)
# # print('MM')
#     Run_epoch += 10
#     # Run_epoch += 10

# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['mse'], label='Training MSE')
# plt.plot(history.history['val_mse'], label='Validation MSE')
# plt.title('Training and Validation MSE')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.legend()

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test_original[690:,0], label='Actual')
# plt.plot(predictions[690:,0], label='Predicted')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('GYA-ZGGG Predicted Results')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test_original[690:,1], label='Actual')
# plt.plot(predictions[690:,1], label='Predicted')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('ZGGG-YIN Predicted Results')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test_original[690:,2], label='Actual')
# plt.plot(predictions[690:,2], label='Predicted')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('ATAGA-ZGGG Predicted Results')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test_original[690:,3], label='Actual')
# plt.plot(predictions[690:,3], label='Predicted')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('IGONO-ZGGG Predicted Results')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test_original[690:,4], label='Actual')
# plt.plot(predictions[690:,4], label='Predicted')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('ZGGG-LMN Predicted Results')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test_original[690:,5], label='Actual')
# plt.plot(predictions[690:,5], label='Predicted')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('IDUMA-ZGGG Predicted Results')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test_original[690:,6], label='Actual')
# plt.plot(predictions[690:,6], label='Predicted')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('ZGGG-VIBOS Predicted Results')
# plt.legend()
# plt.show()
# plt.plot(epochs, train_loss, 'b', label='Training Loss')
# plt.plot(epochs, val_loss, 'r', label='Validation Loss')
# plt.plot(Run_epoch, train_loss, 'b', label='Training Loss')
# plt.plot(Run_epoch, val_loss, 'r', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

Run_epoch = 150 # first training 50 epoch, and then add 10 epoch every time 初始训练epoch，以后每次加10，运行15次
for i in range(2):
# for i in range(20):
    model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, accuracy = build_model(
        X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, Y_train, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, Y_test,
        Y_test_original, batch_size=64, epochs=Run_epoch, a=a, time_lag=6)
    # global_start_time = time.time()    
        # model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, accuracy = build_model(X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, Y_train, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, Y_test, Y_test_original, batch_size=64, epochs=Run_epoch, a=a, time_lag=6)
        # global_start_time = time.time()
    save_data("testresult_MMM", model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, accuracy, Run_epoch)
# print('MM')
    Run_epoch += 10

