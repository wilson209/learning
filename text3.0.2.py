# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:56:14 2019

@author: ASUS
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#使用numpy生成200个随机点,linspace()函数可以生成指定范围内的点，最后列表目的是把数据变成二维数据
x_data = np.linspace(-4.5,4.5,500)[:,np.newaxis]
noise = np.random.normal(0,1,x_data.shape)
y_data = np.square(x_data) + noise - 25
 
#定义两个placeholder(占位符)，规定是1列
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
 
#使用神经网络进行训练测试
 
#定义神经网络的中间层（隐藏层）
#权重
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
#偏置
biases_L1 = tf.Variable(tf.zeros([1,10]))
#传入中间层（隐藏层）的值
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
#由中间层（隐藏层）输出的值,激活函数使用双曲正切函数
L1 = tf.nn.tanh(Wx_plus_b_L1)
 
#定义神经网络的输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = Wx_plus_b_L2
#prediction = tf.nn.tanh(Wx_plus_b_L2)
 
#定义二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降法训练，最小话代价函数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
#初始化变量
init = tf.global_variables_initializer()
#定义会话
with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        #feed操作，需要参数的时候再传入
        sess.run(train_step,feed_dict = {x:x_data,y:y_data})
    
    #获得预测值
    predicton_value = sess.run(prediction,feed_dict = {x:x_data})
    
    #绘图显示
    plt.figure()
    plt.plot(x_data,y_data)
    plt.plot(x_data,predicton_value,'r-',lw = 2)
    plt.show()

