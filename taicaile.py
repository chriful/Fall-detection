#!/usr/bin/env python
# coding: utf-8
import paddle as paddle
import paddle.fluid as fluid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from multiprocessing import cpu_count
crop_size = 256
resize_size = 256
BATCH_SIZE = 128
def train_reader(crop_size, resize_size):
    f = open(r'/home/spwux/Wch/datapre/sdtrain.txt',encoding = "utf-8")
    a = list(f)
    for line in a:
        line=line.strip('\n')
        img,label=line.split(' ')
        img = Image.open(img)    
        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = img[(2, 1, 0), :, :] / 255.0
        yield img, int(label)
data_shape = [3, 256, 256]
img = fluid.layers.data(name='img', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64') 
#predict =  convolutional_neural_network(img) 
#optimizer =fluid.optimizer.Adam(learning_rate=0.001)
#optimizer.minimize(avg_cost)
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feeder = fluid.DataFeeder( feed_list=[img, label],place=place)
# In[12]:209494
print("开始train_reader")

ddata_reader=fluid.io.batch(fluid.io.shuffle(train_reader(crop_size, resize_size),buf_size=256),batch_size=32)
#train_reader = paddle.batch(train_reader(crop_size, resize_size),batch_size=BATCH_SIZE)  
print('train_reader好了')
#test_reader = paddle.batch(test_reader(crop_size, resize_size),batch_size=BATCH_SIZE) 
EPOCH_NUM = 20
model_save_dir = "/home/spwux/Wch/standdownModel"
for pass_id in range(EPOCH_NUM):
    print('# 开始训练')
    for batch_id, data in enumerate(ddata_reader()):                               
        print("遍历train_reader的迭代器，并为数据加上索引batch_id")
#        train_cost,train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
#                             feed=feeder.feed(data),                        #喂入一个batch的数据
#                             fetch_list=[avg_cost, acc])                    #fetch均方误差和准确率
        if batch_id % 10 == 0:                                             
#            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' % 
#            (pass_id, batch_id, train_cost[0], train_acc[0]))
            print('oheiyao')
            