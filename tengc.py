#!/usr/bin/env python
# coding: utf-8



#导入需要的包
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
#用于训练的数据提供器
#train_reader = paddle.batch(
#    paddle.reader.shuffle(paddle.dataset.cifar.train10(), 
#                          buf_size=128*100),           
#    batch_size=BATCH_SIZE)                                
#用于测试的数据提供器
#test_reader = paddle.batch(
#    paddle.dataset.cifar.test10(),                            
#    batch_size=BATCH_SIZE)                                

#def train_mapper(sample):
#    img_path, label,crop_size, resize_size = sample
#    
#    try:
#        img = Image.open(img_path)
#        # 统一图片大小
#        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
#       
#        # 把图片转换成numpy值
#        img = np.array(img).astype(np.float32)
#        # 转换成CHW
#        img = img.transpose((2, 0, 1))
#        # 转换成BGR
#        img = img[(2, 1, 0), :, :] / 255.0
##        print('调用tain_mapper')
##        
#      
#        return img, int(label)
#    except:
#       
#        print("%s 该图片错误，请删除该图片并重新创建图像数据列表" % img_path)

def train_reader(crop_size, resize_size):
#    dataset = ImageFolder('/home/data3/wch/sdtrain')

    num0=0
    num1=0
#        for line in dataset.imgs:
#            img=line[0]
#            label=line[1]
#            if label==0:
#                num0+=1
#            else:
#                num1+=1
    import os
    f = open(r'/home/spwux/Wch/datapre/sdtrain.txt',encoding = "utf-8")
    a = list(f)
#        
#        img=[]
#        labellabel=[]
    for line in a:
        line=line.strip('\n')
        img,label=line.split(' ')
        img = Image.open(img)
        # 统一图片大小
        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
       
        # 把图片转换成numpy值
        img = np.array(img).astype(np.float32)
        # 转换成CHW
        img = img.transpose((2, 0, 1))
        # 转换成BGR
        img = img[(2, 1, 0), :, :] / 255.0
#        print('调用tain_mapper')
#        
      
     

        yield img, int(label)
#    f.close()
#    print(num0,num1)
#    print('train_reader调用')
    

#def train_mapper(sample):
#    img_path, label,crop_size, resize_size = sample
#    
#    try:
#        img = Image.open(img_path)
#        # 统一图片大小
#        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
#       
#        # 把图片转换成numpy值
#        img = np.array(img).astype(np.float32)
#        # 转换成CHW
#        img = img.transpose((2, 0, 1))
#        # 转换成BGR
#        img = img[(2, 1, 0), :, :] / 255.0
##        print('调用tain_mapper')
##        
#      
#        return img, int(label)
#    except:
#       
#        print("%s 该图片错误，请删除该图片并重新创建图像数据列表" % img_path)

#def train_reader(crop_size, resize_size):
##    dataset = ImageFolder('/home/data3/wch/sdtrain')
#    def reader():
#        num0=0
#        num1=0
##        for line in dataset.imgs:
##            img=line[0]
##            label=line[1]
##            if label==0:
##                num0+=1
##            else:
##                num1+=1
#        import os
#        f = open(r'/home/spwux/Wch/datapre/sdtrain.txt',encoding = "utf-8")
#        a = list(f)
##        
##        img=[]
##        labellabel=[]
#        for line in a:
#            line=line.strip('\n')
#            img,label=line.split(' ')
#
#            yield img, label ,crop_size, resize_size
#        f.close()
#        print(num0,num1)
#        print('train_reader调用')
#    
#    return paddle.reader.xmap_readers(train_mapper, reader,cpu_count(),60000)



def test_mapper(sample):
    img, label, crop_size, resize_size = sample
    img = Image.open(img)
    # 统一图像大小
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, int(label)



def test_reader(crop_size, resize_size):
	dataset = ImageFolder('/home/data3/wch/sdtest')
	def reader():
		num0=0
		num1=0
		for line in dataset.imgs:
			img=line[0]
			label=line[1]
			if label==0:
				num0+=1
			else:
				num1+=1
			yield img, label,crop_size, resize_size
		print(num0,num1)
	return paddle.reader.xmap_readers(test_mapper, reader,cpu_count(),30000)
  




# 
# 

# In[4]:


def convolutional_neural_network(img):
    # 第一个卷积-池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,         # 输入图像
        filter_size=5,     # 滤波器的大小
        num_filters=20,    # filter 的数量。它与输出的通道相同
        pool_size=2,       # 池化核大小2*2
        pool_stride=2,     # 池化步长
        act="relu")        # 激活类型
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # 第二个卷积-池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.layers.batch_norm(conv_pool_2)
    # 第三个卷积-池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_2,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # 以softmax为激活函数的全连接输出层，10类数据输出10个数字
    prediction = fluid.layers.fc(input=conv_pool_3, size=2, act='softmax')
    return prediction


# **（2）定义数据**

# In[5]:


#定义输入数据
data_shape = [3, 256, 256]
img = fluid.layers.data(name='img', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')



# 获取分类器，用cnn进行分类
predict =  convolutional_neural_network(img)


# **（4）定义损失函数和准确率**
# 


# In[7]:


# 获取损失函数和准确率
#cost = fluid.layers.softmax_cross_entropy_with_logits(input=predict, label=label)
cost = fluid.layers.cross_entropy(input=predict, label=label) # 交叉熵
#cost=predict*predict
avg_cost = fluid.layers.mean(cost)                            # 计算cost中所有元素的平均值
acc = fluid.layers.accuracy(input=predict, label=label)       #使用输入和标签计算准确率




# 定义优化方法
optimizer =fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)
print("完成")

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 创建执行器，初始化参数
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# **(2)定义数据映射器**
# 
# DataFeeder 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 Executor
#  
# 

# In[10]:


feeder = fluid.DataFeeder( feed_list=[img, label],place=place)



# In[12]:209494
print("开始train_reader")
#train_reader(crop_size, resize_size)
#
#train_reader = paddle.batch(train_reader(crop_size, resize_size),batch_size=BATCH_SIZE)
#sf=paddle.reader.shuffle(train_reader(crop_size, resize_size),buf_size=60000)
#print('nenene')
#
#
#
#
#train_reader = paddle.batch(sf,batch_size=BATCH_SIZE)
ddata_reader=fluid.io.batch(fluid.io.shuffle(train_reader(crop_size, resize_size),buf_size=256),batch_size=32)
#train_reader = paddle.batch(train_reader(crop_size, resize_size),batch_size=BATCH_SIZE)  
print('train_reader好了')
#test_reader = paddle.batch(test_reader(crop_size, resize_size),batch_size=BATCH_SIZE) 
EPOCH_NUM = 20
model_save_dir = "/home/spwux/Wch/standdownModel"
# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)
for pass_id in range(EPOCH_NUM):
    print('# 开始训练')
    for batch_id, data in enumerate(ddata_reader()):                               
#        print("遍历train_reader的迭代器，并为数据加上索引batch_id")
        train_cost,train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
                             feed=feeder.feed(data),                        #喂入一个batch的数据
                             fetch_list=[avg_cost, acc])                    #fetch均方误差和准确率
        

        
        
        #每100次batch打印一次训练、进行一次测试
        if batch_id % 10 == 0:                                             
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' % 
            (pass_id, batch_id, train_cost[0], train_acc[0]))
            

    # 开始测试
  
    test_costs = []                                                         #测试的损失值
    test_accs = []                                                          #测试的准确率
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,                 #执行测试程序
                                      feed=feeder.feed(data),               #喂入数据
                                      fetch_list=[avg_cost, acc])           #fetch 误差、准确率
        test_costs.append(test_cost[0])                                     #记录每个batch的误差
        test_accs.append(test_acc[0])                                       #记录每个batch的准确率
    
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))                         #计算误差平均值（误差和/误差的个数）
    test_acc = (sum(test_accs) / len(test_accs))                            #计算准确率平均值（ 准确率的和/准确率的个数）
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
    
#保存模型
# 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print ('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,
                              ['images'],
                              [predict],
                              exe)
print('训练模型保存完成！')



# # **Step5.模型预测**
# 
# **（1）创建预测用的Executor**

# In[13]:


infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope() 


# **(2)图片预处理**
# 
# 在预测之前，要对图像进行预处理。
# 
# 首先将图片大小调整为32*32，接着将图像转换成一维向量，最后再对一维向量进行归一化处理。

# In[18]:


def load_image(file):
        #打开图片
        im = Image.open(file)
        #将图片调整为跟训练数据一样的大小  32*32，                   设定ANTIALIAS，即抗锯齿.resize是缩放
        im = im.resize((256,256),Image.ANTIALIAS)
        #建立图片矩阵 类型为float32
        im = np.array(im).astype(np.float32)
        #矩阵转置 
        im = im.transpose((2, 0, 1))                               
        #将像素值从【0-255】转换为【0-1】
        im = im / 255.0
        #print(im)       
        im = np.expand_dims(im, axis=0)
        # 保持和之前输入image维度一致
        print('im_shape的维度：',im.shape)
        return im


# **(3)开始预测**
# 
# 通过fluid.io.load_inference_model，预测器会从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测。

# In[22]:


with fluid.scope_guard(inference_scope):
    #从指定目录中加载 推理model(inference model)
    [inference_program, # 预测用的program
     feed_target_names, # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。 
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。
                                                    infer_exe)     #infer_exe: 运行 inference model的 executor
    
    infer_path=['dog1.jpg','dog2.jpg','dog3.jpg','Cat1.jpg','Cat2.jpg','Cat3.jpg']
    label_list = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
        ]
    predict_result = []
    for image in infer_path:
        img = Image.open(image)
        plt.imshow(img)   
        plt.show()    
        
        img = load_image(image)
    
        results = infer_exe.run(inference_program,                 #运行预测程序
                                feed={feed_target_names[0]: img},  #喂入要预测的img
                                fetch_list=fetch_targets)          #得到推测结果
        predict_result.append(label_list[np.argmax(results[0])])
    print("infer results: \n", predict_result)

