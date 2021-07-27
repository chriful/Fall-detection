import os
from PIL import Image
f = open(r'/home/spwux/Wch/datapre/sdtrain.txt',encoding = "utf-8")
a = list(f)
imge=[]
labellabel=[]
for line in a:
    line=line.strip('\n')
    img,label=line.split(' ')
    img = Image.open(img)
        # 统一图片大小
    img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
    print('这图片加载了')
        # 把图片转换成numpy值
    img = np.array(img).astype(np.float32)
        # 转换成CHW
    img = img.transpose((2, 0, 1))
        # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
#    return img, int(label)
   
#print(imge)
#print(labellabel)
#print(type(a[1]))
f.close()

#def train_reader(crop_size, resize_size):
#    dataset = ImageFolder('/home/data3/wch/sdtrain')
#    def reader():
#        num0=0
#        num1=0
#        for line in dataset.imgs:
#            img=line[0]
#            label=line[1]
#            if label==0:
#                num0+=1
#            else:
#                num1+=1
#            yield img, label ,crop_size, resize_size
#        print(num0,num1)
#    return paddle.reader.xmap_readers(train_mapper, reader,cpu_count(),209496)