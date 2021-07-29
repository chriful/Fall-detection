# Copyright (c) 2018-present, Baidu, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################                                                        
"""Functions for inference."""

import sys
import argparse
import functools
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import time
from lib import pose_resnet
from utils.transforms import flip_back
from utils.utility import *

import cv2
import threading
import os
from lib import preprocess

flag = 0
ff=0
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
cos=0
start =0
#yapf: disable
#add_arg('batch_size', int, 32, "Minibatch size.")
#add_arg('dataset', str, 'mpii', "Dataset")
#add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
#add_arg('kp_dim', int, 16, "Class number.")
#add_arg('None', str, None, "Whether to resume None.")
#add_arg('True', bool, True, "Flip test")
#add_arg('True', bool, True, "Shift heatmap")


# yapf: enable


def print_immediately(s):
#    print(s)
    sys.stdout.flush()


# class DemoDataset(BaseCVDataset):
#    def __init__(self):
#        # 数据集存放位置
#
#        self.dataset_dir = ""
#        super(DemoDataset, self).__init__(
#            base_path=self.dataset_dir,
#            train_list_file="dataset/train_list.txt",
#            validate_list_file="dataset/validate_list.txt",
#            test_list_file="dataset/test_list.txt",
#            label_list_file="dataset/label_list.txt",
#        )


#def showImg():
#    global flag
#    cap = cv2.VideoCapture("/home/data3/fall/human_pose_estimation")
#    while 1:
#        ret, frame = cap.read()
#        # cv2.imshow("cap",frame)
#
#        if flag is 0:
#            cv2.imwrite("/home/data3/fall/human_pose_estimation/test/human36.jpg", frame)
#            flag = 1
#            print(flag)
#
##        if cv2.waitKey(100) & 0xff == ord('q'):
##            break
#    cap.release()
##    cv2.destroyAllWindows()
def showImg():
    global flag
    global ff
    global start
    i=0
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("D:\\fsdownload\\human_pose_estimation-11\\video\\55.mp4")
    while 1:
#        time.sleep(0.1)
        if start==1:
            ret,frame = cap.read()
            cv2.imshow('cap',frame)
            i = i + 1
#        cv2.imwrite("/home/data3/fall/human_pose_estimation/test/004.jpg", frame)
            if flag is 0:
#        		
                cv2.imwrite("C:\\Users\\LiuXing\\Desktop\\Fall-detection\\Fall-detection\\test\\004.jpg", frame)


            # cv2.imwrite("E:/cameracap/cap.jpg",frame)
            # cv2.imwrite("./cap.jpg", frame)
                flag = 1
            #if i==1000:
            #    break
            if cv2.waitKey(20) & 0xff == ord('q'):
            
                break
    ff=1
    cap.release()
    cv2.destroyAllWindows()
def recognize():
    global flag
    global ff
    global cos
    global start
    import lib.mpii_reader as reader
    IMAGE_SIZE = [384, 384]
    FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]

    # Image and target
    image = layers.data(name='image', shape=[3, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype='float32')
    file_id = layers.data(name='file_id', shape=[1, ], dtype='int')
    model = pose_resnet.ResNet(layers=50, kps_num=16, test_mode=True)
    output = model.net(input=image, target=None, target_weight=None)
    scope1 = fluid.Scope()
    with fluid.scope_guard(scope1):
        place = fluid.CUDAPlace(0) if True else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())         
        fluid.io.load_persistables(exe,'checkpoints/pose-resnet50-mpii-384x384')
    start=1
    fetch_list = [image.name, output.name]
    j=1
    start=1
    count=0
    monitor=0
    while ff==0:
        if flag is 1:
            begin_time = time.time()
            print("准备测试")
            test_reader = paddle.batch(reader.test(), batch_size=1)
            feeder = fluid.DataFeeder(place=place, feed_list=[image, file_id])
            with fluid.scope_guard(scope1):
                test_exe = fluid.ParallelExecutor(
                    use_cuda=True if True else False,
                    main_program=fluid.default_main_program().clone(for_test=True),
                    loss_name=None)


            for batch_id, data in enumerate(test_reader()):
                print_immediately("Processing batch #%d" % batch_id)
                num_images = len(data)

                file_ids = []
                for i in range(num_images):
                    file_ids.append(data[i][1])

                input_image, out_heatmaps = test_exe.run(fetch_list=fetch_list,
                                                         feed=feeder.feed(data))
                data_fliped = []
                for i in range(num_images):
                	data_fliped.append((data[i][0][:, :, ::-1], data[i][1]))
                _, output_flipped = test_exe.run(fetch_list=fetch_list,
                                                     feed=feeder.feed(data_fliped))

                output_flipped = flip_back(output_flipped, FLIP_PAIRS)

                output_flipped[:, :, :, 1:] = \
                		output_flipped.copy()[:, :, :, 0:-1]

                out_heatmaps = (out_heatmaps + output_flipped) * 0.5
                cod,ss,st = save_predict_results(input_image, out_heatmaps, file_ids, fold_name='results')
                if j==1:
                    offset = 0
                    cos=cod
                else:
                    offset = cos-cod
                    cos=cod
                
            
            j+=1
            img = cv2.imread("C:\\Users\\LiuXing\\Desktop\\Fall-detection\\Fall-detection\\results\\rendered_0000004.png")
            # if abs(offset) > 10:
                # offset = '%.2f'% abs(offset)
                # monitor=3
                # text = "offset:"+str(offset)
                # text1= "state:"+str('SAFE')
                # text2= "state:"+str('FALL')
                # text4= "Monitor"
                # state = demo.pre('D:\\fsdownload\\human_pose_estimation-11\\test\\004.jpg')
                # cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                # cv2.putText(img, text2, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0,255), 2)
                # cv2.putText(img, text4, (240, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            if monitor > 0:
                offset = '%.2f'% abs(offset)
                text = "offset:"+str(offset)
                text1= "state:"+str('SAFE')
                text2= "state:"+str('FALL')
                text3= "state:"+str('DANGER')
                text4= "Monitor"
                state = preprocess.pre('C:\\Users\\LiuXing\\Desktop\\Fall-detection\\Fall-detection\\test\\004.jpg')
                cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                cv2.putText(img, text4, (240, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                
                if state==0:
                    monitor=3
                    count=min(count+1,1)
                    if count==1:                    
                        cv2.putText(img, text2, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, text3, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 2)
                elif state==1: 
                    monitor=3
                    cv2.putText(img, text3, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 2)
                else:
                    cv2.putText(img, text1, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                    monitor=max(monitor-1,0)
                    #count=max(count-1,0)
                    #monitor=0
                    count=0
            
                
            elif abs(offset) < 3:
                offset = '%.2f'% abs(offset)
                text = "offset:"+str(offset)
                text1= "state:"+str('SAFE')
                text2= "state:"+str('FALL')
                ss='%.2f'% abs(ss)
                text5= "det:"+str(ss)
                st='%.2f'% abs(st)
                text6= "det:"+str(st)
                state = preprocess.pre('C:\\Users\\LiuXing\\Desktop\\Fall-detection\\Fall-detection\\test\\004.jpg')
                cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                cv2.putText(img, text1, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                cv2.putText(img, text5, (200, 150), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                cv2.putText(img, text6, (20, 150), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                
            else:
                offset = '%.2f'% abs(offset)
                text = "offset:"+str(offset)
                text1= "state:"+str('SAFE')
                text2= "state:"+str('FALL')
                text3= "state:"+str('DANGER')
                text4= "Monitor"
                
                state = preprocess.pre('C:\\Users\\LiuXing\\Desktop\\Fall-detection\\Fall-detection\\test\\004.jpg')
                cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                
                if state==0:
                    monitor=3
                    count=min(count+1,1)
                    if count==1:
                        cv2.putText(img, text2, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0, 255), 2)
                    else:
                        cv2.putText(img, text3, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255,0), 2)
                    cv2.putText(img, text4, (240, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                elif state==1:
                    monitor=3
                    cv2.putText(img, text3, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 2)
                    cv2.putText(img, text4, (240, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    #count=max(count-1,0)
                    #count=0
                else:
                    cv2.putText(img, text1, (200, 350), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
            cv2.imshow("inference", img)
            cv2.waitKey(1)
            end_time = time.time()
            run_time = end_time-begin_time
            print ('测试一张图片时间：',run_time)
            flag = 0


if __name__ == '__main__':
#    args = parser.parse_args()
    
    check_cuda(True)
    t1 = threading.Thread(target=showImg)
    t2 = threading.Thread(target=recognize) 
    t1.start()
    t2.start()
    
