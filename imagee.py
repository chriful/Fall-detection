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

flag = 0
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
#add_arg('batch_size', int, 32, "Minibatch size.")
#add_arg('dataset', str, 'mpii', "Dataset")
#add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
#add_arg('kp_dim', int, 16, "Class number.")
#add_arg('None', str, None, "Whether to resume None.")
#add_arg('True', bool, True, "Flip test")
#add_arg('True', bool, True, "Shift heatmap")


# yapf: enable
import lib.mpii_reader as reader
IMAGE_SIZE = [384, 384]
FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
print("aaa")
print(flag)
image = layers.data(name='image', shape=[3, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype='float32')
file_id = layers.data(name='file_id', shape=[1, ], dtype='int')
model = pose_resnet.ResNet(layers=50, kps_num=16, test_mode=True)
def print_immediately(s):
    print(s)
    sys.stdout.flush()
output = model.net(input=image, target=None, target_weight=None)
place = fluid.CUDAPlace(0) if True else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
fluid.io.load_persistables(exe, 'checkpoints/pose-resnet50-mpii-384x384')
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
    i=0
    cap = cv2.VideoCapture("/home/data3/fall/human_pose_estimation/006.mp4")
    while 1:
        time.sleep(0.1)
        ret,frame = cap.read()
        # cv2.imshow("cap",frame)
#        i = i + 1
#        print(i)
#        cv2.imwrite("/home/data3/fall/human_pose_estimation/test/004.jpg", frame)
        if flag is 0:
#        		
            cv2.imwrite("/home/data3/fall/human_pose_estimation/test/004.jpg", frame)


            # cv2.imwrite("E:/cameracap/cap.jpg",frame)
            # cv2.imwrite("./cap.jpg", frame)
            flag = 1

        # if cv2.waitKey(100) & 0xff == ord('q'):
        #     break
    cap.release()
    # cv2.destroyAllWindows()
def recognize():
    global flag
	    # Image and target
   
    
    while 1:        
        if flag is 1:
            begin_time = time.time()
            
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            
           


            test_reader = paddle.batch(reader.test(), batch_size=1)
            feeder = fluid.DataFeeder(place=place, feed_list=[image, file_id])
            test_exe = fluid.ParallelExecutor(
                use_cuda=True if True else False,
                main_program=fluid.default_main_program().clone(for_test=True),
                loss_name=None)

            fetch_list = [image.name, output.name]
            print("准备测试")
            

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
                save_predict_results(input_image, out_heatmaps, file_ids, fold_name='results')
            flag = 0
            end_time = time.time()
            run_time = end_time-begin_time
            print ('测试一张图片时间：',run_time)


if __name__ == '__main__':
#    args = parser.parse_args()
    
    check_cuda(True)
    t1 = threading.Thread(target=showImg)
    t2 = threading.Thread(target=recognize) 
    t1.start()
    t2.start()
    
