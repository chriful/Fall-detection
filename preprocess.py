import paddle.fluid as fluid
from PIL import Image
import numpy as np

# 保存预测模型路径
save_path = 'C:\\Users\\LiuXing\\Desktop\\Fall-detection\\Fall-detection\\fall.inference.modelx'
# 从模型中获取预测程序、输入数据名称列表、分类器
# 预处理图片
scope = fluid.Scope()
with fluid.scope_guard(scope):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    [infer_program, feeded_var_names, target_var] = (
        fluid.io.load_inference_model(dirname=save_path, executor=exe))
def load_image(file):
    im = Image.open(file)
    im = im.resize((256, 256), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
# PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)。
# PaddlePaddle要求数据顺序为CHW，所以需要转换顺序。
    im = im.transpose((2, 0, 1))
# CIFAR训练图片通道顺序为B(蓝),G(绿),R(红),
# 而PIL打开图片默认通道顺序为RGB,因为需要交换通道。
    im = im[(2, 1, 0), :, :]  # BGR
    im = im / 255.0
    im = np.expand_dims(im, axis=0)
    return im
def pre(path):
    img = load_image(path)
    with fluid.scope_guard(scope):
            result = exe.run(program=infer_program,
                             feed={feeded_var_names[0]: img},
                             fetch_list=target_var)
    print('results', result)
    label_list = [
            "down", "mid","stand"
    ]
    print("infer results: %s" % label_list[np.argmax(result[0])])
    return np.argmax(result[0])
pre('C:\\Users\\LiuXing\\Desktop\\Fall-detection\\Fall-detection\\image\\7.jpg')