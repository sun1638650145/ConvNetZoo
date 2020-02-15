from PredictEngine import predict_with_ckpt
from PredictEngine import predict_with_pb
from PredictEngine import predict_with_sm

mode = input("INFO:请输入使用的模式(CheckPoint、FROZEN_MODEL或SAVED_MODEL):")
image = input("INFO:请输入要测试的图片路径:")
size = int(input("INFO:请输入神经网络的输入尺寸:"))
dir = input("INFO:请输入复现数据的路径:")
num_of_classes = int(input("INFO:请输入分类数:"))
predcit_list = []
for i in range(num_of_classes):
    predcit_list.append(input("INFO:请输入第%d类名称:" % (i+1)))
if mode == 'CheckPoint':
    model_name = input("INFO:请输入使用的模型:")
    ans = predict_with_ckpt(image=image,
                        net_size=size,
                        model_name=model_name,
                        num_of_classes=num_of_classes,
                        ckpt_dir=dir,
                        predict_list=predcit_list)
elif mode == 'FROZEN_MODEL':
    ans = predict_with_pb(image=image,
                          net_size=size,
                          pb_dir=dir,
                          predict_list=predcit_list)
elif mode == 'SAVED_MODEL':
    ans = predict_with_sm(image=image,
                          net_size=size,
                          sm_dir=dir,
                          predict_list=predcit_list)
print('预测的结果是:' + ans[1])