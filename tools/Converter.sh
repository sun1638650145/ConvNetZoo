#!/bin/zsh
#请使用tensorflowjs_converter v0.8.6和tensorflow r1.x生成的数据

input_format='tf_saved_model'
saved_model_dir='./saved_model'
read -p 'INFO:请输入转换后保存数据的路径:' export_dir

#实例
#tensorflowjs_converter \
#--input_format=tf_saved_model \
#--output_node_names='Softmax' \
#--saved_model_tags=serve \
#saved_model_complex \
#web_model

tensorflowjs_converter \
--input_format=$input_format \
--output_node_names='Softmax' \
--saved_model_tags=serve \
$saved_model_dir \
$export_dir

rm -rf ./saved_model