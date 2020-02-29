import os
import tensorflow as tf
from PIL import Image

#图片路径
orig_picture = './apple_test'
#文件路径
filepath = './tfrecords/'
#存放图片个数
num_samples = 2670
#第几个图片
num = 0
#类别
classes = ['apple_healthy', 'apple_scab', 'apple_black_rot',  'apple_Cedar_apple_rust']
#tfrecords格式文件名
ftrecordfilename = ("apple_testdata_100_470_2.tfrecords")
writer = tf.python_io.TFRecordWriter(filepath+ftrecordfilename)
#类别和路径
for index, name in enumerate(classes):
    print(index)
    print(name)
    class_path = orig_picture + "/" + name + "/"
    for img_name in os.listdir(class_path):
        num = num+1
        # print('路径',class_path)
        # print('第几个图片：',num)
        # print('图片名：',img_name)

        img_path = class_path+img_name  # 每一个图片的地址
        img = Image.open(img_path, 'r')
        size = img.size
        print(size[1], size[0])
        print(size)
        # print(img.mode)
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(
             features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'img_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
            'img_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()

