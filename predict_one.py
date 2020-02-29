# ========================导入文件==============================================
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning
# ======================================================================
# 获取一张图片
def get_one_image(train_dir):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train_dir)
    ind = np.random.randint(0, n)
    img_dir = train_dir[ind]  # 随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()
    imag = img.resize([100, 100])  # 由于图片在预处理阶段已经resize，因此该命令可略
    image = np.array(imag)
    return image

# ======================================================================
# 测试图片
def evaluate_one_image(image_array):

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 4

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)   # 图片标准化函数，加速神经网络的训练
        image = tf.reshape(image, [1, 100, 100, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[100, 100, 3])

        logs_train_dir = './model_100_dropout=0.5_ckpt_2'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            print('预测的标签为：')
            print(max_index)
            print('预测的结果为：')
            print(prediction)


            if max_index == 0:
                print('This is a healthy apple tree with possibility %.6f.' % prediction[:, 0])
            elif max_index == 1:
                print('This is a sick apple tree.The disease is Apple scab with possibility %.6f.' % prediction[:, 1])
            elif max_index == 2:
                print('This is a sick apple tree.The disease is Apple black rot with possibility %.6f.' % prediction[:, 2])
            else:
                print('This is a sick apple tree.The disease is Cedar-apple rust with possibility %.6f.' % prediction[:, 3])


# ======================================================================
def pre_one_img(picname):

   img = Image.open(pre_image_dir + picname)
   img = img.convert('RGB')
   plt.imshow(img)
   plt.show()
   img = img.resize((100, 100))  # 设置需要转换的图片大小
   plt.imshow(img)
   plt.show()
   image_arr = np.array(img)
   return image_arr



if __name__ == '__main__':
    # 预测图片存放位置
    pre_image_dir = './prediction/'
    image_name = '2_3.jpg'
    img = pre_one_img(image_name)
    evaluate_one_image(img)
