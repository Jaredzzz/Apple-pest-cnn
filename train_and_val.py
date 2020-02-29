# ======================================================================
#导入文件
import os
import numpy as np
import tensorflow as tf
import input_data
import math
import model
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


IMG_W = 100
IMG_H = 100
N_CLASSES = 4
BATCH_SIZE = 10
EVA_BATCH_SIZE = 1
learning_rate = 0.0001
MAX_STEP = 10000
train_data_dir = './tfrecords/apple_traindata_100_2600_2.tfrecords'
test_data_dir = './tfrecords/apple_testdata_100_470_2.tfrecords'
logs_train_dir = './model_100_dropout=0.25_ckpt_2'

logs_train = './train_logs'
def train():
    with tf.name_scope('input'):
        #read train
        tra_image_batch, tra_label_batch = input_data.read_TFRecord(data_dir=train_data_dir,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 in_classes=N_CLASSES)
        #read test
        val_image_batch, val_label_batch = input_data.read_TFRecord(data_dir=test_data_dir,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False,
                                                 in_classes=N_CLASSES)
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    logits = model.inference(x, BATCH_SIZE, N_CLASSES)
    loss = model.losses(logits, y)
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits, y)

    saver = tf.train.Saver(tf.global_variables())

    # 这个是log汇总记录
    summary_op = tf.summary.merge_all()

    # 产生一个会话
    sess = tf.Session()

    # 产生一个writer来写log文件
    tra_summary_writer = tf.summary.FileWriter(logs_train, sess.graph)

    # 所有节点初始化
    sess.run(tf.global_variables_initializer())
    # 队列监控
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        array_loss = []
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                            feed_dict={x: tra_images, y: tra_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.2f, accuracy: %.2f%%' % (step, tra_loss, tra_acc * 100))
                array_loss.append(tra_loss)
            # if step % 200 == 0 or (step + 1) == MAX_STEP:
            #    val_images, val_labels = sess.run([val_image_batch, val_label_batch])
            #    val_loss, val_acc = sess.run([loss, acc],
            #                                feed_dict={x: val_images, y: val_labels})
            #   print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100))

                # summary_str = sess.run(summary_op)
                # val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'apple_model_dropout=0.25_100_3.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        plt.plot(array_loss)
        plt.xlabel('The sampling point ')
        plt.ylabel('loss')
        plt.title("The variation of the loss")
        plt.grid(True)
        plt.show()

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def num_correct_prediction(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
        the number of correct predictions
    """
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct
def evaluate():
    with tf.Graph().as_default():
        logs_test = './test_logs'
        logs_test_dir = './model_100_dropout=0.25_ckpt_2'
        test_data_dir = './tfrecords/apple_testdata_100_470_2.tfrecords'
        n_test = 470


        # read test
        val_image_batch, val_label_batch = input_data.read_TFRecord(data_dir=test_data_dir,
                                                                    batch_size=EVA_BATCH_SIZE,
                                                                    shuffle=False,
                                                                    in_classes=N_CLASSES)

        logits = model.inference(val_image_batch, EVA_BATCH_SIZE, N_CLASSES)

        summary_op = tf.summary.merge_all()

        sess = tf.Session()
        val_writer = tf.summary.FileWriter(logs_test, sess.graph)
        correct = num_correct_prediction(logits, val_label_batch)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_test_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('\nEvaluating......')
                num_step = int(math.floor(n_test / EVA_BATCH_SIZE))
                num_sample = num_step * EVA_BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' % num_sample)
                print('Total correct predictions: %d' % total_correct)
                print('Average accuracy: %.2f%%' % (100 * total_correct / num_sample))
                summary_str = sess.run(summary_op)
                val_writer.add_summary(summary_str, step)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)



if __name__=="__main__":
    # evaluate()
    train()
