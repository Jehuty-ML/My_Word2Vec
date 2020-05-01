import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

tf.flags.DEFINE_string("dictionary_path", "./data/dictionary.json", "词典数据所在的磁盘路径!!")
tf.flags.DEFINE_string("checkpoint_dir", "./running/model", "模型持久化文件路径!!")

FLAGS = tf.flags.FLAGS

def main(_):
    dictionary_path =FLAGS.dictionary_path
    words = json.load(open(dictionary_path, 'r', encoding='utf-8-sig'))

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 一、网络结构的恢复&模型参数的恢复
            checkpoint_dir = FLAGS.checkpoint_dir
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
            print(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path and os._exists('{}.meta'.format(ckpt.model_checkpoint_path)):
                tf.logging.info('恢复执行图')
                saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
                tf.logging.info('恢复模型参数')
                saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
                tf.logging.info("模型恢复完成!!!")
            else:
                raise Exception("从文件夹中没有发现训练好的模型参数以及执行图文件!!!")

            # 二、从恢复的执行图中获取业务需要的tensor对象
            embedding_table_tensor = tf.get_default_graph().get_tensor_by_name("w2v/embedding/embedding_table:0")

            # 三、当前业务决定的，因为Word2Vec只需要embedding表
            embedding_table = sess.run(embedding_table_tensor)

            # 四、使用sklearn中的相关API进行可视化操作
            tf.logging.info("开始进行可视化构建!!!!")
            embedding_table = embedding_table[:1000, :]
            tsne = TSNE()
            tsne_embedding_table = tsne.fit_transform(embedding_table)
            plt.subplot(figsize=(13,6))
            for idx in range(np.shape(embedding_table)[0]):
                x, y = tsne_embedding_table[idx, :]
                plt.scatter(x, y, color="steelblue")  # 绘制这个点
                plt.annotate("{}_{}".format(idx, words[idx]), (x, y), alpha=0.7)
            plt.show()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()