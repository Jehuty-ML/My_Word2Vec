# -- encoding:utf-8 --

import os
import json
import shutil
import scipy.io as sio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

tf.flags.DEFINE_string("dictionary_path", "./data/dictionary.json", "词典数据所在的磁盘路径!!")
tf.flags.DEFINE_string("checkpoint_dir", "./running/model", "模型持久化文件路径!!")

FLAGS = tf.flags.FLAGS


class W2VPredictor(object):
    def __init__(self, dictionary_path, checkpoint_dir, unknown_word="<UNKNOWN>"):
        tf.logging.info("模型恢复....")
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # 一、网络结构的恢复&模型参数的恢复
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
                print(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path and os.path.exists("{}.meta".format(ckpt.model_checkpoint_path)):
                    tf.logging.info("恢复执行图....")
                    saver = tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))
                    tf.logging.info("恢复模型参数....")
                    saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
                    tf.logging.info("模型恢复完成!!!")
                else:
                    raise Exception("从文件夹中没有发现训练好的模型参数以及执行图文件!!!")

                # 二、从恢复的执行图中获取业务需要的tensor对象
                embedding_table_tensor = tf.get_default_graph().get_tensor_by_name("w2v/embedding/embedding_table:0")
                # print(embedding_table_tensor)

                # 三、当前业务决定的，因为Word2Vec只需要embedding表
                embedding_table = sess.run(embedding_table_tensor)

        self.UNKNOWN_WORD = unknown_word
        self.embedding_table = embedding_table
        tf.logging.info("加载词典....")
        self.words = json.load(open(dictionary_path, 'r', encoding="utf-8-sig"))
        assert len(self.words) == len(self.embedding_table), "单词数量和维度大小不一致!!!"
        # 将word和embedding合并到一起
        self.word_to_embedding = {}
        for idx in range(len(self.words)):
            self.word_to_embedding[self.words[idx]] = self.embedding_table[idx]

    def show_word_embedding(self, number_words=None):
        tf.logging.info("开始进行可视化构建!!!!")
        if number_words is None:
            embedding_table = self.embedding_table
        else:
            embedding_table = self.embedding_table[:number_words, :]
        tsne = TSNE()
        tsne_embedding_table = tsne.fit_transform(embedding_table)
        plt.subplots(figsize=(13, 6))
        for idx in range(np.shape(embedding_table)[0]):
            x, y = tsne_embedding_table[idx, :]
            plt.scatter(x, y, color="steelblue")  # 绘制这个点
            plt.annotate("{}_{}".format(idx, self.words[idx]), (x, y), alpha=0.7)
        plt.show()

    def fetch_word_embedding(self, word, return_unknown=True):
        """
        基于给定的单词，返回其对应的词向量
        :param word: 单词
        :param return_unknown: 如果单词word不存在，那么根据该参数决定是否返回unknown单词对应的向量
        :return: 返回单词向量
        :exception KeyError: 当单词不存在，并且return_unknown为False的时候，抛出KeyError异常
        """
        if word not in self.word_to_embedding:
            if return_unknown:
                word = self.UNKNOWN_WORD
            else:
                raise KeyError("单词[{}]不存在!!".format(word))
        return self.word_to_embedding[word]

    def save(self, save_dir, as_text=True, encoding="utf-8"):
        """
        将需要的数据持久化为其它语言可以读取的数据格式
        :param save_dir:  数据存储的文件夹路径，如果文件夹存在，那么会进行删除重新构建
        :param as_text: 存储的时候是存储为什么数据格式，当as_text为True的时候，存储为文本文件的格式；当as_text为False的时候，存储为matlab的数据格式
        :param encoding: 编码
        :return:
        """
        # 1.输出文件夹的判断
        if os.path.exists(save_dir):
            tf.logging.info("输出文件夹删除!!!")
            shutil.rmtree(save_dir)  # 级联删除，就是先删除文件夹中的文件以及子文件，再删除文件夹本身
        os.makedirs(save_dir)

        # 2. 数据输出操作
        if as_text:
            tf.logging.info("基于文本文件的方式进行数据的输出....")
            with open(os.path.join(save_dir, "vectors.txt"), "w", encoding=encoding) as writer:
                # 输出单词总数目以及单词向量维度大小
                writer.writelines("{}\t{}\n".format(*np.shape(self.embedding_table)))
                # 遍历所有单词输出
                for _word, _vector in self.word_to_embedding.items():
                    writer.writelines("{}\t{}\n".format(_word, " ".join(map(str, _vector))))
        else:
            tf.logging.info("基于Matlab数据格式的方式进行数据的输出....")
            sio.savemat(
                os.path.join(save_dir, "vectors.mat"),
                mdict={
                    '__version__': "1.0.0",
                    'words': self.words,
                    'embeddings': self.embedding_table
                },
                appendmat=True,
                format='5'  # 给定Matlab的格式版本，可选:4 or 5
            )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    predictor = W2VPredictor(
        dictionary_path=FLAGS.dictionary_path,
        checkpoint_dir=FLAGS.checkpoint_dir
    )
    # 可视化操作
    # predictor.show_word_embedding()

    # 获取单词对应的向量
    word = "侯亮平"
    vector = predictor.fetch_word_embedding(word)
    print("单词【{}】对应的向量为:\n{}".format(word, vector))
    # 保存
    predictor.save(save_dir="./deploy_model", as_text=False)
