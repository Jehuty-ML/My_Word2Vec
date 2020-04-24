import tensorflow as tf
from nets.w2vnet import CBOWNetWork
from utils.data_utils import Datamanager
import os


# parameters
# =====================================
# 模型训练数据参数
tf.flags.DEFINE_string("data_path", "./data/train.cbow.data", "训练数据所在的磁盘路径!!")
tf.flags.DEFINE_string("dictionary_path", "./data/dictionary.json", "词典数据所在的磁盘路径!!")

# =====================================
# 网络结构的参数
tf.flags.DEFINE_string("network_name", "w2v", "网络结构名称!!")
tf.flags.DEFINE_integer("embedding_size", 128, "Embedding的维度大小!!")

# =====================================
# Word2Vec的参数
tf.flags.DEFINE_string("structure", "cbow", "Word2Vec的结构!!cbow/skip-gram")
tf.flags.DEFINE_integer("window", 4, "窗口大小!!")
tf.flags.DEFINE_boolean("cbow_mean", True, "CBOW结构中，合并上下文数据的时候，是否计算均值!!")

# =====================================
# 训练参数
tf.flags.DEFINE_integer("max_epoch", 10, "最大迭代的Epoch的次数!!")
tf.flags.DEFINE_integer("batch_size", 1000, "批次大小!!")
tf.flags.DEFINE_integer("num_sampled", 100, "负采样的类别数目!!")
tf.flags.DEFINE_string("optimizer_name", "adam", "优化器名称!!")
tf.flags.DEFINE_float("learning_rate", 0.001, "学习率!!")
tf.flags.DEFINE_float("regularization", 0.00001, "L2 Loss惩罚项系数!!")

# ====================================
# 模型持久化参数
tf.flags.DEFINE_string("checkpoint_dir", "./running/model", "模型持久化文件路径!!")
tf.flags.DEFINE_integer("checkpoint_per_batch", 100, "给定模型持久化的间隔批次大小!!")

# ====================================
# 模型可视化参数
tf.flags.DEFINE_string("summary_dir", "./running/graph", "模型可视化数据存储路径!!")

FLAGS = tf.flags.FLAGS



def main(_):
    #模型参数校验
    if not os.path.exists(FLAGS.data_path):
        raise Exception('数据文件夹不存在，请检查参数data_path')
    if not os.path.exists(FLAGS.dictionary_path):
        raise Exception("词典数据文件夹不存在，请检查参数!!!")
    assert FLAGS.structure in ['cbow', 'skipgram'], "仅支持cbow和skipgram这两个Word2Vec结构,请检查参数!!"
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.summary_dir):
        os.makedirs(FLAGS.summary_dir)


   # 训练数据加载
    tf.logging.info('开始加载数据。')
    train_data_manager = Datamanager(
        data_path="./data/train.cbow.data",
        dictionary_path="./data/dictionary.json",
        window=4,
        structure='cbow',
        batch_size=8,
        encoding='utf-8-sig',
        shuffle=True
    )


    with tf.Graph().as_default():
        #一、网络构建
        if FLAGS.structure == 'cbow':
            tf.logging.info('开始构建CBOW模型结构！')
            model = CBOWNetWork(
                name=FLAGS.network_name,
                vocab_size=train_data_manager.word_size,
                embedding_size=FLAGS.embedding_size,
                is_mean=FLAGS.cbow_mean,
                window=FLAGS.window,
                num_sampled=FLAGS.num_sampled,
                regularization=FLAGS.regularization,
                optimizer_name=FLAGS.optimizer_name,
                learning_rate=FLAGS.learning_rate,
                checkpoint_dir=FLAGS.checkpoint_dir
            )
        else:
            tf.logging.info('开始构建skipgram模型结构！')
            model = CBOWNetWork(
                name=FLAGS.network_name,
                vocab_size=train_data_manager.vocab_size,
                embedding_size=FLAGS.embedding_size,
                window=FLAGS.window,
                num_sampled=FLAGS.num_sampled,
                regularization=FLAGS.regularization,
                optimizer_name=FLAGS.optimizer_name,
                learning_rate=FLAGS.learning_rate,
                checkpoint_dir=FLAGS.checkpoint_dir
            )


        tf.logging.info('开始构建模型前向网络')
        model.interface()

        tf.logging.info('开始构建损失函数')
        loss = model.losses()

        tf.logging.info('开始构建优化器及训练对象')
        _, train_op = model.optimizer(loss=loss)

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir=FLAGS.summary_dir,
                                       graph=tf.get_default_graph())

        #二、模型训练
        with tf.Session() as sess:
            #模型参数初始化

            tf.logging.info('恢复模型继续训练。。。。')
            model.restore(session=sess)

            #迭代训练
            for epoch in range(10):
                for batch_x, batch_y in train_data_manager:
                    _, _loss, _step, _summary = sess.run([train_op, loss, model.global_step, summary_op],
                                        feed_dict={model.input_x: batch_x,
                                                   model.target: batch_y})
                    print('Epoch:{}, Step:{}, Loss:{}'.format(epoch, _step, _loss))
                    writer.add_summary(summary=_summary, global_step=_step)

                    if _step % FLAGS.checkpoint_per_batch == 0:
                        model.save(session=sess)

            model.save(session=sess)
            writer.close()



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
