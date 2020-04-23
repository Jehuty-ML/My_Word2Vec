import tensorflow as tf
from nets.w2vnet import CBOWNetWork
from utils.data_utils import Datamanager


def main(_):
    with tf.Graph().as_default():
        #一、网络构建
        model = CBOWNetWork()

        tf.logging.info('开始构建模型前向网络')
        model.interface()

        tf.logging.info('开始构建损失函数')
        loss = model.losses()

        tf.logging.info('开始构建优化器及训练对象')
        optimizer, train_op = model.optimizer(loss=loss)


        #二、模型训练
        with tf.Session() as sess:
            #模型参数初始化
            if ckpt and ckpt.checkpoint:
                tf.logging.info('恢复模型继续训练。。。。')
                model.restore(model.model_checkpoint)
            else:
                tf.logging.info('模型初始化。。。。')
                sess.run(tf.global_variable_initializer())

            #训练数据加载
            tf.logging.info('开始加载数据。')
            train_data_manager = Datamanager()

            #迭代训练
            for epoch in range(10):
                for batch_x, batch_y in train_data_manager:
                    _, _loss = sess.run([train_op, loss],
                                        feed_dict={model.input_x: batch_x,
                                                   model.input_y: batch_y})



if __name__ == '__main__':
    tf.logging.set_verbosity('tf.logging.INFO')
    tf.app.run()
