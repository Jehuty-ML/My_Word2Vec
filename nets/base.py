import tensorflow as tf
import numpy as np
import os

class NetWork:
    def __init__(self, name='W2V', vocab_size=19000, embedding_size=128, is_mean='true', window=4,
                 num_sampled=100, regularization=0.001, optimizer_name='adam', learning_rate=0.001,
                 checkpoint_dir="./running/model"
                 ):
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.is_mean = is_mean
        self.window = window
        self.num_sampled =num_sampled
        self.regularization = regularization
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.beta1 = 0.9,  #adam优化器参数
        self.beta2 = 0.999,  #adam优化器参数
        self.epsilon = 1e-8,  #adam优化器参数
        self.adadelta_rho = 0.95 # Adadelta优化器参数
        self.checkpoint_dir = checkpoint_dir  # 模型持久化文件夹
        self.checkpoint_path = os.path.join(self.checkpoint_dir, '{}.ckpt'.format(self.name.lower()))


        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.input_x = None  # [B,T]
        self.target = None  # [B,1]
        self.training = None  # []
        self.global_step = None  # []
        self.features = None  # [B,E]
        self.embedding_table = None  # [V,E]
        self.saver = None  # 模型参数恢复、持久化等操作对象

def interface(self):
    raise Exception('请在子类进行前向网络构建')

def loss(self):
    raise Exception('请在子类进行损失函数构建')


def optimizer(self, loss):
    with tf.variable_scope('train'):
        if self.optimizer_name == 'adam':
            opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=self.adam_beta1,
                beta2=self.adam_beta2,
                epsilon=self.epsilon
            )
        elif self.optimizer_name == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate,
                rho=self.adadelta_rho,
                epsilon=self.epsilon
            )
        elif self.optimizer_name == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        else:
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        train_opt = opt.minimize(loss, global_step=self.global_step)

    return opt, train_opt

    def metrics(self, loss=None):
        """
        模型评估值的构建
        :param loss:
        :return:
        """
        pass

    def restore(self, session):
        if self.saver is None:
            self.saver =tf.train.Saver()

        session.run(tf.global_variables_initializer())

        ckpt =tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info('restore the weight from {}'.format(ckpt.model_checkpoint_path))
            self.saver.restore(session, save_path=ckpt.model_checkpoint_path)
            self.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

    def save(self, session):
        # 0. 参数判断
        if self.saver is None:
            self.saver = tf.train.Saver()

        # 1. 保存操作
        tf.logging.info("Store the model weight to '{}'".format(self.checkpoint_path))
        self.saver.save(session, save_path=self.checkpoint_path, global_step=self.global_step)
