import tensorflow as tf
import numpy as np
import os
from nets.base import NetWork



class SkipGramNetwork(NetWork):
    def __init__(self, name='W2V', vocab_size=19000, embedding_size=128, window=4,
                 num_sampled=100, regularization=0.001, optimizer_name='adam', learning_rate=0.001,
                 checkpoint_dir="./running/model"
                 ):
        NetWork.__init__(self, name=name, vocab_size=vocab_size, embedding_size=embedding_size, is_mean=is_mean,
                         window=window,
                         num_sampled=num_sampled, regularization=regularization, optimizer_name=optimizer_name,
                         learning_rate=learning_rate,
                         checkpoint_dir=checkpoint_dir)
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window = window
        self.num_sampled =num_sampled  #负采样个数
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

        self.input_x = None  # [B,1]
        self.target = None  # [B,T]
        self.training = None  # []
        self.global_step = None  # []
        self.features = None  # [B,E]
        self.embedding_table = None  # [V,E]
        self.saver = None  # 模型参数恢复、持久化等操作对象


    def interface(self):
        #前向网络构建
        with tf.variable_scope(self.name):
            with tf.variable_scope('placeholder'):
                self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_x')
                self.target = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='target')
                self.training = tf.placeholder_with_default(True, shape=[], name='target')
                self.global_step = tf.train.get_or_create_global_step()

        #embedding
        with tf.variable_scope('embedding'):
            self.embedding_table = tf.get_variable('embedding_table',
                                              shape=[self.vocab_size, self.embedding_size],
                                              dtype=tf.float32)
            # [B, 1] --> [B, 1, E]
            vectors = tf.nn.embedding_lookup(params=self.embedding_table, ids=self.input_x)

        #对于输入进行合并，得到最终特征属性
        with tf.variable_scope('merge'):
            #[B, 1, E]  -->  [B, E]
            features = tf.squeeze(vectors, axis=1)

        #属性给定
        self.features = tf.identity(features, name='features')




    def losses(self):
        #构建损失函数
        with tf.variable_scope('loss'):
            weight = tf.get_variable('weight',
                                     [self.vocab_size, self.embedding_size])
            bias = tf.get_variable('bias',
                                     [self.vocab_size])


        def train_loss():
            #仅训练进行负采样
            _loss = tf.nn.nce_loss(weights=weight,  #shape:[V, E]
                         biases=bias,   #shape:[V,]
                         labels=self.target,  #[B, num_true] num_true是每个样本存在预测标签数, 这里是1
                         inputs=self.features,  #
                         num_sampled=self.num_sampled,  #针对每个批次抽取负例数
                         num_classes=self.vocab_size,
                         num_true=1)
            _loss = tf.reduce_mean(_loss, name='train_loss')
            return _loss


        def eval_loss():
            #logits [B, V]
            logits = tf.nn.bias_add(tf.matmul(self.features, weight, transpose_b=True), bias=bias)
            #logits 对实际值进行哑编码操作（对应位置为1，存在多个位置为1)
            labels = tf.one_hot(self.target, depth=self.vocab_size)  # [B,T] --> [B,T,V]
            labels = tf.reduce_sum(labels, axis=1) # [B,T,V] --> [B,V]
            #注意这里是二分类，用sigmoid而不是softmax
            _loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                           logits=logits)

            _loss = tf.reduce_mean(_loss, name='eval_loss')
            return _loss




        loss = tf.cond(pred=self.training,
                    true_fn=train_loss,
                    false_fn=eval_loss)
        tf.summary.scalar('loss', loss)

        l2_loss = tf.nn.l2_loss(self.embedding_table) + tf.nn.l2_loss(weight) + tf.nn.l2_loss(bias)
        l2_loss = self.regularization * l2_loss
        tf.summary.scalar('l2_loss', l2_loss)

        #合并损失
        total_loss = loss + l2_loss
        tf.summary.scalar('total_loss', total_loss)
        return total_loss

