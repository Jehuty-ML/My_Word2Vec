import tensorflow as tf
import numpy as np
import os

class CBOWNetWork:
    def __init__(self, name='W2V', vocab_size=1000, embedding_size=128, is_mean='true', window=4,
                 num_sampled=100, regularization=0.001, optimizer='adam', learning_rate=0.001,
                 checkpoint_dir="./running/model"
                 ):
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.is_mean = is_mean
        self.window = window
        self.num_sampled =num_sampled
        self.regularization = regularization
        self.optimizer = optimizer.lower()
        self.optimizer = optimizer
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
        #前向网络构建
        with tf.variable_scope(self.name):
            with tf.variable_scope('placeholder'):
                self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.window], name='input_x')
                self.target = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='target')
                self.training = tf.placeholder_with_default(True, shape=[], name='target')
                self.global_step = tf.train.get_or_create_global_step()

        #embedding
        with tf.variable_scope('embedding'):
            self.embedding_table = tf.get_variable('embedding_table',
                                              shape=[self.vocab_size, self.embedding_size],
                                              dtype=tf.float32)

            vectors = tf.nn.embedding_lookup(params=self.embedding_table, ids=self.input_x)

        #对于输入进行合并，得到最终特征属性
        with tf.variable_scope('merge'):
            #[B, T, E]  -->  [B, E]
            if self.is_mean:
                features = tf.reduce_mean(vectors, axis=1)
            else:
                features = tf.reduce_sum(vectors, axis=1)

        #属性给定
        self.features = tf.identity(features, name='features')




    def losses(self):
        #构建损失函数
        with tf.variable_scope('loss'):
            weight = tf.get_variable('weight',
                                     [self.vocab_size, self.embedding_size])
            bias = tf.get_variable('bias',
                                     [self.embedding_size])


        def train_loss():
            #仅训练进行负采样
            _loss = tf.nn.sampled_softmax_loss(weights=weight,  #shape:[V, E]
                         biases=bias,
                         labels=self.target,  #[B, num_true] num_true是每个样本存在预测标签数
                         inputs=self.features,  #
                         num_sampled=self.num_sampled,  #针对每个批次抽取负例数
                         num_classes=self.num_sampled,
                         num_true=1)
            _loss = tf.reduce_mean(_loss, name='train_loss')
            return _loss


        def eval_loss():

            logits = tf.nn.bias_add(tf.matmul(self.features, weight, transpose_b=True), bias=bias)

            labels = tf.reshape(self.traget, shape=[-1])

            _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=logits)

            _loss = tf.nn.reduce_mean(_loss, name='eval_loss')




        loss = tf.cond(pred=self.training,
                    true_fn=train_loss(),
                    false_fn=eval_loss())
        tf.summary.scalar('loss', loss)

        l2_loss = tf.nn.l2_loss(self.embedding_table) + tf.nn.l2_loss(weight) + tf.nn.l2_loss(bias)
        l2_loss = self.regularization * l2_loss
        tf.summary.scalar('l2_loss', l2_loss)

        #合并损失
        total_loss = loss + l2_loss
        tf.summary.scalar('total_loss', total_loss)
        return total_loss



    def optimizer(self, loss, *args):
        with tf.variable_scope('train'):
            if self.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                             beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon,)
            elif self.optimizer == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(
                    learning_rate=self.learning_rate,
                    rho=self.adadelta_rho,
                    epsilon=self.epsilon
                )
            elif self.optimizer == 'adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
            else:
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        #构建优化器

    def metrics(self, loss=None):
        #模型评估
        pass

    def restore(self, session):
        #模型参数恢复
        if self.saver is None:
            self.saver = tf.train.Saver()

        session.run(tf.global_variable_initilizer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info('Restore model weight from {}'.format(ckpt.model_checkpoint_path))
            self.saver.restore(session, save_path=ckpt.model_checkpoint_path)
            self.saver.recover_last_checkpoints(ckpt.all_model_checkpoint)


    def save(self, session):
        #模型持久化
        if self.saver is None:
            self.saver = tf.train.Saver()

        tf.logging.info('Restore the model weight to {}'.format(self.checkpoint_path))

        self.saver.save(session, save_path=self.checkpoint_path, global_step=self.global_step)



class SkipGramNetwork:
    pass