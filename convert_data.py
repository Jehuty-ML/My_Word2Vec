import tensorflow as tf
from utils.data_utils import convert_sentences_to_words
from utils.data_utils import build_dictionary
from utils.data_utils import convert_words_to_record
import os



tf.app.flags.DEFINE_string('opt', 'split',
                           '给定操作类型， split：分词')
tf.app.flags.DEFINE_string('split_input_file', None,
                           '给定分词操作的输入路径')
tf.app.flags.DEFINE_string('split_output_file', None,
                           '给定分词操作的输出路径')
tf.app.flags.DEFINE_string('dictionary_input_file', None,
                           '给定分词操作的输出路径')
tf.app.flags.DEFINE_string('dictionary_output_file', None,
                           '给定分词操作的输出路径')
tf.app.flags.DEFINE_string('record_input_file', None,
                           '给定训练数据构建操作的输入数据磁盘路径')
tf.app.flags.DEFINE_string('record_output_file', None,
                           '给定训练数据构建操作的输出数据磁盘路径')
tf.app.flags.DEFINE_integer('window', None,
                           '给定训练数据窗口大小')
tf.app.flags.DEFINE_string('structure', 'cbow',
                           '给定训练数据构建方式：cbow or skip-gram')
tf.app.flags.DEFINE_boolean("record_allow_padding", True,
                            "给定构建数据的时候是否进行填充操作!!")


FLAGS = tf.app.flags.FLAGS


def main(_):
    #1. 获取操作
    operation = FLAGS.opt.lower()

    #2. 判断类型
    if operation == 'split':
        tf.logging.info('进行原始数据分词操作')
        #a. 获取路径
        split_input_file = FLAGS.split_input_file
        split_output_file = FLAGS.split_output_file
        #b. 路径判断
        if split_input_file is None or not os.path.isfile(split_input_file):
            raise Exception('请检查参数split_input_file')
        if split_output_file is None:
            raise Exception('请检查参数split_output_file')
        #3. 进行分词调用
        convert_sentences_to_words(in_file=split_input_file, out_file=split_output_file)
        tf.logging.info('分词操作完成！')
    elif operation == 'dictionary':
        tf.logging.info('进行词典构建操作')
        #a. 获取路径
        dictionary_input_file = FLAGS.dictionary_input_file
        dictionary_output_file = FLAGS.dictionary_output_file
        #b. 路径判断
        if dictionary_input_file is None or not os.path.isfile(dictionary_input_file):
            raise Exception('请检查参数dictionary_input_file')
        if dictionary_output_file is None:
            raise Exception('请检查参数dictionary_output_file')
        #3. 进行分词调用
        build_dictionary(in_file=dictionary_input_file, out_file=dictionary_output_file)
        tf.logging.info('词典构建操作完成！')
    elif operation == 'record':
        tf.logging.info('进行训练数据构建。。。')
        # a. 获取路径
        record_input_file = FLAGS.record_input_file
        record_output_file = FLAGS.record_output_file
        window = FLAGS.window
        structure = FLAGS.structure
        # b. 路径判断
        if record_input_file is None or not os.path.isfile(record_input_file):
            raise Exception('请检查参数record_input_file')
        if record_output_file is None:
            raise Exception('请检查参数dictionary_output_file')
        # 3. 进行分词调用
        convert_words_to_record(in_file=record_input_file,
                                out_file=record_output_file,
                                window=window,
                                structure=structure.lower(),
                                allow_padding=FLAGS.record_allow_padding)
        tf.logging.info('训练数据构建完成！')
    else:
        tf.logging.WARN('参数异常，请检查参数：opt')


if __name__ == '__main__':
    #设置日志级别
    tf.logging.set_verbosity(tf.logging.INFO)
    #运行
    tf.app.run()


