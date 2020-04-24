import jieba
from collections import Counter
from collections import defaultdict
import re
import os
import json
import numpy as np

re_han = re.compile(r'([\u4E00-\u9FD5]+)', re.U)
PAD = '<PAD>'  #用来填充数据
UNKNOWN = '<UNKNOWN>'



def split_words(sentence):
    '''
    对给定的文本进行分词处理
    :return: 
    '''
    #添加自定义词典


    #停用词去除


    #如果是英语，逐个字母返回
    for word in jieba.lcut(sentence):
        if re_han.match(sentence):
            yield word
        else:
            for ch in word:
                yield ch





def convert_sentences_to_words(in_file, out_file, encoding='utf-8-sig'):
    '''
    对输入的文件做分词处理
    :param in_file:
    :param out_file:
    :return:
    '''
    with open(in_file, 'r', encoding='utf-8-sig') as reader:
        with open(out_file, 'w', encoding='utf-8-sig') as writer:
            for sentence in reader:
                sentence = sentence.strip()
                #过滤空行
                if len(sentence) == 0:
                    continue
                words = split_words(sentence)
                result = ' '.join(words)
                writer.writelines('%s\n' % result)



def build_dictionary(in_file, out_file, encoding='utf-8-sig'):
    '''
    基于分词的数据，进行字典构建
    :param in_file:
    :param out_file:
    :return:
    '''
    words = defaultdict(int)
    with open(in_file, 'r', encoding=encoding) as reader:
        for sentence in reader:
            sentence = sentence.strip()

            for word in sentence.split(' '):
                if len(word) > 0:
                    words[word] += 1

        #排序
        words = sorted(words, reverse=True)

        words = [PAD, UNKNOWN] + words

        dir_name = os.path.dirname(out_file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        json.dump(words, open(out_file, 'w', encoding=encoding), indent=4, ensure_ascii=False)


def build_record(words, window=4, structure='cbow', allow_padding=True):
    '''
    基于给定的参数获取训练数据，并以数据组形式返回
    :param words:
    :param window:
    :param structure:
    :return:
    '''
    n_words = len(words)
    start, end =(0, n_words) if allow_padding else (window//2, n_words-window+window//2)
    for idx in range(start, end):
        #假定中心词为idx, 然后分别获取上下文单词
        center_word = words[idx]
        surrounding_words = words[max(idx - window // 2, 0): idx]  #获取上文单词
        surrounding_words = [PAD] * (window // 2 - len(surrounding_words)) + surrounding_words
        surrounding_words += words[idx+1: idx+window-len(surrounding_words)+1]
        surrounding_words += [PAD] * (window-len(surrounding_words))
        if structure == 'cbow':
            yield surrounding_words + [center_word]
        else:
            yield [center_word] + surrounding_words






def convert_words_to_record(in_file, out_file, encoding='utf-8-sig', window=4, structure='cbow', allow_padding=True):
    '''
    将原始数据进行转换，并输出到磁盘
    :param in_file:
    :param out_file:
    :param encoding:
    :param window:
    :param structure:
    :param allow_padding:构建数据的时候是否允许填充
    :return:
    '''
    with open(in_file, 'r', encoding=encoding) as reader:
        with open(out_file, 'w', encoding=encoding) as writer:
            for sentence in reader:
                #1. 前后空格去除+转换为单词列表
                words = sentence.strip().split(' ')
                #2. 数据过滤
                if len(words) == 0:
                    continue
                if not allow_padding and len(words) <= window:
                    continue

                #3. 基于生成的数据，输出
                for record in build_record(words, window, structure, allow_padding):
                    writer.writelines('%s\n' % ' '.join(record))


class Datamanager:
    def __init__(self,
                 data_path,
                 dictionary_path,
                 batch_size=16,
                 encoding='utf-8-sig',
                 structure='cbow',
                 window=4,
                 shuffle=True
                 ):

        self.structure = structure
        self.window = window
        self.batch_size = batch_size
        self.shuffle = shuffle

        #一、构建字典
        #1.从磁盘加载字典
        words = json.load(open(dictionary_path, 'r', encoding=encoding))
        #2. 单词到id的映射
        self.word_size = len(words) #总单词数
        self.word_to_id= dict(zip(words, range(self.word_size)))
        self.id_to_word = words  #list

        #二、数据加载
        X, Y=[], []
        unknown_id = self.word_to_id[UNKNOWN]
        with open(data_path, 'r', encoding=encoding) as reader:
            for line in reader:
                #将数据划分成单词
                sample_words = line.strip().split(' ')
                if len(sample_words) != window+1:
                    continue
                #转化为id

                sample_words_ids = [self.word_to_id.get(word, unknown_id) for word in sample_words]
                if self.structure == 'cbow':
                    #上下文预测中心词
                    x = sample_words_ids[:-1]
                    y = sample_words_ids[-1:]
                else:
                    x = sample_words_ids[0:1]
                    y = sample_words_ids[1:]

                X.append(x)
                Y.append(y)
            self.X = np.asarray(X)  # X, [total_sample,window] or [total_sample,1]
            self.Y = np.asarray(Y)  # Y, [total_sample,1] or [total_sample,window]
            self.total_samples = len(self.X)
            self.total_batch = int(np.ceil(self.total_samples/self.batch_size))  #往上取整

    def __iter__(self):
        #产生序列
        if self.shuffle:
            total_index = np.random.permutation(self.total_samples)
            #输入int或者ndarray都行，都是返回一个打乱的序列
        else:
            total_index = np.arange(self.total_samples)

        #按照批次获取数据
        for batch_index in range(self.total_batch):
            start = batch_index * self.batch_size
            end = start + self.batch_size
            index = total_index[start:end]

            batch_x = self.X[index]
            batch_y = self.Y[index]
            yield batch_x, batch_y

        #结束批次
        raise StopIteration


    def __len__(self):
        return self.total_batch

if __name__ == '__main__':
    #测试build_record
    # for word in build_record(['我','是','小明','来自','中国'], window=4, structure='cbow'):
    #     print(word)

    datamanager = Datamanager(
                data_path="../data/train.cbow.data",
                dictionary_path="../data/dictionary.json",
                window=4,
                structure='cbow',
                batch_size=8,
                encoding='utf-8-sig',
                shuffle=True
    )
    print(len(datamanager))

    count = 0
    for batch_x, batch_y in datamanager:
        print(batch_x,'\n', batch_y)
        count += 1
        if count > 3:
            break





