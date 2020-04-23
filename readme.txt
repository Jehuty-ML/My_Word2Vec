1.文本数据分词
    运行命令：
        python convert_data.py --opt=split --split_input_file=./data/in_the_name_of_people.txt --split_output_file=./data/words_sentence.txt

2. 词典构建
    运行命令：
        python convert_data.py --opt=dictionary --dictionary_input_file=./data/words_sentence.txt
        --dictionary_output_file=./data/dictionary.json

3.训练数据转换
    运行命令：
        python convert_data.py --opt=record --record_input_file=./data/words_sentence.txt
        --record_output_file=./data/train.cbow.data --window=4 --structure=cbow --record_allow_padding=False

        python convert_data.py --opt=record --record_input_file=./data/words_sentence.txt
        --record_output_file=./data/train.skip_gram.data --window=4 --structure=skip_gram --record_allow_padding=True


4.数据加载
    详见:utils.data_utils.DataManager



5.训练
    a. 搭建大框架
    b. 参照算法原理实现