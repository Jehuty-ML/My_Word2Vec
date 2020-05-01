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
        B: batch_size,
        T: window, 窗口大小
        V: vocab_size, 词表词汇数量
        E: embedding_size, 词向量维度
    c. 代码整理
    d. 运行命令：
        python train.py --data_path=./data/train.cbow.data --dictionary_path=./data/dictionary.json --network_name=w2v --embedding_size=128 --structure=cbow --window=4 --cbow_mean=True --max_epoch=10 --batch_size=1000 --num_sampled=100 --optimizer_name=adam --learning_rate=0.001 --regularization=0.00001 --checkpoint_dir=./running/model/cbow --checkpoint_per_batch=100 --summary_dir=./running/graph/cbow
        python train.py --data_path=./data/train.skipgram.data --dictionary_path=./data/dictionary.json --network_name=w2v --embedding_size=128 --structure=skipgram --window=4 --max_epoch=10 --batch_size=1000 --num_sampled=100 --optimizer_name=adam --learning_rate=0.001 --regularization=0.00001 --checkpoint_dir=./running/model/skipgram --checkpoint_per_batch=100 --summary_dir=./running/graph/skipgram


6. 理解一下word2vec负采样的执行过程（以CBOW为例）
    运行命令：
        python train.py --data_path=./data/train.cbow.data --dictionary_path=./data/dictionary.json --network_name=w2v --embedding_size=128 --structure=cbow
         --window=4 --cbow_mean=True --max_epoch=10 --batch_size=1000 --num_sampled=100 --optimizer_name=adam --learning_rate=0.001 --regularization=0.00001
          --checkpoint_dir=./running/model/cbow --checkpoint_per_batch=100 --summary_dir=./running/graph/cbow
    a. 普通全连接的执行过程（普通损失函数的构建过程）
    b. 负采样的执行过程

