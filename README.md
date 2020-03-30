# CatE

The source code used for Discriminative Topic Mining via Category-Name Guided Text Embedding, published in WWW 2020. The code structure (especially file reading and saving functions) is adapted from the [Word2Vec implementation](https://github.com/tmikolov/word2vec).

## Requirements

* GCC compiler (used to compile the source c file): See the [guide for installing GCC](https://gcc.gnu.org/wiki/InstallingGCC).

## Run the Code

We provide a shell script ``run.sh`` for compiling the source file and performing topic mining on an example corpus.

**Note: When preparing the training text corpus, make sure each line in the file is one document/paragraph.**

## Hyperparameters

Invoke the command without arguments for a list of hyperparameters and their meanings:
```
$ ./src/cate
Parameters:
        ##########   Input/Output:   ##########
        -train <file> (mandatory argument)
                Use text data from <file> to train the model
        -topic-name <file>
                Use <file> to provide the topic names/keywords; if not provided, unsupervised embeddings will be trained
        -res <file>
                Use <file> to save the topic mining results
        -k <int>
                Set the number of terms per topic in the output file; default is 10
        -word-emb <file>
                Use <file> to save the resulting word embeddings
        -topic-emb <file>
                Use <file> to save the resulting topic embeddings
        -spec <file>
                Use <file> to save the resulting word specificity value; if not provided, embeddings will be trained without specificity
        -load-emb <file>
                The pretrained embeddings will be read from <file>
        -binary <int>
                Save the resulting vectors in binary moded; default is 0 (off)
        -save-vocab <file>
                The vocabulary will be saved to <file>
        -read-vocab <file>
                The vocabulary will be read from <file>, not constructed from the training data

        ##########   Embedding Training:   ##########
        -size <int>
                Set dimension of text embeddings; default is 100
        -iter <int>
                Set the number of iterations to train on the corpus (performing topic mining); default is 5
        -pretrain <int>
                Set the number of iterations to pretrain on the corpus (without performing topic mining); default is 2
        -expand <int>
                Set the number of terms to be added per topic; default is 1
        -window <int>
                Set max skip length between words; default is 5
        -sample <float>
                Set threshold for occurrence of words. Those that appear with higher frequency in the training data
                will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        -negative <int>
                Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
        -threads <int>
                Use <int> threads (default 12)
        -min-count <int>
                This will discard words that appear less than <int> times; default is 5
        -alpha <float>
                Set the starting learning rate; default is 0.025
        -debug <int>
                Set the debug mode (default = 2 = more info during training)

See run.sh for an example to set the arguments
```

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{meng2020discriminative,
  title={Discriminative Topic Mining via Category-Name Guided Text Embedding},
  author={Meng, Yu and Huang, Jiaxin and Wang, Guangyuan and Wang, Zihan and Zhang, Chao and Zhang, Yu and Han, Jiawei},
  booktitle={The Web Conference},
  year={2020}
}
```
