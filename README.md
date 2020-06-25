# CatE

The source code used for Discriminative Topic Mining via Category-Name Guided Text Embedding, published in WWW 2020. The code structure (especially file reading and saving functions) is adapted from the [Word2Vec implementation](https://github.com/tmikolov/word2vec).

## Requirements

* GCC compiler (used to compile the source c file): See the [guide for installing GCC](https://gcc.gnu.org/wiki/InstallingGCC).

## Example Datasets

We provide two example datasets, the [New York Times annotated corpus](datasets/nyt/) and the [Yelp dataset challenge](datasets/yelp/), which are used in the paper. We also provide a shell script ``run.sh`` for compiling the source code and performing topic mining on the two example datasets. You should be able to obtain similar results as reported in the paper.

## Preparing Your Datasets

### Corpus and Inputs

You will need to first create a directory under `datasets` (e.g., `datasets/your_dataset`) and put two files in it:

* A text file of the corpus, e.g., `datasets/your_dataset/text.txt`. **Note: When preparing the text corpus, make sure each line in the file is one document/paragraph.**
* A text file with the category names/keywords for each category, e.g., `datasets/your_dataset/topics.txt` where each line contains the seed words for one category. You can provide arbitrary number of seed words in each line (at least 1 per category; if there are multiple seed words, separate them with whitespace characters). **Note: You need to ensure that every provided seed word appears in the vocabulary of the corpus.**

### Preprocessing

You can use any tool to preprocess the corpus (e.g. tokenization, lowercasing). If you do not have a specific idea, you can use our provided [preprocessing tool](preprocess). Simply add your corpus directory to [`auto_phrase.sh`](/preprocess/auto_phrase.sh#L16) and run it. The script assumes that the raw corpus is named `text.txt`, and will generate a phrase-segmented, lowercased corpus named `phrase_text.txt` under the same directory. If your corpus contains non-printable ASCII characters, you may use [this command](https://stackoverflow.com/a/27480803) to remove them.

### Pretrained Embedding (Optional)

We provide a 100-dimensional pretrained word2vec embedding `word2vec_100.zip`. You can also use other pretrained embeddings (use the `-load-emb` argument to specify the pretrained embedding file). Pretrained embedding is optional (omit the `-load-emb` argument if you do not use pretrained embedding), but generally will result in better embedding initialization and higher-quality topic mining results.

## Command Line Arguments

Invoke the command without arguments for a list of parameters and their meanings:
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
