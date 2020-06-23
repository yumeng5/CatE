// The code structure (especially file reading and saving functions) is adapted from the Word2Vec implementation
//          https://github.com/tmikolov/word2vec

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_WORDS_TOPIC 1000

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int corpus_max_size = 40000000;  // Maximum 40M documents in the corpus
const int topic_max_num = 1000;  // Maximum 1000 topics in the corpus

typedef float real;                    // Precision of float numbers

struct vocab_word {
    long long cn;
    int *point;
    char *word;
};

struct topic {
  // emb: category embedding
  // wt_score: cosine similarity of all words to a topic
  // grad: gradient for TopicEmb
  real *emb, *wt_score, *grad;
  // *cur_words: array of vocabulary indices of current retrieved representative words
  // init_size: how many seed words are given
  // cur_size: total number of current retrieved representative words
  int *cur_words, init_size, cur_size;
  // *category_name: name of the category
  char *category_name;
};

struct topic *topics;
char train_file[MAX_STRING], load_emb_file[MAX_STRING], res_file[MAX_STRING];
char output_file[MAX_STRING], topic_output[MAX_STRING], context_output[MAX_STRING], doc_output[MAX_STRING], spec_file[MAX_STRING], topic_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash, *docs;
long long *doc_sizes;
long long vocab_max_size = 1000, vocab_size = 0, corpus_size = 0, layer1_size = 100;
long long train_words = 0, word_count_total = 0, iter = 10, pretrain_iters = 2, file_size = 0;
int is_pretrain = 1;
real alpha = 0.025, starting_alpha, global_lambda = 1.5, spec_lambda = 0.5, sample = 1e-3;
real *syn0, *syn1neg, *syn1doc, *expTable, *wt_score_ptr;
real *kappa;
clock_t start;

int with_global = 1;
int with_topic = 0;
int with_spec = 0;

int negative = 5;
const int table_size = 1e8;
int *word_table, *doc_table;

// topic variables
int num_topic = 0; // number of topics
int num_per_topic = 10;
int words_per_reg = 96;
int expand = 1; // how many words to add to each topic per iteration
int *rankings; // ranking for each topic
real topic_lambda = 20; // scaling for topic loss
int rank_ensemble = 1;

int topic_pivot_idx;
int similaritySearchSize;
int *sim_rankings, *spec_rankings;

int load_emb = 0;
int load_emb_with_v = 0;
int fix_seed = 1;

int SimCompare(const void *a, const void *b) { // large -> small
  return (wt_score_ptr[*(int *) a] < wt_score_ptr[*(int *) b]) - (wt_score_ptr[*(int *) a] > wt_score_ptr[*(int *) b]);
}

int SpecCompare(const void *a, const void *b) { // small -> large
  return (kappa[*(int *) a] > kappa[*(int *) b]) - (kappa[*(int *) a] < kappa[*(int *) b]);
}

int RankProduct(const void *a, const void *b) { // small -> large
  int rank_a, rank_b;
  rank_a = (spec_rankings[*(int *) a] > 0 && sim_rankings[*(int *) a] > 0) ? spec_rankings[*(int *) a] * sim_rankings[*(int *) a] : -1;
  rank_b = (spec_rankings[*(int *) b] > 0 && sim_rankings[*(int *) b] > 0) ? spec_rankings[*(int *) b] * sim_rankings[*(int *) b] : -1;
  if (rank_a > 0 && rank_b > 0)
    return (rank_a > rank_b) - (rank_a < rank_b);
  else
    return (rank_a < rank_b) - (rank_a > rank_b);
}

int RankEnsemble(const void *a, const void *b) { // small -> large
  int rank_a, rank_b;
  rank_a = (spec_rankings[*(int *) a] > 0 && sim_rankings[*(int *) a] > 0) ? 1 / spec_rankings[*(int *) a] + 10 / sim_rankings[*(int *) a] : -1;
  rank_b = (spec_rankings[*(int *) b] > 0 && sim_rankings[*(int *) b] > 0) ? 1 / spec_rankings[*(int *) b] + 10 / sim_rankings[*(int *) b] : -1;
  return (rank_a < rank_b) - (rank_a > rank_b);
}

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  word_table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    word_table[a] = i;
    if (a / (double) table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void InitDocTable() {
  int a, i;
  double doc_len_pow = 0;
  double d1, power = 0.75;
  doc_table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < corpus_size; a++) doc_len_pow += pow(docs[a], power);
  i = 0;
  d1 = pow(docs[i], power) / doc_len_pow;
  for (a = 0; a < table_size; a++) {
    doc_table[a] = i;
    if (a / (double) table_size > d1) {
      i++;
      d1 += pow(docs[i], power) / doc_len_pow;
    }
    if (i >= corpus_size) i = corpus_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *) "</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Locate line number of current file pointer
int FindLine(FILE *fin) {
  long long pos = ftell(fin);
  long long lo = 0, hi = corpus_size - 1;
  while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (doc_sizes[mid] > pos) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

int IntCompare(const void * a, const void * b) { 
  return *(int*)a - *(int*)b; 
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) { // assert all sortings will be the same (since c++ qsort is not stable..)
  if (((struct vocab_word *) b)->cn == ((struct vocab_word *) a)->cn) {
    return strcmp(((struct vocab_word *) b)->word, ((struct vocab_word *) a)->word);
  }
  return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("[ERROR] training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *) "</s>");

  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } 
    else if (i == 0) {
      vocab[i].cn++;
      doc_sizes[corpus_size] = ftell(fin);
      corpus_size++;
      if (corpus_size >= corpus_max_size) {
        printf("[ERROR] Number of documents in corpus larger than \"corpus_max_size\"! Set a larger \"corpus_max_size\" in Line 20 of cate.c!\n");
        exit(1);
      }
    }
    else {
      vocab[i].cn++;
      docs[corpus_size]++;
    }
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("[ERROR] Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void LoadEmb(char *emb_file, real *emb_ptr) {
  long long a, b, c;
  int *vocab_match_tmp = (int *) calloc(vocab_size + 1, sizeof(int));
  int pretrain_vocab_size = 0, vocab_size_tmp = 0, word_dim;
  char *current_word = (char *) calloc(MAX_STRING, sizeof(char));
  real *syn_tmp = NULL;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn_tmp, 128, (long long) layer1_size * sizeof(real));
  if (syn_tmp == NULL) {
    printf("[ERROR] Memory allocation failed\n");
    exit(1);
  }
  printf("Loading embedding from file %s\n", emb_file);
  if (access(emb_file, R_OK) == -1) {
    printf("[ERROR] File %s does not exist\n", emb_file);
    exit(1);
  }
  // read embedding file
  FILE *fp = fopen(emb_file, "r");
  fscanf(fp, "%d", &pretrain_vocab_size);
  fscanf(fp, "%d", &word_dim);
  if (layer1_size != word_dim) {
    printf("[ERROR] Embedding dimension incompatible with pretrained file!\n");
    exit(1);
  }
  vocab_size_tmp = 0;

  for (c = 0; c < pretrain_vocab_size; c++) {
    fscanf(fp, "%s", current_word);
    a = SearchVocab(current_word);
    if (a == -1) {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &syn_tmp[b]);
    }
    else {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &emb_ptr[a * layer1_size + b]);
      vocab_match_tmp[vocab_size_tmp] = a;
      vocab_size_tmp++;
    }
  }
  printf("In vocab: %d\n", vocab_size_tmp);
  qsort(&vocab_match_tmp[0], vocab_size_tmp, sizeof(int), IntCompare);
  vocab_match_tmp[vocab_size_tmp] = vocab_size;
  int i = 0;
  for (a = 0; a < vocab_size; a++) {
    if (a < vocab_match_tmp[i]) {
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        emb_ptr[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
      }
    }
    else if (i < vocab_size_tmp) {
      i++;
    }
  }

  fclose(fp);
  free(current_word);
  free(emb_file);
  free(vocab_match_tmp);
  free(syn_tmp);
}

void ReadTopic() {
  long long a, b;
  int flag, vocab_idx;
  char tmp_word[MAX_STRING];
  memset(tmp_word, '\0', sizeof(tmp_word));
  topics = (struct topic *)calloc(topic_max_num, sizeof(struct topic));
  FILE *fp = fopen(topic_file, "r");
  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  while ((read = getline(&line, &len, fp)) != -1) {
    flag = 0;
    line[read - 1] = 0;
    if (line[read - 2] == '\r') // windows line ending
      line[read - 2] = 0;
    read = strlen(line);
    while (read > 0 && (line[read - 1] == ' ' || line[read - 1] == '\n')) read--;
    if (read == 0) {
      printf("[ERROR] Empty String!\n");
      exit(1);
    }
    if (line[0] == ' ') {
      printf("[ERROR] String starting with space!\n");
      exit(1);
    }
    for (int i = 0; i + 1 < read; i++) {
      if (line[i] == ' ' && line[i + 1] == ' ') {
        printf("[ERROR] Consecutive spaces\n");
        exit(1);
      }
    }
    int st = 0;
    
    // initialize topic 
    topics[num_topic].cur_words = (int *)calloc(MAX_WORDS_TOPIC, sizeof(int));
    topics[num_topic].emb = (real *)calloc(layer1_size, sizeof(real));
    topics[num_topic].grad = (real *)calloc(layer1_size, sizeof(real));
    topics[num_topic].wt_score = (real *)calloc(vocab_size, sizeof(real));
    topics[num_topic].category_name = (char *)calloc(MAX_STRING, sizeof(char));
    topics[num_topic].init_size = 0;

    // read words in line
    for (int i = 0; i <= read; i++) {
      if (line[i] == ' ' || i == read) {
        strncpy(tmp_word, line + st, i - st);
        tmp_word[i - st] = 0;
        if ((vocab_idx = SearchVocab(tmp_word)) != -1) {
          topics[num_topic].cur_words[topics[num_topic].init_size++] = vocab_idx;
          if (flag == 0) {
            strcpy(topics[num_topic].category_name, tmp_word);
            flag = 1;
          }
        } else {
          printf("[ERROR] Topic name %s not found in vocabulary!\n", tmp_word);
          exit(1);
        }
        st = i + 1;
      }
    }
    topics[num_topic].cur_size = topics[num_topic].init_size;
    num_topic++;
  }
  fclose(fp);

  topics = (struct topic *)realloc(topics, num_topic * sizeof(struct topic));
  for (a = 0; a < num_topic; a++) {
    if (topics[a].init_size == 0) {
      printf("[ERROR] No word in topic!\n");
      exit(1);
    }
  }
  printf("Read %d topics\n", num_topic);
  for (a = 0; a < num_topic; a++) {
    for (b = 0; b < topics[a].init_size; b++) {
      printf("%s\t", vocab[topics[a].cur_words[b]].word);
    }
    printf("\n");
  }
  rankings = (int *)calloc(vocab_size, sizeof(int));
  for (a = 0; a < vocab_size; a++) rankings[a] = a;
  spec_rankings = (int *)calloc(vocab_size, sizeof(int));
  for (a = 0; a < vocab_size; a++) spec_rankings[a] = a;
  sim_rankings = (int *)calloc(vocab_size, sizeof(int));
  for (a = 0; a < vocab_size; a++) sim_rankings[a] = a;
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {
    printf("[ERROR] Memory allocation failed\n");
    exit(1);
  }
  a = posix_memalign((void **) &kappa, 128, (long long) vocab_size * sizeof(real));
  if (kappa == NULL) {
    printf("[ERROR] Memory allocation failed (kappa)\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) kappa[a] = 1.0;
  a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(real));
  if (syn1neg == NULL) {
    printf("[ERROR] Memory allocation failed (syn1neg)\n");
    exit(1);
  }
  a = posix_memalign((void **) &syn1doc, 128, (long long) corpus_size * layer1_size * sizeof(real));
  if (syn1doc == NULL) {
    printf("[ERROR] Memory allocation failed (syn1doc)\n");
    exit(1);
  }
  
  for (a = 0; a < corpus_size; a++)
    for (b = 0; b < layer1_size; b++)
      syn1doc[a * layer1_size + b] = 0;
  
  if (load_emb_file[0] != 0) {
    char *center_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    strcpy(center_emb_file, load_emb_file);
    LoadEmb(center_emb_file, syn0);
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
  } else {
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
      }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
  }
  // Read topic names
  if (with_topic) {
    ReadTopic();
  }
}

void TopicEmb() {
  long long i, j, k, word, c;
  real f, sum_exp;
  real *word_grad = (real *)calloc(layer1_size, sizeof(real));
  real *exp_list = (real *)calloc(num_topic, sizeof(real));

  // zero out gradient
  for (i = 0; i < num_topic; i++) 
    for (c = 0; c < layer1_size; c++) 
      topics[i].grad[c] = 0;
  for (i = 0; i < num_topic; i++)
    for (j = 0; j < topics[i].cur_size; j++) {
      word = topics[i].cur_words[j];
      sum_exp = 0;
      for (k = 0; k < num_topic; k++) {
        f = 0;
        for (c = 0; c < layer1_size; c++) f += syn0[c + word*layer1_size] * topics[k].emb[c];
        exp_list[k] = exp(f);
        sum_exp += exp(f);
      }
      for (c = 0; c < layer1_size; c++) word_grad[c] = 0;
      for (k = 0; k < num_topic; k++) {
        f = alpha * topic_lambda / (topics[i].cur_size * num_topic) * (exp_list[k] / sum_exp - (i == k ? 1 : 0));
        for (c = 0; c < layer1_size; c++) 
          word_grad[c] -= f * topics[k].emb[c];
        for (c = 0; c < layer1_size; c++)
          topics[k].grad[c] -= f * syn0[c + word*layer1_size];
      }
      for (c = 0; c < layer1_size; c++)
        syn0[c + word*layer1_size] += word_grad[c];
    }
  // update topic embeddings
  for (i = 0; i < num_topic; i++)
    for (c = 0; c < layer1_size; c++)
      topics[i].emb[c] += topics[i].grad[c];

  free(word_grad);
  free(exp_list);
}

void *TrainEmb(void *id) {
  long long a, b, d, doc, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = is_pretrain ? pretrain_iters : 1;
  unsigned long long next_random = (long long) id;
  int word_counter = 0;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);

  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_total += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_total / (real) (iter * train_words + 1) * 100,
               word_count_total / ((real) (now - start + 1) / (real) CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_total / (real) (iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      doc = FindLine(fi);
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) /
                     vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real) 65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_total += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
      continue;
    }

    // update topic-related embeddings
    if (!is_pretrain) {
      word_counter += 1;
      if (word_counter == words_per_reg) word_counter = 0;
      if (word_counter == 0) {
        TopicEmb();
      }
    }

    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    // b = next_random % window;
    b = 0;

    for (a = b; a < window * 2 + 1 - b; a++)
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

        // NEGATIVE SAMPLING
        real kappa_update = 0.0;
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            target = word_table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          real tmp_kappa_update = f;
          f *= kappa[last_word];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * kappa[last_word] * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * kappa[last_word] * syn0[c + l1];
          if (with_spec) {
            tmp_kappa_update *= kappa[last_word];
            if (tmp_kappa_update > MAX_EXP) g = (label - 1) * alpha;
            else if (tmp_kappa_update < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int) ((tmp_kappa_update + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            kappa_update += g * tmp_kappa_update / kappa[last_word];
          }
        }
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        if (with_spec) {
          kappa[last_word] += spec_lambda * kappa_update;
          if (kappa[last_word] < 0) kappa[last_word] = 0;
        }
      }

    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    real kappa_update = 0;
    for (d = 0; d < negative + 1; d++) {
      if (d == 0) {
        target = doc;
        label = 1;
      } else {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        target = doc_table[(next_random >> 16) % table_size];
        if (target == doc) continue;
        label = 0;
      }
      l2 = target * layer1_size;
      f = 0;
      for (c = 0; c < layer1_size; c++) f += syn0[c + word * layer1_size] * syn1doc[c + l2];
      real tmp_kappa_update = f;
      f *= kappa[word];
      if (f > MAX_EXP) g = (label - 1) * alpha;
      else if (f < -MAX_EXP) g = (label - 0) * alpha;
      else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
      g *= global_lambda;
      kappa_update += g * tmp_kappa_update;
      for (c = 0; c < layer1_size; c++) neu1e[c] += g * kappa[word] * syn1doc[c + l2];
      for (c = 0; c < layer1_size; c++) syn1doc[c + l2] += g * kappa[word] * syn0[c + word * layer1_size];
    }
    for (c = 0; c < layer1_size; c++) syn0[c + word * layer1_size] += neu1e[c];
    if (with_spec) {
      kappa[word] += spec_lambda * kappa_update;
      if (kappa[word] < 0) kappa[word] = 0;
    }

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void ExpandTopic() {
  long a, b, c;
  int cur_sz, flag;
  real norm;
  printf("\n");
  for (a = 0; a < num_topic; a++) {
    for (b = 0; b < vocab_size; b++) {
      topics[a].wt_score[b] = 0;
      norm = 0;
      for (c = 0; c < layer1_size; c++) {
        topics[a].wt_score[b] += topics[a].emb[c] * syn0[b * layer1_size + c];
        norm += syn0[b * layer1_size + c] * syn0[b * layer1_size + c];
      }
      topics[a].wt_score[b] /= sqrt(norm);
    }
    wt_score_ptr = topics[a].wt_score;
    qsort(rankings, vocab_size, sizeof(int), SimCompare);

    if (fix_seed) {
      cur_sz = topics[a].init_size;
      for (b = 0; b < vocab_size; b++) {
        flag = 0;
        for (c = 0; c < topics[a].init_size; c++) {
          if (rankings[b] == topics[a].cur_words[c]) {
            flag = 1;
            break;
          }
        }
        if (flag == 0) topics[a].cur_words[cur_sz++] = rankings[b];
        if (cur_sz >= topics[a].cur_size + expand) break;
      }
    }
    else {
      for (b = 0; b < topics[a].cur_size + expand; b++) topics[a].cur_words[b] = rankings[b];
    }
    
    topics[a].cur_size += expand;
    printf("Category (%s): \t", topics[a].category_name);
    for (b = 0; b < topics[a].cur_size; b++) {
      printf("%s ", vocab[topics[a].cur_words[b]].word);
    }
    printf("\n");
  }
}

void TopicMine() {
  long a, b, c, iter_count;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

  // initialize topic embedding as averaged seed word embedding
  for (a = 0; a < num_topic; a++)
    for (b = 0; b < layer1_size; b++) {
      for (c = 0; c < topics[a].init_size; c++) {
        topics[a].emb[b] += syn0[topics[a].cur_words[c] * layer1_size + b];
      }
      topics[a].emb[b] /= topics[a].init_size;
    }
  is_pretrain = 0;
  for (iter_count = pretrain_iters; iter_count < iter; iter_count++) {
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainEmb, (void *) a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    ExpandTopic();
  }
}

void TrainModel() {
  long a, b;
  FILE *fo;
  pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  if (with_spec) printf("Training with specificity; Specificity values output to file %s\n", spec_file);
  if (with_topic) printf("Reading topics from file %s\n", topic_file);
  if (context_output[0] != 0) printf("Context embedding output to: %s\n", context_output);
  if (doc_output[0] != 0) printf("Document embedding output to: %s\n", doc_output);

  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (with_topic && topic_output[0] == 0) {
    printf("[ERROR] No topic embedding output file provided!\n");
    return;
  }
  if (topic_file[0] != 0) {
    if (access(topic_file, R_OK) == -1) {
      printf("[ERROR] Topic file not exist!\n");
      return;
    }
  }
  InitNet();
  InitUnigramTable();
  InitDocTable();
  start = clock();
  if (with_topic) {
    printf("Pre-training for %lld epochs, in total %lld + %lld = %lld epochs\n", pretrain_iters, pretrain_iters, iter,
           pretrain_iters + iter);
    iter += pretrain_iters;
  } else pretrain_iters = iter;
  is_pretrain = 1;

  // Unsupervised embedding training
  if (pretrain_iters > 0) {
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainEmb, (void *) a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  }

  // Topic mining
  if (with_topic) {
    TopicMine();
  }

  // Save the word vectors
  if (output_file[0] != 0) {
    fo = fopen(output_file, "wb");
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary)
        for (b = 0; b < layer1_size; b++) {
          fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
        }
      else
        for (b = 0; b < layer1_size; b++) {
          fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        }
      fprintf(fo, "\n");
    }
    fclose(fo);
  }
  
  // Save the context vectors
  if (context_output[0] != 0) {
    fo = fopen(context_output, "wb");
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      for (b = 0; b < layer1_size; b++) {
        fprintf(fo, "%lf ", syn1neg[a * layer1_size + b]);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);
  }

  // Save the document vectors
  if (doc_output[0] != 0) {
    fo = fopen(doc_output, "wb");
    fprintf(fo, "%lld %lld\n", corpus_size, layer1_size);
    for (a = 0; a < corpus_size; a++) {
      fprintf(fo, "%ld ", a);
      for (b = 0; b < layer1_size; b++) {
        fprintf(fo, "%lf ", syn1doc[a * layer1_size + b]);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);
  }

  // Save the specificity values
  if (with_spec) {
    fo = fopen(spec_file, "wb");
    fprintf(fo, "%lld\n", vocab_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      fprintf(fo, "%lf\n", kappa[a]);
    }
    fclose(fo);
  }

  // Save the topic embeddings
  if (with_topic && topic_output[0] != 0) {
    fo = fopen(topic_output, "wb");
    fprintf(fo, "%d\n", num_topic);
    for (a = 0; a < num_topic; a++) {
      fprintf(fo, "%s ", topics[a].category_name);
      for (b = 0; b < layer1_size; b++) {
        fprintf(fo, "%lf ", topics[a].emb[b]);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);
  }
}

void WriteResult() {
  long a, b, c;
  int cur_sz, flag;
  real norm;
  FILE *fo = fopen(res_file, "wb");
  printf("Topic mining results written to file %s\n", res_file);
  for (a = 0; a < num_topic; a++) {
    for (b = 0; b < vocab_size; b++) {
      topics[a].wt_score[b] = 0;
      norm = 0;
      for (c = 0; c < layer1_size; c++) {
        topics[a].wt_score[b] += topics[a].emb[c] * syn0[b * layer1_size + c];
        norm += syn0[b * layer1_size + c] * syn0[b * layer1_size + c];
      }
      topics[a].wt_score[b] /= sqrt(norm);
    }
    wt_score_ptr = topics[a].wt_score;

    qsort(rankings, vocab_size, sizeof(int), SimCompare);

    if (rank_ensemble && topics[a].init_size == 1) {
      for (b = 0; b < vocab_size; b++) sim_rankings[rankings[b]] = (b < num_per_topic + topics[a].init_size) ? b + 1 : -1;    
      qsort(rankings, vocab_size, sizeof(int), SpecCompare);
      for (b = 0; b < vocab_size; b++) spec_rankings[rankings[b]] = b + 1;
      qsort(rankings, vocab_size, sizeof(int), RankEnsemble);
    }

    fprintf(fo, "Category (%s):\n", topics[a].category_name);
    cur_sz = 0;
    for (b = 0; b < vocab_size; b++) {
      flag = 0;
      for (c = 0; c < topics[a].init_size; c++) {
        if (rankings[b] == topics[a].cur_words[c]) {
          flag = 1;
          break;
        }
      }
      if (flag == 0) {
        fprintf(fo, "%s ", vocab[rankings[b]].word);
        cur_sz++;
      }
      if (cur_sz >= num_per_topic) break;
    }
    fprintf(fo, "\n");
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("[ERROR] Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("Parameters:\n");

    printf("\t##########   Input/Output:   ##########\n");
    printf("\t-train <file> (mandatory argument)\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-topic-name <file>\n");
    printf("\t\tUse <file> to provide the topic names/keywords; if not provided, unsupervised embeddings will be trained\n");
    printf("\t-res <file>\n");
    printf("\t\tUse <file> to save the topic mining results\n");
    printf("\t-k <int>\n");
    printf("\t\tSet the number of terms per topic in the output file; default is 10\n");
    printf("\t-word-emb <file>\n");
    printf("\t\tUse <file> to save the resulting word embeddings\n");
    printf("\t-topic-emb <file>\n");
    printf("\t\tUse <file> to save the resulting topic embeddings\n");
    printf("\t-spec <file>\n");
    printf("\t\tUse <file> to save the resulting word specificity value; if not provided, embeddings will be trained without specificity\n");
    printf("\t-load-emb <file>\n");
    printf("\t\tThe pretrained embeddings will be read from <file>\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    

    printf("\n\t##########   Embedding Training:   ##########\n");
    printf("\t-size <int>\n");
    printf("\t\tSet dimension of text embeddings; default is 100\n");
    printf("\t-iter <int>\n");
    printf("\t\tSet the number of iterations to train on the corpus (performing topic mining); default is 5\n");
    printf("\t-pretrain <int>\n");
    printf("\t\tSet the number of iterations to pretrain on the corpus (without performing topic mining); default is 2\n");
    printf("\t-expand <int>\n");
    printf("\t\tSet the number of terms to be added per topic per iteration; default is 1\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    
    printf("\nSee run.sh for an example to set the arguments\n");
    return 0;
  }
  output_file[0] = 0;
  topic_file[0] = 0;
  context_output[0] = 0;
  topic_output[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  res_file[0] = 0;
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-k", argc, argv)) > 0) num_per_topic = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-expand", argc, argv)) > 0) expand = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-res", argc, argv)) > 0) strcpy(res_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-word-emb", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  // if ((i = ArgPos((char *) "-context", argc, argv)) > 0) strcpy(context_output, argv[i + 1]);
  // if ((i = ArgPos((char *) "-doc", argc, argv)) > 0) strcpy(doc_output, argv[i + 1]);
  if ((i = ArgPos((char *) "-topic-emb", argc, argv)) > 0) strcpy(topic_output, argv[i + 1]);
  if ((i = ArgPos((char *) "-spec", argc, argv)) > 0) {
    strcpy(spec_file, argv[i + 1]);
    with_spec = 1;
  }
  if ((i = ArgPos((char *) "-topic-name", argc, argv)) > 0) {
    strcpy(topic_file, argv[i + 1]);
    with_topic = 1;
  }
  if ((i = ArgPos((char *) "-pretrain", argc, argv)) > 0) pretrain_iters = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-load-emb", argc, argv)) > 0) strcpy(load_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);

  if (with_topic == 1 && res_file[0] == 0) {
    printf("[ERROR] No topic mining result file name provided! Use \"-res\" to indicate the file name to write results to!\n");
    return 1;
  }
  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  docs = (int *) calloc(corpus_max_size, sizeof(int));
  doc_sizes = (long long *) calloc(corpus_max_size, sizeof(long long));
  expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    expTable[i] = expTable[i] / (expTable[i] + 1);
  }
  TrainModel();
  if (with_topic) WriteResult();
  return 0;
}
