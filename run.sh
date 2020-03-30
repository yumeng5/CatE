# dataset directory
dataset=nyt

# text file name; one document per line
text_file=phrase_text.txt

# category name file
topic_file=locations.txt

topic=$(echo ${topic_file} | cut -d'.' -f 1)

# load pretrained word2vec embedding
pretrain_emb=word2vec_100.txt

if [ ! -f "$pretrain_emb" ] && [ -f "word2vec_100.zip" ]; then
    echo "Unzipping downloaded pretrained embedding"
    unzip word2vec_100.zip && rm word2vec_100.zip
fi

cd src
make cate
cd ..

./src/cate -train ./datasets/${dataset}/${text_file} -topic-name ./datasets/${dataset}/${topic_file} \
	-load-emb ${pretrain_emb} \
	-spec ./datasets/${dataset}/emb_${topic}_spec.txt \
	-res ./datasets/${dataset}/res_${topic}.txt -k 10 -expand 1 \
	-word-emb ./datasets/${dataset}/emb_${topic}_w.txt -topic-emb ./datasets/${dataset}/emb_${topic}_t.txt \
	-size 100 -window 5 -negative 5 -sample 1e-3 -min-count 5 \
	-threads 20 -binary 0 -iter 10 -pretrain 2
