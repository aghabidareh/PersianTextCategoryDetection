from pickle import load
import nltk

with open('stopwords.txt' , 'r' , encoding='utf-8') as stopwords_file:
    stopwords = stopwords_file.readlines()
stopwords = [line.replace('\n' , '') for line in stopwords]

nltk_stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(nltk_stopwords)

with open('./model.h5' , 'rb') as f:
    model = load(f)

with open('./normalizer.h5' , 'rb') as f:
    normalizer = load(f)

with open('./stem_lemmatize.h5' , 'rb') as f:
    stemmer_lemmatizer = load(f)

with open('./tokenizer.h5' , 'rb') as f:
    tokenizer = load(f)

with open('./vectorizer.h5' , 'rb') as f:
    vectorizer = load(f)

with open('./label_encoder.h5' , 'rb') as f:
    label_encoder = load(f)

