import csv
import pickle
import re
import time

from decouple import config
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

twitter_modelfile = config('TWITTER_MODELFILE')
encoded_twitter_modelfile = config('TWITTER_ENCODED_MODELFILE')
encoded_twitter_modelfile_parsed = config('TWITTER_ENCODED_PARSED_MODELFILE')
google_modelfile = config('GOOGLE_MODELFILE')
wikipedia_modelfile = config('WIKIPEDIA_MODELFILE')
twitter_pca_modelfile = config('TWITTER_PCA_MODELFILE')

#truncate = 13
truncate = 30

base_stop_words = stopwords.words('portuguese')
stop_words = frozenset(base_stop_words)

vector_models = [
    'Twitter',
    'Encoded_Twitter',
    'PCA_Twitter',
    'Google',
    'CBOWGen',
    'SkipGramGen',
    'Pt-BR_Word2Vec',
    'Pt-BR_GloVe',
    'Pt-BR_FastText',
    'Wikipedia',
]

twitter_cbow_gen = {
    50: 'D:/models/Generated/W2V_cbow_50.txt',
    100: 'D:/models/Generated/W2V_cbow_100.txt',
    200: 'D:/models/Generated/W2V_cbow_200.txt',
    400: 'D:/models/Generated/W2V_cbow_400.txt',
}

twitter_skipgram_gen = {
    50: 'D:/models/Generated/W2V_skipgram_50.txt',
    100: 'D:/models/Generated/W2V_skipgram_100.txt',
    200: 'D:/models/Generated/W2V_skipgram_200.txt',
    400: 'D:/models/Generated/W2V_skipgram_400.txt',
}

glove_embeddings = {
    25: 'D:/models/glove.twitter.27B.25d.txt',
    50: 'D:/models/glove.twitter.27B.50d.txt',
    100: 'D:/models/glove.twitter.27B.100d.txt',
    200: 'D:/models/glove.twitter.27B.200d.txt',
    300: 'D:/models/glove.6B.300d.txt',
}

glove_gen_embeddings = {
    50: 'D:/models/Generated/GloVe_50.txt',
    100: 'D:/models/Generated/GloVe_100.txt',
    200: 'D:/models/Generated/GloVe_200.txt',
    400: 'D:/models/Generated/GloVe_400.txt',
}

glove_parsings = {
    25: 'D:/models/glove2vec.twitter.27B.25d.txt',
    50: 'D:/models/glove2vec.twitter.27B.50d.txt',
    100: 'D:/models/glove2vec.twitter.27B.100d.txt',
    200: 'D:/models/glove2vec.twitter.27B.200d.txt',
    300: 'D:/models/glove.6B.300d.parsed.txt',
}

glove_gen_parsings = {
    50: 'D:/models/Generated/GloVe2Vec_50.txt',
    100: 'D:/models/Generated/GloVe2Vec_100.txt',
    200: 'D:/models/Generated/GloVe2Vec_200.txt',
    400: 'D:/models/Generated/GloVe2Vec_400.txt',
}

pt_br_w2v = {
    50: 'D:/models/word2vec_s50.txt',
    100: 'D:/models/word2vec_s100.txt',
    300: 'D:/models/word2vec_s300.txt',
    600: 'D:/models/word2vec_s600.txt',
    1000: 'D:/models/word2vec_s1000.txt',
}

pt_br_w2v_pca = {
    (600, 300): 'D:/Models/word2vec_pca_normalize_600-300.txt',
    (1000, 300): 'D:/Models/word2vec_pca_normalize_1000-300.txt',
}

pt_br_glove = {
    50: 'D:/models/glove_s50.txt',
    100: 'D:/models/glove_s100.txt',
    300: 'D:/models/glove_s300.txt',
    600: 'D:/models/glove_s600.txt',
    1000: 'D:/models/glove_s1000.txt',
}

pt_br_glove_pca = {
    (600, 300): 'D:/Models/glove_pca_normalize_600-300.txt',
    (1000, 300): 'D:/Models/glove_pca_normalize_1000-300.txt',
}

pt_br_fasttext = {
    50: 'D:/models/fasttext_s50.txt',
    100: 'D:/models/fasttext_s100.txt',
    300: 'D:/models/fasttext_s300.txt',
    600: 'D:/models/fasttext_s600.txt',
    1000: 'D:/models/fasttext_s1000.txt',
}

pt_br_fast_pca = {
    (600, 300): 'D:/Models/fasttext_pca_normalize_600-300.txt',
    (1000, 300): 'D:/Models/fasttext_pca_normalize_1000-300.txt',
}

# Cleaning process to remove any punctuation, parentheses, question marks.
# This leaves only alphanumeric characters.
remove_special_chars = re.compile("[^A-Za-z0-9 ]+")


class Word2VecModel():

    def __init__(
        self, train_dataset, test_dataset, class_qtd, base_model,
        set_dimensions=None,
        reduced_dimenstions=None
    ):
        self.start_time = time.clock()
        self.classdataset = class_qtd
        self.word2vec_type = base_model

        self.train_tweets = self.parse_tweets(train_dataset)
        self.test_tweets = self.parse_tweets(test_dataset)

        if base_model == 'CBOWGen':
            if set_dimensions:
                self.word2Vec_model = KeyedVectors.load_word2vec_format(
                    twitter_cbow_gen[set_dimensions],
                    encoding='utf-8'
                )
                self.dimension = self.word2Vec_model.vector_size
                self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
            else:
                raise ValueError
        elif base_model == 'SkipGramGen':
            if set_dimensions:
                self.word2Vec_model = KeyedVectors.load_word2vec_format(
                    twitter_skipgram_gen[set_dimensions],
                    encoding='utf-8'
                )
                self.dimension = self.word2Vec_model.vector_size
                self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
            else:
                raise ValueError
        elif base_model == 'Pt-BR_Word2Vec':
            if set_dimensions:
                if reduced_dimenstions:
                    self.word2Vec_model = KeyedVectors.load_word2vec_format(
                        pt_br_w2v_pca[(set_dimensions, reduced_dimenstions)],
                        encoding='utf-8'
                    )
                else:
                    self.word2Vec_model = KeyedVectors.load_word2vec_format(
                        pt_br_w2v[set_dimensions],
                        encoding='utf-8'
                    )
                self.dimension = self.word2Vec_model.vector_size
                self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
            else:
                raise ValueError
        elif base_model == 'Pt-BR_GloVe':
            if set_dimensions:
                if reduced_dimenstions:
                    self.word2Vec_model = KeyedVectors.load_word2vec_format(
                        pt_br_glove_pca[(set_dimensions, reduced_dimenstions)],
                        encoding='utf-8'
                    )
                else:
                    self.word2Vec_model = KeyedVectors.load_word2vec_format(
                        pt_br_glove[set_dimensions],
                        encoding='utf-8'
                    )
                self.dimension = self.word2Vec_model.vector_size
                self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
            else:
                raise ValueError
        elif base_model == 'Pt-BR_FastText':
            if set_dimensions:
                if reduced_dimenstions:
                    self.word2Vec_model = KeyedVectors.load_word2vec_format(
                        pt_br_fast_pca[(set_dimensions, reduced_dimenstions)],
                        encoding='utf-8'
                    )
                else:
                    self.word2Vec_model = KeyedVectors.load_word2vec_format(
                        pt_br_fasttext[set_dimensions],
                        encoding='utf-8'
                    )
                self.dimension = self.word2Vec_model.vector_size
                self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
            else:
                raise ValueError
        elif base_model == 'Wikipedia':
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                wikipedia_modelfile,
                encoding='utf-8'
            )
            self.dimension = self.word2Vec_model.vector_size

            self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model == 'Twitter':
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                twitter_modelfile, binary=True, encoding='latin-1'
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model == 'Encoded_Twitter':
            _ = glove2word2vec(encoded_twitter_modelfile, encoded_twitter_modelfile_parsed)
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                encoded_twitter_modelfile_parsed, encoding='latin-1'
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model == 'PCA_Twitter':
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                twitter_pca_modelfile, encoding='utf-8'
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model == 'Twitter_200':
            tw_model = 'D:/Models/Generated/pca_embed_200.txt'
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                tw_model, encoding='utf-8'
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model in ['Google', 'random']:
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                google_modelfile, binary=True
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = 12  # 90 percentile value of number of words in a tweet based on Google
        print(f'The model has {self.dimension} dimensions')

    def clean(self, sentence):
        return re.sub(remove_special_chars, "", sentence.lower())

    def parse_tweets(self, filename):
        with open(filename, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            tweets = []
            for tweet in reader:
                tweet[2] = self.clean(tweet[2])
                tweets.append(tweet)
        return tweets

    def vectorize(self):
        train_labels = [int(tweet[0]) for tweet in self.train_tweets]
        test_labels = [int(tweet[0]) for tweet in self.test_tweets]

        if self.word2vec_type in vector_models:
            vectorizer = CountVectorizer(
                min_df=1, stop_words=stop_words,
                ngram_range=(1, 1),
                analyzer=u'word'
            )
            analyzer = vectorizer.build_analyzer()
            train_vectors = self.model_vectorize(
                tweet_base=self.train_tweets,
                analyzer=analyzer
            )
            test_vectors = self.model_vectorize(
                tweet_base=self.test_tweets,
                analyzer=analyzer
            )
        elif self.word2vec_type == 'random':
            train_vectors = self.random_vectorize(tweet_base=self.train_tweets)
            test_vectors = self.random_vectorize(tweet_base=self.test_tweets)

        print("{} word2vec matrix has been created as the input layer".format(
            self.word2vec_type
        ))

        vectorizing_time = time.clock() - self.start_time
        return {
            'test_id': [tweet[1] for tweet in self.test_tweets],
            'test_tweets': [tweet[2] for tweet in self.test_tweets],
            'train_vectors': train_vectors,
            'train_labels': train_labels,
            'test_vectors': test_vectors,
            'test_labels': test_labels, 
            'vectorizing_time': vectorizing_time,
        }

    def model_vectorize(self, tweet_base, analyzer):
        values = np.zeros(
            (len(tweet_base), self.tweet_length, self.dimension),
            dtype=np.float32
        )

        for i in range(len(tweet_base)):
            words_seq = analyzer(tweet_base[i][2])
            index = 0
            for word in words_seq:
                if index < self.tweet_length:
                    try:
                        values[i, index, :] = self.word2Vec_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break
        return values

    def random_vectorize(self, tweet_base):
        max_val = np.amax(self.word2Vec_model.syn0)
        min_val = np.amin(self.word2Vec_model.syn0)

        values = np.zeros(
            (len(tweet_base), self.tweet_length, self.dimension),
            dtype=np.float32
        )
        for i in range(len(tweet_base)):
            values[i, :, :] = min_val +\
                              (max_val - min_val) * np.random.rand(
                                                        self.tweet_length,
                                                        self.dimension
                                                    )
        return values

    def save(self, test_id, test_tweets, train_vectors, train_labels, test_vectors, test_labels):
        filename = 'Models/1_Word2Vec_' + self.word2vec_type +\
                    '_' + self.classdataset + '.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(
                [test_id, test_tweets, train_vectors, test_vectors, train_labels, test_labels],
                f
            )


class GloveModel():

    def __init__(
        self,
        train_dataset,
        test_dataset,
        class_qtd,
        dimensions,
        translate=False,
        generated=False
    ):
        # Necessário apenas se o modelo GloVe
        # não estiver "traduzido" para Word2Vec
        if translate:
            glove_embedding = glove_gen_embeddings[dimensions] if generated else glove_embeddings[dimensions]
            glove_parsing = glove_gen_parsings[dimensions] if generated else glove_parsings[dimensions]
            _ = glove2word2vec(glove_embedding, glove_parsing)

        self.classdataset = class_qtd
        glove_file = glove_gen_parsings[dimensions] if generated else glove_parsings[dimensions]
        self.gloVe_model = KeyedVectors.load_word2vec_format(
            glove_file,
            encoding='utf-8'
        )
        self.dimension = self.gloVe_model.vector_size

        self.train_tweets = self.parse_tweets(train_dataset)
        self.test_tweets = self.parse_tweets(test_dataset)
        self.tweet_length = truncate  # 90 percentile value of number of words in a tweet based on Twitter

    def clean(self, sentence):
        return re.sub(remove_special_chars, "", sentence.lower())

    def parse_tweets(self, filename):
        with open(filename, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            tweets = []
            for tweet in reader:
                tweet[2] = self.clean(tweet[2])
                tweets.append(tweet)
        return tweets

    def vectorize(self):
        train_labels = [int(tweet[0]) for tweet in self.train_tweets]
        test_labels = [int(tweet[0]) for tweet in self.test_tweets]

        vectorizer = CountVectorizer(
            min_df=1, stop_words=stop_words,
            ngram_range=(1, 1),
            analyzer=u'word'
        )

        analyzer = vectorizer.build_analyzer()
        train_vectors = self.model_vectorize(
            tweet_base=self.train_tweets,
            analyzer=analyzer
        )
        test_vectors = self.model_vectorize(
            tweet_base=self.test_tweets,
            analyzer=analyzer
        )

        print(f'{self.dimension} dimensions glove matrix has been created as'
              ' the input layer')

        return train_vectors, train_labels, test_vectors, test_labels

    def model_vectorize(self, tweet_base, analyzer):
        values = np.zeros(
            (len(tweet_base), self.tweet_length, self.dimension),
            dtype=np.float32
        )

        for i in range(len(tweet_base)):
            words_seq = analyzer(tweet_base[i][2])
            index = 0
            for word in words_seq:
                if index < self.tweet_length:
                    try:
                        values[i, index, :] = self.gloVe_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break
        return values

    def save(self, train_vectors, train_labels, test_vectors, test_labels):
        filename = f'Models/1_GloVe_{self.dimension}_{self.classdataset}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(
                [train_vectors, test_vectors, train_labels, test_labels],
                f
            )

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
