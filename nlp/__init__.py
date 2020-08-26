from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer

import re
import unicodedata

class NLP:

    def tokenizer(features):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = item.strip()
            item = item.lower()
            item = word_tokenize(item)
            features[i] = item

        return features

    def remove_numbers(features):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [NLP.number_remove(t) for t in item]
            item = list(filter(None, item))
            features[i] = item

        return features

    def number_remove(word):
        return re.sub('[0-9\\\]', '', word)

    def remove_small_words(features, min_length = 3):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [t for t in item if len(t) >= min_length]
            features[i] = item

        return features

    def remove_stop_words(features, language = 'english'):
        if (language == 'portuguese'):
            stopword = stopwords.words('portuguese')
        else:
            stopword = stopwords.words('english')

        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [t for t in item if t not in stopword]
            features[i] = item

        return features

    def  lemmatizer(features, language = 'english'):
        if (language == 'portuguese'):
            lemmatizer = RSLPStemmer().stem
        else:
            lemmatizer = WordNetLemmatizer().lemmatize

        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [lemmatizer(t) for t in item]
            item = list(filter(None, item))
            features[i] = item

        return features

    def remove_punctuation(features):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [NLP.punctuation_remove(t) for t in item]
            item = list(filter(None, item))
            features[i] = item

        return features


    def punctuation_remove(word):
        nfkd = unicodedata.normalize('NFKD', word)
        word = u"".join([c for c in nfkd if not unicodedata.combining(c)])
        return re.sub('[^a-zA-Z0-9 \\\]', '', word)


class TextFilter:

    def __init__ (self, language='english'):
        self._language = language

    def fit(self, X, y):
        pass

    def transform(self, X):
        features = NLP.tokenizer(X)
        features = NLP.remove_numbers(features)
        features = NLP.remove_small_words(features)
        features = NLP.remove_stop_words(features, self._language)
        features = NLP.lemmatizer(features, self._language)
        features = NLP.remove_punctuation(features)
        return [ ' '.join(row) for row in features ]

    def fit_transform(self, X, y):
        return self.transform(X)
