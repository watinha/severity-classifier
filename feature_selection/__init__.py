import random

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

class USES:

  def __init__(self, n=50, language='english', max_iteration=50,
               classifier=DecisionTreeClassifier(), extractor=TfidfVectorizer(),
               random_state=42):
    self._language = language
    self._score = []
    self._n = n
    self._max_iteration = max_iteration
    self._classifier = classifier
    self._extractor = extractor
    self._best_acc = 0
    self._best_features = []
    self._random_state = random_state


  def _word_count (self, X, y):
    severity0 = []
    severity1 = []
    for i in range(len(y)):
      if (y[i] == 1):
        severity1.append(X[i])
      else:
        severity0.append(X[i])

    vectorizer0 = CountVectorizer(stop_words=self._language)
    count0 = vectorizer0.fit_transform([' '.join(severity0)])
    vectorizer1 = CountVectorizer(stop_words=self._language)
    count1 = vectorizer1.fit_transform([' '.join(severity1)])
    all_words = set(list(vectorizer0.vocabulary_.keys()) + list(vectorizer1.vocabulary_.keys()))

    map0 = {}
    map1 = {}
    for word in all_words:
      try:
        map0[word] = count0.toarray()[0][vectorizer0.vocabulary_[word]]
      except:
        map0[word] = 0

    for word in all_words:
      try:
        map1[word] = count1.toarray()[0][vectorizer1.vocabulary_[word]]
      except:
        map1[word] = 0

    return (severity0, severity1, all_words, map0, map1)

  def _calculate_aff_score(self, severity0, severity1, all_words, map0, map1):
    for word in all_words:
      n_0 = len(severity0)
      n_1 = len(severity1)
      n_w = map0[word] + map1[word]
      n_0_w = map0[word]
      n_1_w = map1[word]

      aff_0_w = n_0_w / (n_0 + n_w - n_0_w)
      aff_1_w = n_1_w / (n_1 + n_w - n_1_w)

      self._score.append({ 'word': word, 'score': (aff_0_w - aff_1_w) })

    self._score = sorted(self._score, key=lambda x: x['score'])

  def _sample(self, features_subset, iteration):
    return random.sample(features_subset, self._n)

  def fit(self, X, y):
    (severity0, severity1, all_words, map0, map1) = self._word_count(X, y)
    self._calculate_aff_score(severity0, severity1, all_words, map0, map1)

    features_subset = list(map(lambda x: x['word'], self._score[:self._n] + self._score[-self._n:]))

    random.seed(self._random_state)
    for i in range(self._max_iteration):
      features = self._sample(features_subset, i)
      self._extractor.vocabulary = features
      X_new = self._extractor.fit_transform(X)
      self._classifier.fit(X_new, y)
      acc = self._classifier.score(X_new, y)
      if acc > self._best_acc:
        self._best_acc = acc
        self._best_features = features

  def transform(self, X):
    self._extractor.vocabulary = self._best_features
    return self._extractor.transform(X)

  def fit_transform(self, X, y):
    self.fit(X, y)
    return self.transform(X)
