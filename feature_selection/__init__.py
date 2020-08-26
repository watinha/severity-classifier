import random, math

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

class FeatureSearch:

  def __init__(self, n=50, language='english', max_iteration=100,
               classifier=DecisionTreeClassifier(), extractor=TfidfVectorizer(),
               random_state=42, strategy=None):
    self._language = language
    self._score = []
    self._n = n
    self._max_iteration = max_iteration
    self._classifier = classifier
    self._extractor = extractor
    self._best_acc = 0
    self._best_features = []
    self._random_state = random_state
    self._strategy = strategy


  def _word_count (self, X, y):
    severity0 = []
    severity1 = []
    for i in range(len(y)):
      if (y[i] == 1):
        severity1.append(X[i])
      else:
        severity0.append(X[i])

    vectorizer0 = CountVectorizer(stop_words=self._language, binary=True)
    count0 = vectorizer0.fit_transform(severity0).toarray()
    vectorizer1 = CountVectorizer(stop_words=self._language, binary=True)
    count1 = vectorizer1.fit_transform(severity1).toarray()
    all_words = set(list(vectorizer0.vocabulary_.keys()) + list(vectorizer1.vocabulary_.keys()))

    map0 = {}
    map1 = {}
    for word in all_words:
      try:
        bugs_with_word = list(map(lambda bug: bug[vectorizer0.vocabulary_[word]], count0))
        map0[word] = sum(bugs_with_word)
      except:
        map0[word] = 0

    for word in all_words:
      try:
        bugs_with_word = list(map(lambda bug: bug[vectorizer1.vocabulary_[word]], count1))
        map1[word] = sum(bugs_with_word)
      except:
        map1[word] = 0

    return (severity0, severity1, all_words, map0, map1)

  def fit(self, X, y):
    (severity0, severity1, all_words, map0, map1) = self._word_count(X, y)
    self._score = self._strategy._calculate_aff_score(severity0, severity1, all_words, map0, map1)

    features_subset = list(map(lambda x: x['word'], self._score[:self._n] + self._score[-self._n:]))

    random.seed(self._random_state)
    for i in range(1, self._max_iteration + 1):
      features = self._strategy._sample(features_subset, self._n, i)
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


class USES:

  def _calculate_aff_score(self, severity0, severity1, all_words, map0, map1):
    score = []

    for word in all_words:
      n_0 = len(severity0)
      n_1 = len(severity1)
      n_w = map0[word] + map1[word]
      n_0_w = map0[word]
      n_1_w = map1[word]

      aff_0_w = n_0_w / (n_0 + n_w - n_0_w)
      aff_1_w = n_1_w / (n_1 + n_w - n_1_w)

      score.append({ 'word': word, 'score': (aff_1_w - aff_0_w) })

    return sorted(score, key=lambda x: x['score'])


  def _sample(self, features_subset, n, iteration):
    return random.sample(features_subset, n)


class USESPlus:

  def _calculate_aff_score(self, severity0, severity1, all_words, map0, map1):
    score = []

    for word in all_words:
      n_w = map0[word] + map1[word]
      n_0_w = map0[word]
      n_1_w = map1[word]

      aff_0_w = n_0_w / n_w
      aff_1_w = n_1_w / n_w

      score.append({ 'word': word, 'score': abs(aff_1_w - aff_0_w) })

    return sorted(score, key=lambda x: x['score'])


  def _sample(self, features_subset, n, iteration):
    last = int(math.ceil(iteration / 2.0))
    start = iteration - last
    return features_subset[:start] + features_subset[-last:]


