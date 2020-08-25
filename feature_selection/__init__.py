from sklearn.feature_extraction.text import CountVectorizer

class USES:

  def __init__(self, language='english'):
    self._language = language
    self._words = []
    self._map0 = {}
    self._map1 = {}

  def fit(self, X, y):
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

    for word in vectorizer0.vocabulary_:
        self._map0[word] = count0.toarray()[0][vectorizer0.vocabulary_[word]]

    for word in vectorizer1.vocabulary_:
        self._map1[word] = count1.toarray()[0][vectorizer1.vocabulary_[word]]
