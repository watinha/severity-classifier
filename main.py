import sys,json,np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from feature_selection import FeatureSearch, USES, USESPlus
from nlp import TextFilter

if (len(sys.argv) < 2):
  print('Pass the dataset folder parameter...')
  sys.exit(1)


def load_json(path):
  f = open(path, 'r')
  bugs = list(filter(lambda x: len(x) > 0, f.read().split('\n')))
  f.close()
  bugs = [ json.loads(bug)['DESCRIPTION'].lower() for bug in bugs ]
  return bugs


severity0_path = './%s/class0.txt' % (sys.argv[1])
severity1_path = './%s/class1.txt' % (sys.argv[1])

bugs0 = load_json(severity0_path)
bugs1 = load_json(severity1_path)

f = TextFilter(language='english')
X = f.transform(bugs0 + bugs1)
y = np.zeros(len(bugs0)).tolist() + np.ones(len(bugs1)).tolist()

pipe = Pipeline([
  #('extractor', TfidfVectorizer(stop_words='english')),
  #('extractor with selection', FeatureSearch(strategy=USES())),
  ('extractor with selection', FeatureSearch(strategy=USESPlus())),
  ('classifier', DecisionTreeClassifier())
])

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
results = cross_validate(pipe, X, y, cv=skf, scoring=('f1', 'precision', 'recall', 'roc_auc'))
print(results)
print('F-Score: %f' % (results['test_f1'].mean()))
print('Precision: %f' % (results['test_precision'].mean()))
print('Recall: %f' % (results['test_recall'].mean()))
print('ROC-AUC: %f' % (results['test_roc_auc'].mean()))
