import sys,json,np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from feature_selection import USES

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

X = bugs0 + bugs1
y = np.zeros(len(bugs0)).tolist() + np.ones(len(bugs1)).tolist()

pipe = Pipeline([
  ('extractor', TfidfVectorizer(stop_words='english')),
  ('classifier', DecisionTreeClassifier())
])

(USES(language='english')).fit(X, y)

#skf = StratifiedKFold(n_splits=10, random_state=42)
#results = cross_validate(pipe, X, y, cv=skf, scoring=('f1_macro', 'precision', 'recall', 'roc_auc'))
#print(results)
#print('F-Score: %f' % (results['test_f1_macro'].mean()))
#print('Precision: %f' % (results['test_precision'].mean()))
#print('Recall: %f' % (results['test_recall'].mean()))
#print('ROC-AUC: %f' % (results['test_roc_auc'].mean()))
