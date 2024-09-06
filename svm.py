import time
import numpy as np
from sklearn import svm, metrics
from sklearn import naive_bayes 
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier
from scipy import sparse
import sys
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'lsvm', 'Model string.')  # 'lsvm', 'rbfsvm', 'mlp', 'nb'
flags.DEFINE_string('npz', 'dd', 'Input npz data.')
flags.DEFINE_string('resfile', 'rr', 'Output result file.')
flags.DEFINE_string('mode', 'eval', 'Evaluate mode or predict mode.') 
flags.DEFINE_integer('pca', 0, 'Reduce dimensions (0 means no pca)')
flags.DEFINE_integer('svd', 0, 'Reduce dimensions (0 means no svd)')
flags.DEFINE_integer('seed', 12, 'random seed.')

# Parse flags
flags.FLAGS([sys.argv[0]] + sys.argv[1:])

seed = int(FLAGS.seed)
np.random.seed(seed)

t_start = time.time()

# Load data
print("Loading data from:", str(FLAGS.npz))
data = np.load(str(FLAGS.npz), allow_pickle=True)

features = data['features'][()]
y_train = data['y_train'][()]
train_mask = data['train_mask'][()]
y_val = data['y_val'][()]
val_mask = data['val_mask'][()]
y_test = data['y_test'][()]
test_mask = data['test_mask'][()]

if int(FLAGS.pca) != 0:
    pca = PCA(n_components=int(FLAGS.pca), random_state=seed)
    features = features.toarray()
    pca.fit(features) 
    features = sparse.lil_matrix(pca.transform(features), dtype='float32')

if int(FLAGS.svd) > 0:
    svd = TruncatedSVD(n_components=FLAGS.svd, random_state=seed)
    svd.fit(features) 
    features = sparse.lil_matrix(svd.transform(features), dtype='float32')

X = features[np.where(train_mask == True)]
y = y_train[np.where(train_mask == True)]
X_test = features[np.where(test_mask == True)]
y_test = y_test[np.where(test_mask == True)]

y = np.argmax(y, axis=1)
y_test = np.argmax(y_test, axis=1)

if str(FLAGS.model) == 'lsvm':
    clf = svm.SVC(kernel='linear')
elif str(FLAGS.model) == 'rbfsvm':
    clf = svm.SVC(kernel='rbf')
elif str(FLAGS.model) == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, solver='adam', verbose=0, tol=1e-4, random_state=None, learning_rate='constant', learning_rate_init=.01)
elif str(FLAGS.model) == 'nb':
    clf = naive_bayes.GaussianNB()
    X = X.toarray()
    X_test = X_test.toarray()
elif str(FLAGS.model) == 'xgboost':
    clf = XGBClassifier(learning_rate=1,
        n_estimators=10,
        max_depth=3,
        min_child_weight=1,
        gamma=0.,
        subsample=0.8,
        colsample_btree=0.8,
        objective='multi:softmax',
        random_state=seed
        ) 
    X = X.toarray()
    X_test = X_test.toarray()
elif str(FLAGS.model) == 'randomforest':
    clf = RandomForestClassifier()
    X = X.toarray()
    X_test = X_test.toarray()

clf.fit(X, y)
y_pred = clf.predict(X_test)

total_time = time.time() - t_start

with open(str(FLAGS.resfile), 'a') as f:
    if str(FLAGS.mode) == 'eval':
        f.write(' '.join(sys.argv) + '\n')
        f.write("OA : {:0.5f} ".format(metrics.accuracy_score(y_test, y_pred)))
        f.write("Kappa : {:0.5f} ".format(metrics.cohen_kappa_score(y_test, y_pred)))
        f.write("F1 : {:0.5f} ".format(metrics.f1_score(y_test, y_pred, average='weighted')))
        f.write("total time: {:.5f}s \n".format(total_time))
    elif str(FLAGS.mode) == 'pred':
        for i in range(y_pred.shape[0]):
            f.write(str(y_pred[i]) + '\n')
