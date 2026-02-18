import numpy as np
from compute_additional_metrics import pca_features
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

np.random.seed(0)
real = np.random.randn(50,200)
synth = real.copy()
X = np.vstack([real, synth])
y = np.hstack([np.zeros(len(real)), np.ones(len(synth))])
Xf = pca_features(X, n_components=50)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=2000)
aucs = []
for i,(tr,te) in enumerate(cv.split(Xf,y)):
    y_test = y[te]
    print(f'Fold {i}: test classes unique = {np.unique(y_test)}, counts = {np.bincount(y_test.astype(int))}')
    if len(np.unique(y_test))<2:
        print(' skip (single class)')
        continue
    clf.fit(Xf[tr], y[tr])
    probs = clf.predict_proba(Xf[te])[:,1]
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception as e:
        auc = None
        print(' auc compute error:', e)
    print(' auc=', auc)
    print(' probs sample:', probs[:10])
    print(' y_test sample:', y_test[:10])
    aucs.append(auc if auc is not None else 0.5)
print('mean auc', np.mean(aucs) if len(aucs)>0 else None)
print('Example predictions on full set:')
clf.fit(Xf, y)
probs_all = clf.predict_proba(Xf)[:,1]
print('probs_all min,max,mean:', probs_all.min(), probs_all.max(), probs_all.mean())
print('Sample y counts:', np.bincount(y.astype(int)))
