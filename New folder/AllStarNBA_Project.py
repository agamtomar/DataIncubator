


"""


"""

# Importing modules
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import logging

logging.basicConfig(level=logging.DEBUG)


# 10 Year all NBA dataset
year = 10
train = pd.read_csv('train_%dyr.csv' % year)
test = pd.read_csv('test_%dyr.csv' % year)


#
df1 = train['AllNBA10yr'].groupby(train['agePlayer'])





cat_columns = ['groupPosition', 'slugTeamBREF']

columns = ['maxYrsPlayed', 'agePlayer', 'countGames', 'minutes',
           'ratioPER', 'pctTrueShooting', 'pct3PRate', 'pctFTRate', 'pctORB',
           'pctDRB', 'pctTRB', 'pctAST', 'pctSTL', 'pctBLK', 'pctTOV', 'pctUSG',
           'ratioOWS', 'ratioDWS', 'ratioWS', 'ratioWSPer48', 'ratioOBPM',
           'ratioDBPM', 'ratioBPM', 'ratioVORP', 'countTeamsPlayerSeason',
           'countGamesStarted', 'pctFG', 'pctFG3', 'pctFG2', 'pctEFG', 'pctFT',
           'minutesTotals', 'fgmTotals', 'fgaTotals', 'fg3mTotals', 'fg3aTotals',
           'fg2mTotals', 'fg2aTotals', 'ftmTotals', 'ftaTotals', 'orbTotals',
           'drbTotals', 'trbTotals', 'astTotals', 'stlTotals', 'blkTotals',
           'tovTotals', 'pfTotals', 'ptsTotals', 'countTeamsPlayerSeasonTotals',
           'minpergame']


X_train = np.array(train.loc[:, columns])
X_test = np.array(test.loc[:, columns])

y_train = np.array(train.loc[:, 'AllNBA%dyr' % year])
y_test = np.array(test.loc[:, 'AllNBA%dyr' % year])

# Creating DataFrames to save the results
classifier_names = ['Ridge', 'Logistic (L1 Norm)', 'Logistic (L2 Norm)', 'SVM_Linear', 'SVM_Poly (degree 3)', 'SVM_RBF', 'NBC', 'LDA', 'QDA']
Results_df = pd.DataFrame(index=classifier_names, columns=['In-Sample Accuracy', 'Out-Sample Accuracy'], dtype=np.float)

# Feature Selection
# Using Ridge (L2 Regularization - Not Scale invariant)

from sklearn.linear_model import RidgeClassifierCV
from sklearn import metrics

"""
By default, RidgeClassifierCV performs Generalized Cross-Validation, 
which is a form of efficient Leave-One-Out cross-validation.
"""
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

# Create Ridge Classifier with possible alpha values
ridge_lambdas = np.arange(0.1, 200, 0.1)
clf_Ridge_CV = RidgeClassifierCV(alphas=ridge_lambdas, cv=5, scoring='roc_auc')

# Fit the Ridge Classifier
clf_Ridge_CV.fit(X_train_std, y_train)

print('Selected Lambda Value for Ridge Model:', clf_Ridge_CV.alpha_)

# Performing Ridge Classification
from sklearn.linear_model import RidgeClassifier
clf_Ridge = RidgeClassifier(alpha=clf_Ridge_CV.alpha_)

clf_Ridge.fit(X_train_std, y_train)

y_pred = clf_Ridge.predict(scaler.fit_transform(X_test))

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_Ridge.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))


Results_df.loc['Ridge', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_Ridge.predict(X_train_std))
Results_df.loc['Ridge', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)

Ridge_coeff_df = pd.DataFrame(index=columns, data=clf_Ridge.coef_.reshape(-1,1), columns=['Ridge Coefficient'])



#########################  Using Logistic Regression  #############################################

from sklearn.linear_model import LogisticRegressionCV

# Logistic with L1 Norm (Lasso)
clf_LogReg_L1Norm_CV = LogisticRegressionCV(Cs=100, cv=5, penalty='l1', solver='saga', scoring='roc_auc')
clf_LogReg_L2Norm_CV = LogisticRegressionCV(Cs=100, cv=5, penalty='l2', scoring='roc_auc')

clf_LogReg_L1Norm_CV.fit(X_train_std, y_train)
clf_LogReg_L2Norm_CV.fit(X_train_std, y_train)

print('Selected C Value for Logistic Regression (L1 Norm Penalty) Model:', clf_LogReg_L1Norm_CV.C_[0])
print('Selected C Value for Logistic Regression (L2 Norm Penalty) Model:', clf_LogReg_L2Norm_CV.C_[0])

# Fit the Logistic regression
from sklearn.linear_model import LogisticRegression

clf_LogReg_L1Norm = LogisticRegression(C=clf_LogReg_L1Norm_CV.C_[0], penalty='l1')
clf_LogReg_L1Norm.fit(X_train_std, y_train)


y_pred = clf_LogReg_L1Norm.predict(scaler.fit_transform(X_test))

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_LogReg_L1Norm.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

Results_df.loc['Logistic (L1 Norm)', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_LogReg_L1Norm.predict(X_train_std))
Results_df.loc['Logistic (L1 Norm)', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)


LogReg_L1Norm_coeff_df = pd.DataFrame(index=columns, data=clf_LogReg_L1Norm.coef_.reshape(-1,1), columns=['Logistic (L1 norm) Coefficient'])



clf_LogReg_L2Norm = LogisticRegression(C=clf_LogReg_L2Norm_CV.C_[0], penalty='l2')
clf_LogReg_L2Norm.fit(X_train_std, y_train)

y_pred = clf_LogReg_L2Norm.predict(scaler.fit_transform(X_test))

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_LogReg_L2Norm.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

Results_df.loc['Logistic (L2 Norm)', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_LogReg_L2Norm.predict(X_train_std))
Results_df.loc['Logistic (L2 Norm)', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)


LogReg_L2Norm_coeff_df = pd.DataFrame(index=columns, data=clf_LogReg_L2Norm.coef_.reshape(-1,1), columns=['Logistic (L2 norm) Coefficient'])


# print Ridge and Logistic Coefficients
coeff_df = pd.concat([Ridge_coeff_df, LogReg_L1Norm_coeff_df, LogReg_L2Norm_coeff_df], axis=1)


sns.set(style="whitegrid")

for col in coeff_df.columns:
    plt.figure()
    sns.barplot(x=col, y=coeff_df.index, data=coeff_df)



#######################################################################################################################
#######################################################################################################################

############## Fitting SVM Model #############
from sklearn import svm

# SVM with Linear Kernel
clf_SVM_Linear = svm.SVC(kernel='linear')
clf_SVM_Linear.fit(X_train_std, y_train)


y_pred = clf_SVM_Linear.predict(X_test_std)

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_SVM_Linear.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

Results_df.loc['SVM_Linear', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_SVM_Linear.predict(X_train_std))
Results_df.loc['SVM_Linear', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)


# SVM with Polynomial Kernel
clf_SVM_Poly = svm.SVC(kernel='poly')
clf_SVM_Poly.fit(X_train_std, y_train)


y_pred = clf_SVM_Poly.predict(X_test_std)

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_SVM_Poly.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

Results_df.loc['SVM_Poly (degree 3)', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_SVM_Poly.predict(X_train_std))
Results_df.loc['SVM_Poly (degree 3)', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)


# SVM with RBF (Radial Basis Function) Kernel
clf_SVM_RBF = svm.SVC(kernel='rbf')
clf_SVM_RBF.fit(X_train_std, y_train)


y_pred = clf_SVM_RBF.predict(X_test_std)

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_SVM_RBF.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

Results_df.loc['SVM_RBF', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_SVM_RBF.predict(X_train_std))
Results_df.loc['SVM_RBF', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)


################################### Fitting NBC Model #################################################################
from sklearn.naive_bayes import GaussianNB

clf_NB_Gaussian = GaussianNB()
clf_NB_Gaussian.fit(X_train_std, y_train)

y_pred = clf_NB_Gaussian.predict(X_test_std)

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_NB_Gaussian.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

Results_df.loc['NBC', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_NB_Gaussian.predict(X_train_std))
Results_df.loc['NBC', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)



################################### Fitting LDA Model #################################################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

clf_LDA = LDA(n_components=2)

"""
A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.

The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix.

The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most discriminative directions.
"""

clf_LDA.fit(X_train_std, y_train)

y_pred = clf_LDA.predict(X_test_std)

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_LDA.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

Results_df.loc['LDA', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_LDA.predict(X_train_std))
Results_df.loc['LDA', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)


############ Fitting QDA Model #################
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
clf_QDA = QDA()
"""
Quadratic Discriminant Analysis

A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.

The model fits a Gaussian density to each class.
"""

clf_QDA.fit(X_train_std, y_train)

y_pred = clf_QDA.predict(X_test_std)

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_QDA.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

Results_df.loc['QDA', 'In-Sample Accuracy'] = metrics.accuracy_score(y_true=y_train, y_pred=clf_QDA.predict(X_train_std))
Results_df.loc['QDA', 'Out-Sample Accuracy'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)


################################################### ROC Curve ########################################################
classifiers = [clf_Ridge, clf_LogReg_L1Norm, clf_LogReg_L2Norm, clf_SVM_Linear, clf_SVM_Poly, clf_SVM_RBF, clf_NB_Gaussian, clf_LDA, clf_QDA]
y_score_df = pd.DataFrame(columns=classifier_names, dtype=np.float)

fig = plt.figure()
sns.set_palette('bright')
ax1 = plt.axes(frameon=True)

# removing top and right part of frame
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# setting ticks position
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

for idx in range(len(classifiers)):
    clf = classifiers[idx]

    if classifier_names[idx] in ['NBC', 'LDA', 'QDA']:
        y_score = clf.predict_proba(X_test_std)[:, 1]
    else:
        y_score = clf.decision_function(X_test_std)

    y_score_df.loc[:, classifier_names[idx]] = y_score

    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_score)

    plt.plot(fpr, tpr, label=classifier_names[idx])

plt.xlabel('Fpr', fontsize=28)
plt.ylabel('Tpr', fontsize=28)
plt.rc('legend', fontsize=28)
plt.tick_params(labelsize=28)
plt.legend()

###################################### Confusion matrix ###########################


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


for idx in range(len(classifiers)):
    clf = classifiers[idx]

    y_train_pred = clf.predict(X_train_std)
    y_test_pred = clf.predict(X_test_std)

    plot_confusion_matrix(y_true=y_train, y_pred=y_train_pred, classes=np.array([0, 1]), normalize=True,
                          title='In-Sample %s' % classifier_names[idx])
    plot_confusion_matrix(y_true=y_test, y_pred=y_test_pred, classes=np.array([0, 1]), normalize=True,
                          title='Out-Sample %s' % classifier_names[idx])


"""
from yellowbrick.classifier import ConfusionMatrix

for idx in range(len(classifiers)):
    clf = classifiers[idx]

    cm = ConfusionMatrix(clf, classes=[0, 1])

    plt.figure()
    cm.score(X_test_std, y_test)
    cm.poof()
    cm.set_title(title=classifier_names[idx])
"""


import scikitplot as skplt
for idx in range(len(classifiers)):
    clf = classifiers[idx]

    y_train_pred = clf.predict(X_train_std)
    y_test_pred = clf.predict(X_test_std)

    skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_test_pred, normalize=False)

