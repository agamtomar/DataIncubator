
# Importing modules
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

np.random.seed(42)

# 10 Year all NBA dataset
year = 10
train = pd.read_csv('train_%dyr.csv' % year)
test = pd.read_csv('test_%dyr.csv' % year)

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

indexes = list(train[train['AllNBA10yr'] == 1].index)
indexes.extend(list(np.random.choice(list(train[train['AllNBA10yr'] == 0].index), 110, replace=False)))


X_train = np.array(train.loc[indexes, columns])
X_test = np.array(test.loc[:, columns])

y_train = np.array(train.loc[indexes, 'AllNBA%dyr' % year])
y_test = np.array(test.loc[:, 'AllNBA%dyr' % year])

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

## Looking the importance of age in getting the All NBA Award

df1 = train['AllNBA10yr'].groupby(train['agePlayer'])
df1.count().plot()
plt.show()

## Feature Selection
from sklearn.linear_model import LogisticRegressionCV

# Logistic with L1 Norm (Lasso)
clf_LogReg_L1Norm_CV = LogisticRegressionCV(Cs=100, cv=5, penalty='l1', solver='saga')
clf_LogReg_L2Norm_CV = LogisticRegressionCV(Cs=100, cv=5, penalty='l2')

# Fit the linear regression
clf_LogReg_L1Norm_CV.fit(X_train_std, y_train)
clf_LogReg_L2Norm_CV.fit(X_train_std, y_train)

from sklearn.linear_model import LogisticRegression

clf_LogReg_L1Norm = LogisticRegression(C=clf_LogReg_L1Norm_CV.C_[0], penalty='l1')
clf_LogReg_L1Norm.fit(X_train_std, y_train)


y_pred = clf_LogReg_L1Norm.predict(scaler.fit_transform(X_test))
LogReg_L1Norm_coeff_df = pd.DataFrame(index=columns, data=clf_LogReg_L1Norm.coef_.reshape(-1,1), columns=['Logistic (L1 norm) Coefficient'])

clf_LogReg_L2Norm = LogisticRegression(C=clf_LogReg_L2Norm_CV.C_[0], penalty='l2')
clf_LogReg_L2Norm.fit(X_train_std, y_train)

y_pred = clf_LogReg_L2Norm.predict(scaler.fit_transform(X_test))

print('In-Sample Accuracy', metrics.accuracy_score(y_true=y_train, y_pred=clf_LogReg_L2Norm.predict(X_train_std)))
print('Out-Sample Accuracy', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

LogReg_L2Norm_coeff_df = pd.DataFrame(index=columns, data=clf_LogReg_L2Norm.coef_.reshape(-1,1), columns=['Logistic (L2 norm) Coefficient'])
coeff_df = pd.concat([LogReg_L1Norm_coeff_df, LogReg_L2Norm_coeff_df], axis=1)


sns.set(style="whitegrid")

for col in coeff_df.columns:
       plt.figure(figsize=(20,12))
       sns.barplot(x=col, y=coeff_df.index, data=coeff_df)


coeff_df['Logistic (L1 norm) Coefficient'].nlargest()
coeff_df['Logistic (L2 norm) Coefficient'].nlargest()
