from pandas import read_csv
#from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from numpy import mean
from numpy import std
from sklearn import model_selection 
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
# load data
data = read_csv('/Users/user/Downloads/Documents/winedata.csv')
dataset = data.values
# split data into X and y
X = dataset[1:501,4:6]
y = dataset[1:501,4]
# encode string class values as integers
X=LabelEncoder().fit_transform(y)
X=X.reshape(500,1)
#print(X)
label_encoded_y = LabelEncoder().fit_transform(y)
# grid search
model = BaggingClassifier(DecisionTreeClassifier())
n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
# summarize results
print("Best Accuracy: %f using %s" % (1+grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot
pyplot.figure(figsize=(15,5))
pyplot.errorbar(n_estimators, means, yerr=stds)
pyplot.title("Bagging Classifier n_estimators vs Log Loss")
pyplot.xlabel('n_estimators')
pyplot.ylabel('Log Loss')
pyplot.show()