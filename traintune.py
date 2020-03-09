#Author:Chirag N Vijay
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve
def randomize(X,y):
	permutation=np.random.permutation(y.shape[0])
	X2=X[permutation,:]
	Y2=y[permutation]
	return X2,Y2
def draw_learning_curves(X, y, estimator, num_trainings):
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()
	plt.title("Learning Curves")
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	plt.plot(train_scores_mean, 'o-', color="g",label="Training score")
	plt.plot(test_scores_mean, 'o-', color="y",label="Cross-validation score")
	plt.legend(loc="best")
	plt.show()

data=pd.read_csv('train.csv')
X=np.array(data[['x1','x2']])
y=np.array(data['y'])
np.random.seed(55)
estimator = SVC(kernel='rbf',gamma=1000)
estimator1 = LogisticRegression()
estimator2 = GradientBoostingClassifier()
X,y=randomize(X,y)
draw_learning_curves(X,y,estimator,20)
