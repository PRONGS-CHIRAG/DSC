#Author : Chirag N Vijay
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc,precision_recall_curve,precision_score,recall_score,f1_score,average_precision_score
from inspect import signature
breast_cancer=load_breast_cancer()
X=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
y=pd.Categorical.from_codes(breast_cancer.target,breast_cancer.target_names)
ensoder = LabelEncoder()
y=pd.Series(ensoder.fit_transform(y))
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
probs=rf.predict_proba(X_test)
malignant_probs=probs[:,1]
fpr,tpr,thresholds = roc_curve(y_test,malignant_probs)
roc_auc=auc(fpr,tpr)
plt.title('Reciever Operator Characterstic')
plt.plot(fpr,tpr,'y',label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
y_pred=rf.predict(X_test)
print("Precision score is")
print(precision_score(y_test,y_pred))
print("recall score is")
print(recall_score(y_test,y_pred))
print("f1 score is")
print(f1_score(y_test,y_pred))
precision,recall,threshold = precision_recall_curve(y_test,y_pred)
average_precision=average_precision_score(y_test,y_pred)
print("Avg Precision score is")
print(average_precision)
step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='r', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='r', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()