#Author: Chirag N Vijay
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from math import log,sqrt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
mails=pd.read_csv('spam.csv',encoding='latin-1')
print(mails.head())
totalmails=mails['v1'].shape[0]
trainindex,testindex=list(),list()
for i in range(mails.shape[0]):
	if np.random.uniform(0,1)<0.75:
		trainindex += [i]
	else:
		testindex += [i]
traindata=mails.loc[trainindex]
testdata=mails.loc[testindex]
traindata.reset_index(inplace = True)
traindata.drop(['index'],axis=1,inplace=True)
print(traindata.head())