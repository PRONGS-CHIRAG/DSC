import numpy as np 
from keras.utils import np_utils
#import np_utils
#from np_utils import to_categorical
import tensorflow as tf 
from keras.models import Sequential
from keras.layers.core import Dense,Activation
tf.python_io=tf
np.random.seed(42)
X=np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y=np.array([[0],[1],[1],[0]]).astype('float32')
#One-hot encoding the data
y=np_utils.to_categorical(y)
#Building the model
xor=Sequential()
#Adding Layers
xor.add(Dense(8,input_dim=2))
xor.add(Activation("tanh"))
xor.add(Dense(2))
xor.add(Activation('sigmoid'))
xor.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
xor.summary()
xor.fit(X,y,nb_epoch=50,verbose=0)
score=xor.evaluate(X,y)
print("\n Accuracy :",score[-1])
print("\n Predictions :")
print(xor.predict_proba(X))

