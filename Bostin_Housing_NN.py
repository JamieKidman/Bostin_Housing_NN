#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from keras.datasets import boston_housing


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

""""
http://lib.stat.cmu.edu/datasets/boston

 Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town !!!!!!! SORRY WHAT
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
 
 """

# The dataset is complete and doesnt have missing values.
# There are moral and potentially legal issues using B so the data will be removed

x_train = np.delete(x_train, 11, axis=1)
x_test = np.delete(x_test, 11, axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(12, input_dim=12, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(12, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(6, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))

model.compile(loss="mean_squared_error", optimizer="RMSprop", metrics=["mean_absolute_error"])

model.fit(x_train, y_train, epochs=5000, use_multiprocessing=True)


# In[38]:


model.evaluate(x_test, y_test)


# In[39]:


model.test_on_batch(x_test, y_test)
var = model.predict(x_test)


# In[40]:


for i, val in enumerate(y_test):
    diff = abs(val - var[i])
    
    print("index: {}\t {:5}  :  {}\t\t difference: {}".format(i, val, var[i], diff))


# In[ ]:




