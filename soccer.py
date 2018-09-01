# -*- coding: utf-8 -*-
'''

@author: Denis Baskan - 878 571

We now use an artificial neural networl to predict the matches in soccer world cup 2018. 
Results will be saved as a csv-file.
We want to predict the score probabilities for a match. We'll take the two highest probabilities in a score vector.
These two values are then our predictions.
Therefore we have to transform the existing score results into a matrix.

Dropouts are used to prevent over-fitting.

Input: nation1,nation2,year,kind
Output: score1,score2


Make sure all packages are up to date. You also might need to update dask with 'conda install dask'


'''

from keras.models import model_from_json
import numpy as np
import pandas as pd
import datetime as dt
from functions import *
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout

#read the (training) dataset
dataset = pd.read_csv('soccer_results_all.csv',delimiter = ';')

#filter out data before 2012
#I decided to drop some data, because only few player had been playing in World Cup 2018 and before 2010
#dataset = dataset[dataset['year']>2009].reset_index(drop=True)
#network has performed worse after filtering data

#read the World Cup 2018 data
data = pd.read_csv('soccer_results_wc18.csv',delimiter = ';')

#data preprocessing for neural network
[x_train,y_train,x_test,y_test] = prepro_nn(dataset)

#%%

################### Create and train the neural network ################### 

#initialising the ANN
classifier = Sequential()

#we need to find out the size of the input layer
n = x_train[0].size 
m = n +50

classifier.add(Dense(units = m, activation = 'relu', input_dim = n))    #adding the input layer and the first hidden layer
classifier.add(Dropout(0.2))                                                 #use dropout to randomly disable nodes
classifier.add(Dense(units = m, kernel_initializer = 'uniform', activation = 'relu')) #adding a hidden layer
classifier.add(Dense(units = m, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = m, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = len(y_train[0]), kernel_initializer = 'uniform', activation = 'sigmoid')) #adding the output layer

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the Training set
history = classifier.fit(x_train, y_train, batch_size = 250, epochs = 100)#2000

plot_nn(history)

#%%

################### Make predictions and evaluate the model ################### 

#predicting the Test set results
y_pred = classifier.predict(x_test)

#find 2 max values that represent the scores
y_pred2 = [x.argsort()[-2:] for x in y_pred]
fyp = [np.array([min(val)%10,max(val)%10]) for val in y_pred2] 
fyp = [item for sublist in fyp for item in sublist] #transform scores into a vector

#save the scores in a new data frame
d = dataset[dataset['kind'] == 'wm2018'] 
d.loc[:,'s1p'] = fyp[::2]  #predicted score for player 1
d['s2p'] = fyp[1::2] #predicted score for player 2
s = [(data[['s1','s2']][((data['n1']==row[1][0]) & (data['n2']==row[1][1]))].values[0]).astype(int) if ((data['n1']==row[1][0]) & (data['n2']==row[1][1])).any() else [-1,-1] for row in d.iterrows()]
d[['s1','s2']] = s

#extract actual scores from data frame
fyt = [item for sublist in d[['s1','s2']].values for item in sublist]

#evaluate model and give points for predictions
points = result(fyp,fyt)
print(points)

# Making the Confusion Matrix (goal-wise predictions)
cm = confusion_matrix(fyt, fyp)

np.trace(cm) / cm.size

d.to_csv('result/resultall_' + str(dt.datetime.now())[:19].replace(':','-') + '_P' + str(points) + '.csv')




#%%
################### Save and load your trained neural network ################### 


# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
 

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
