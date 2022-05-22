# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:23:28 2022

@author: Eugenio Menacho de Góngora
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def getY(data):
    y = []
    
    for i in range(1, len(data)):
        y.append(data.High[i])
    return y





#We read the dataset
data = pd.read_csv("ADAUSDT.csv",low_memory=False, names = ["Date","Open","High", "Low", "Close", "Vol","quote_asset_volume", "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"])
data = data[1::]
data = data.reset_index(drop=True)

#We get the lastests dates
data_lastests_dates = data.Date[len(data)-201:len(data)].reset_index(drop=True)

#Change data to numeric
data[["Open","High", "Low", "Close", "Vol","quote_asset_volume", "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]] = data[["Open","High", "Low", "Close", "Vol","quote_asset_volume", "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]].apply(pd.to_numeric)


#Get the data for y, in this case the next row high value
y_aux = getY(data)

#Drop unnecesary columns
data = data.drop(['Date','quote_asset_volume', 'taker_buy_base_asset_volume', 'Open',"taker_buy_quote_asset_volume"], axis = 1)

#Set the scaler
scaler = MinMaxScaler()

#Scale values
data = scaler.fit_transform(data)


#Now for the LSTM model we use groups of two rows for X, so the Y value is the 
#next one
X = []
y = []

for i in range(2,len(data)-1):
    X.append(data[i-2:i])
    y.append(y_aux[i])


#Get the train and the test values
X_train = np.array(X[0:len(X)-200])
X_test = np.array(X[len(X)-200:len(X)])

y_train = np.array(y[0:len(y)-200])
y_test = np.array(y[len(y)-200:len(y)])



#Define the tensorflow model
model = Sequential() 
model.add(LSTM(units = 20, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))

model.add(LSTM(units = 30, activation = 'relu', return_sequences = True))

model.add(LSTM(units = 50, activation = 'relu', return_sequences = True))

model.add(LSTM(units = 70, activation = 'relu'))

model.add(Dense(units =1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')


#Train the model
iter = 1000
history= model.fit(X_train, y_train, epochs = iter, batch_size =70, validation_split=0.00001)

#Predict the test values
predictions = model.predict(X_test)

#Plot predictions
plt.plot(predictions, label="Predicciones Alto")
plt.plot(y_test, label="Alto")
plt.legend()
plt.title("EPOCHS: " + str(iter) )
plt.show()


#Plot the loss of the model
plt.plot(history.history['loss'][2:30])
plt.plot(history.history['val_loss'][2:30])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


