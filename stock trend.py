import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dense,Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn import linear_model
from datetime import date
import yfinance as yf
from keras.models import load_model
import streamlit as st


start= '2010-01-01'
today='2021-10-28'

col1, col2 = st.columns([1,1])
with col1:
        st.image("woxsen_logo.png")

with col2:
        st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','-')
df=yf.download(user_input,start,today)


st.subheader('Data from 2010 to YESTERDAY')
st.write(df.describe())

st.subheader('Closing price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)






st.subheader('Closing price vs Time chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


test = df
# Target column
target_close = pd.DataFrame(test['Close'])

feature_columns = ['Open', 'High', 'Low', 'Volume']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
feature_minmax_transform = pd.DataFrame(columns=feature_columns, data=feature_minmax_transform_data, index=test.index)


ts_split= TimeSeriesSplit(n_splits=10)
for train_index, test_index in ts_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = target_close[:len(train_index)].values.ravel(), target_close[len(train_index): (len(train_index)+len(test_index))].values.ravel()


        
X_train =np.array(X_train)
X_test =np.array(X_test)

X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])




from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(16, input_shape=(1, X_train.shape[1]), activation='relu', return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=200, batch_size=8, verbose=1, shuffle=False, callbacks=[early_stop])


y_pred_test_lstm = model_lstm.predict(X_tst_t)
y_train_pred_lstm = model_lstm.predict(X_tr_t)
score_lstm= model_lstm.evaluate(X_tst_t, y_test, batch_size=1)

y_pred_test_LSTM = model_lstm.predict(X_tst_t)


st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_LSTM, label='LSTM')

plt.xlabel('Observation')
plt.ylabel('INR_Scaled')
plt.legend()
st.pyplot(fig2)





