import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl as data
from keras.models import load_model
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
start_date='2010-01-01'
end_date='2023-01-01'

st.title('Stock Price Prediction')

user_input=st.text_input('Enter Stock Ticker','TSLA')
user_input2=st.text_input('Enter Start Date',start_date)
user_input3=st.text_input('Enter End Date',end_date)
yf.pdr_override()
df=pdr.get_data_yahoo(user_input,start=user_input2,end=user_input3)

#Descriptio
st.subheader(f'Data from {user_input2} to {user_input3}')
st.write(df.describe())


#Visualisation
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close, "b", label="Close")
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart With 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, "r", label="MA100")
plt.plot(df.Close, "b", label="Close")
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart With 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, "r", label="MA100")
plt.plot(ma200, "g", label="MA200")
plt.plot(df.Close, "b", label="Close")
st.pyplot(fig)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


#Loading the model
load_model=load_model('keras_model.h5')

#Testing the model
past_100_days=data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=load_model.predict(x_test)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Predictions vs Original')
plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel("Time")
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)