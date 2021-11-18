import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta, date
from sklearn.preprocessing import MinMaxScaler

stock_data = []
#Stock input
st.title('Stock Prediction')
code = st.text_input('Enter Stock-Ticker:')
start = st.date_input('Start date')
end = st.date_input('End date')

#Check stock code
try : 
    #feach data
    stock_data = data.DataReader(code,'yahoo',start,end)
except :
    st.markdown('_Enter stock ticker !._')

#Check start date
if start >= end :
    st.markdown('_Start date incorrect_')
else : 
    end_date = end.strftime('%d/%m/%Y')
    start_date = start.strftime('%d/%m/%Y')

#data description
    st.header('Data from '+start_date+' to '+end_date)
    st.write(stock_data.describe())

#plot closing price for the selected period
    st.subheader('Closing Price vs Time chart')
    figure = plt.figure(figsize=(12,6))
    plt.plot(stock_data.Close)
    st.pyplot(figure)
    
#create DataFrame
    s_date = start - timedelta(days=100)
    prediction_data = data.DataReader(code,'yahoo',s_date,end)
    df = pd.DataFrame(prediction_data['Close'])
    max_min = df['Close'].max() - df['Close'].min()
    scaler = MinMaxScaler(feature_range=(0,1))

#load LSTM model
    model = load_model('keras_model.h5',compile=False)

    input_data = scaler.fit_transform(df)
    x = []
    y = []

    for i in range(100 , input_data.shape[0]):
        x.append(input_data[i-100:i])
        y.append(input_data[i,0])

    x , y = np.array(x) , np.array(y)
    y_predicted = model.predict(x)

#Plot predicted price and actual price chart 
    st.subheader('Actual Price / Predicted Price Vs Time')
    figure2 = plt.figure(figsize=(12,6))
    plt.plot(y,'b',label='Original pricee')
    plt.plot(y_predicted,'r',label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(["Actual Price", "Predicted Price"])
    st.pyplot(figure2)

#forcast days input
    num_days = st.selectbox('Predict for : ', ['7 days','14 days','30 days'])
    if num_days == '7 days':
        days = 7
    elif num_days == '14 days':
        days = 14
    else : 
        days = 30

#generate new dataFrame
    forecast_data = data.DataReader(code,'yahoo',date.today()-timedelta(days=200),date.today())
    forecast_training = pd.DataFrame(forecast_data['Close'][len(forecast_data)-100:len(forecast_data)])
    input_forecast = scaler.fit_transform(forecast_training)
    y_forecast = []

    for i in range(0 , days) : 
        x_forecast = []
        x_forecast.append(input_forecast[0:100])
        x_forecast = np.array(x_forecast)
        price = model.predict(x_forecast)
        y_forecast.append(price[0][0])
        x_forecast = np.insert(x_forecast , 100 , price[0][0])
        x_forecast = np.delete(x_forecast , 0 ,axis=0)  
        x_forecast = scaler.fit_transform(pd.DataFrame(x_forecast))
        input_forecast = x_forecast
    
    new_scaler = MinMaxScaler(feature_range=(df['Close'].min(),df['Close'].max()))
    y_forecast = np.asarray(y_forecast)*max_min + df['Close'].min()

#plot forcasted price chart
    st.subheader('Forecast')
    figure3 = plt.figure(figsize=(12,6))
    plt.plot(y_forecast,'r',label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(["Actual Price", "Predicted Price"])
    st.pyplot(figure3)

