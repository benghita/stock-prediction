import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime

df = []
st.title('Stock Prediction')
code = st.text_input('Enter Stock-Ticker:')
start = st.date_input('Start date')
end = st.date_input('End date')

try : 
    df = data.DataReader(code,'yahoo',start,end)
except :
    st.markdown('_Enter stock ticker !._')

if start >= end :
    st.markdown('_Start date incorrect_')
else : 
    end_date = end.strftime('%m/%d/%Y')
    start_date = start.strftime('%m/%d/%Y')
    st.header('Data from '+start_date+' to '+end_date)
    st.write(df.describe())
    st.subheader('Closing Price vs Time chart')
    figure = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(figure)
                    ##
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
    data_test = pd.DataFrame(df['Close'][int(len(df)*0.7):len(df)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)
            #data_test_array = scaler.fit_transform(data_test)

    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_test , ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []

    for i in range(100 , input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test , y_test = np.array(x_test) , np.array(y_test)

    y_predicted = model.predict(x_test)

    st.subheader('prediction')
    figure2 = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original pricee')
    plt.plot(y_predicted,'r',label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(["Actual Price", "Predicted Price"])
    st.pyplot(figure2)

    num_days = st.selectbox('Predict for : ', ['7 days','14 days','30 days'])
    if num_days == '7 days':
        days = 7
    elif num_days == '14 days':
        days = 14
    else : 
        days = 30

    forecast_data = data.DataReader(code,'yahoo',datetime.date.today()-datetime.timedelta(days=200),datetime.date.today())
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


    st.subheader('Forecast')
    figure3 = plt.figure(figsize=(12,6))
    plt.plot(y_forecast,'r',label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(["Actual Price", "Predicted Price"])
    st.pyplot(figure3)

