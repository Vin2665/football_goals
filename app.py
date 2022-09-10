import streamlit as st
import pickle
import numpy as np


data = pickle.load(open('data.pkl', 'rb'))
model = pickle.load(open('BR_reg.pkl', 'rb'))


st.title('Match Goals Prediction')
Attendance = st.selectbox('Enter Attendance', data['Attendance'].unique())
Temperature = st.selectbox('Enter Temperature 2m above ground', data['Temperature'].unique())
Relative_humidity = st.selectbox('Enter Humidity 2m above ground', data['Relative_humidity'].unique())
Dewpoint  = st.selectbox('Enter Dewpoint Temperature 2m above ground', data['Dewpoint '].unique())
SurfacePressure = st.selectbox('Enter Surface pressure', data['SurfacePressure'].unique())
Rain = st.selectbox('Enter Rainfall', data['Rain'].unique())
Wind_speed = st.selectbox('Enter Wind speed 10m above ground level', data['Wind_speed'].unique())
Wind_direction = st.selectbox('Enter wind direction 10m above ground', data['Wind_direction'].unique())


if st.button('Predict Total Goals in match'):

   
        query = np.array([Attendance, Temperature, Relative_humidity, Dewpoint, SurfacePressure, Rain, Wind_speed, Wind_direction])
        query = query.reshape(1, 9)

        result = model.predict(query)

        print(result)

