import streamlit as st
import pickle
import numpy as np


data = pickle.load(open('data.pkl', 'rb'))
model = pickle.load(open('BR_reg.pkl', 'rb'))


st.title('Match Goals Prediction')
Attendance = st.number_input('Enter Attendance', data['Attendance'].unique())
Temperature = st.number_input('Enter Temperature 2m above ground', data['Temperature'].unique())
Relative_humidity = st.number_input('Enter Humidity 2m above ground', data['Relative_humidity'].unique())
Dewpoint  = st.number_input('Enter Dewpoint Temperature 2m above ground', data['Dewpoint '].unique())
SurfacePressure = st.number_input('Enter Surface pressure', data['SurfacePressure'].unique())
Rain = st.number_input('Enter Rainfall', data['Rain'].unique())
Wind_speed = st.number_input('Enter Wind speed 10m above ground level', data['Wind_speed'].unique())
Wind_direction = st.number_input('Enter wind direction 10m above ground', data['Wind_direction'].unique())


if st.button('Predict Total Goals in match'):

   
        query = np.array([Attendance, Temperature, Relative_humidity, Dewpoint, SurfacePressure, Rain, Wind_speed, Wind_direction])
        query = query.reshape(1, 8)

        st.title("Predicted Match Goals Are Between " + str(data.predict(query)[0]))

if __name__ == '__main__':
    main()