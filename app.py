import streamlit as st
import pickle
import numpy as np


data = pickle.load(open('data.pkl', 'rb'))
model = pickle.load(open('BR_reg.pkl', 'rb'))

def main():
    st.title('Match Goals Prediction')
    Attendance = st.number_input('Enter Attendance(In thousands)',min_value=1,max_value=200000,step=1000)
    Temperature = st.number_input('Enter Temperature(°C) 2m above ground',min_value=0.00,max_value=50.00,step=0.1)
    Relative_humidity = st.number_input('Enter Humidity(%) 2m above ground',min_value=1.00,max_value=120.00,step=1.00)
    Dewpoint  = st.number_input('Enter Dewpoint Temperature(°C) 2m above ground',min_value=-30.00,max_value=30.00,step=0.1)
    SurfacePressure = st.number_input('Enter Surface pressure(hPa)',min_value=700.00,max_value=1200.00,step=0.1)
    Rain = st.number_input('Enter Rainfall(mm)',min_value=0.00,max_value=10.00,step=0.1)
    Wind_speed = st.number_input('Enter Wind speed(km/h) 10m above ground level',min_value=1.00,max_value=50.00,step=0.1)
    Wind_direction = st.number_input('Enter wind direction(in degrees) 10m above ground',min_value=0.00,max_value=360.00,step=1.00)


    if st.button('Predict Total Goals in match'):

   
        query = np.array([Attendance, Temperature, Relative_humidity, Dewpoint, SurfacePressure, Rain, Wind_speed, Wind_direction])
        query = query.reshape(1, 8)

        st.title("Predicted Match Goals Are " + str(model.predict(query)[0]))

        result = model.predict(query)

        if result <2:
            st.success(f'The number of goals scored will be between 0 and 2')
        elif result >=2 & result <4:
            st.success(f'The number of goals scored will be between 2 and 4')
        elif result >=4 & result <6:
            st.success(f'The number of goals scored will be between 4 and 6')
        else: 
            st.success(f'The number of goals scored will be more than 6')

if __name__ == '__main__':
    main()