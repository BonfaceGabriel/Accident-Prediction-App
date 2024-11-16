import streamlit as st
import datetime
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout='wide')

# Load the pre-trained model
model_path = "App/models/classifier.joblib"
model = joblib.load(model_path)

# Define the Streamlit app
with st.container():
    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
        </style>
        <h1 class="title">Accident Prediction App</h1>
    """, unsafe_allow_html=True)

st.header('Overview')

cont = st.container(border=True)
cont.write("""
- This app predicts the **Severity** of an accident based on the input data.  
- Fill in the parameters below and click **Submit** to get a prediction.
""")

container_style = """
    <style>
        .container1 {
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 20px;
            border-color: 'white';
        }
    </style>
    """

# User inputs
st.header("Input Parameters")

# Location Characteristics
cont1 = st.container(border=True)
cont1.subheader('Location Characteristics')
c1, c2 = cont1.columns(2)
with c1:
    start_lat = st.number_input("Start Latitude", format="%.2f", help="Enter the starting latitude (e.g., 37.7749)")
    wind_speed_bc = st.number_input("Wind Speed (m/s)", min_value=0.0, help="Enter the wind speed in meters per second")
with c2:
    amenity = st.checkbox("Amenity Nearby", help="Check if an amenity is nearby")
    crossing = st.checkbox("Crossing Nearby", help="Check if there is a crossing nearby")
    junction = st.checkbox("Junction Nearby", help="Check if there is a junction nearby")
    station = st.checkbox("Station Nearby", help="Check if there is a station nearby")
    stop = st.checkbox("Stop Nearby", help="Check if there is a stop nearby")
    traffic_signal = st.checkbox("Traffic Signal Nearby", help="Check if there is a traffic signal nearby")

#Time Attributes

cont2 = st.container(border=True)
cont2.subheader('Time Attributes')
c1, c2 = cont2.columns(2)
with c1:
    d = st.date_input("Accident Start Date")
    t = st.time_input("Accident start time")

with c2:
    timezone_pacific = st.checkbox("US/Pacific Timezone", help="Check if the timezone is US/Pacific")
    astronomical_twilight_night  = st.checkbox("Astronomical Twilight Night", help="Check if the timezone is US/Pacific")


# Road type checkboxes
cont3 = st.container(border=True)
cont3.subheader('Road Type')
c1, c2, c3 = cont3.columns(3)
with c1:
    rd = st.checkbox("Road (Rd)")
    st_ = st.checkbox("Street (St)")
    dr = st.checkbox("Drive (Dr)")
    ave = st.checkbox("Avenue (Ave)")
with c2:
    blvd = st.checkbox("Boulevard (Blvd)")
    ln = st.checkbox("Lane (Ln)")
    highway = st.checkbox("Highway")
    pkwy = st.checkbox("Parkway (Pkwy)")
with c3:
    hwy = st.checkbox("Highway (Hwy)")
    fwy = st.checkbox("Freeway (Fwy)")
    interstate = st.checkbox("Interstate (I-)")

weekday_number = d.weekday()
hour = t.hour
minute = t.minute

total_minute = hour * 60 + minute


# Organize inputs into a DataFrame
input_data = pd.DataFrame({
    "Start_Lat": [start_lat],
    "Amenity": [amenity],
    "Crossing": [crossing],
    "Junction": [junction],
    "Station": [station],
    "Stop": [stop],
    "Traffic_Signal": [traffic_signal],
    "Weekday": [weekday_number],
    "Hour": [hour],
    "Minute": [total_minute],
    "Rd": [rd],
    "St": [st_],
    "Dr": [dr],
    "Ave": [ave],
    "Blvd": [blvd],
    "Ln": [ln],
    "Highway": [highway],
    "Pkwy": [pkwy],
    "Hwy": [hwy],
    "Fwy": [fwy],
    "I-": [interstate],
    "Wind_Speed_bc": [wind_speed_bc],
    "Astronomical_Twilight_Night": [astronomical_twilight_night],
    "Timezone_US/Pacific": [timezone_pacific]
})

data = pd.read_csv('.App/datasets/train_data.csv')
model_data = pd.concat([data, input_data], ignore_index=True)
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(model_data[numeric_cols])
model_data[numeric_cols] = scaled

# Submit button
if st.button("Submit"):
    try:
        # Make a prediction
        prediction = model.predict(model_data)[-1]
        prediction = 1 if prediction > 0.5 else 0
        # print(prediction)
        if prediction == 0:
            prediction_text = "Low Severity Accident"
        elif prediction == 1:
            prediction_text = "Very Severe Accident"
        else:
            prediction_text = "Unknown Prediction"
        st.success(f"Prediction: {prediction}    **{prediction_text}**")
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {e}")

st.write("Ensure all inputs are correctly filled before submitting.")

