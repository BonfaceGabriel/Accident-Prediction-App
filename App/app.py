import streamlit as st
import datetime
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model_path = "./models/classifier.joblib"
model = joblib.load(model_path)

# Define the Streamlit app
st.title("Accident Prediction App")
cont = st.container(border=True)
cont.write("""
This app predicts whether an accident is likely to occur based on the input data.  
Fill in the parameters below and click **Submit** to get a prediction.
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
    start_lat = st.number_input("Start Latitude", format="%.6f", help="Enter the starting latitude (e.g., 37.7749)")
    wind_speed_bc = st.number_input("Wind Speed (m/s)", min_value=0.0, help="Enter the wind speed in meters per second")
with c2:
    amenity = st.checkbox("Amenity Nearby", help="Check if an amenity is nearby")
    crossing = st.checkbox("Crossing Nearby", help="Check if there is a crossing nearby")
    junction = st.checkbox("Junction Nearby", help="Check if there is a junction nearby")
    station = st.checkbox("Station Nearby", help="Check if there is a station nearby")
    stop = st.checkbox("Stop Nearby", help="Check if there is a stop nearby")
    traffic_signal = st.checkbox("Traffic Signal Nearby", help="Check if there is a traffic signal nearby")

# Time Attributes
# d = st.date_input("Accident Start Date", value=None)
# t = st.time_input("Accident start time", value=None)


# date = datetime.datetime(d)
# weekday_number = date.weekday()
# print(weekday_number)


cont2 = st.container(border=True)
cont2.subheader('Time Attributes')
c1, c2 = cont2.columns(2)
with c1:
    weekday = st.slider("Weekday (0=Monday, 6=Sunday)", 0, 6, help="Select the day of the week")
    hour = st.slider("Hour of Day (0-23)", 0, 23, help="Select the hour of the day")
    minute = st.slider("Minute of year", 0, 2000, help="Select the minute of the year")
with c2:
    timezone_pacific = st.checkbox("US/Pacific Timezone", help="Check if the timezone is US/Pacific")

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

# Organize inputs into a DataFrame
input_data = pd.DataFrame({
    "Start_Lat": [start_lat],
    "Amenity": [amenity],
    "Crossing": [crossing],
    "Junction": [junction],
    "Station": [station],
    "Stop": [stop],
    "Traffic_Signal": [traffic_signal],
    "Weekday": [weekday],
    "Hour": [hour],
    "Minute": [minute],
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
    "Timezone_US/Pacific": [timezone_pacific]
})

data = pd.read_csv('./datasets/train_data.csv')
model_data = pd.concat([data, input_data], axis=1)
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[numeric_cols])
data[numeric_cols] = scaled

# Submit button
if st.button("Submit"):
    try:
        # Make a prediction
        prediction = model.predict(data)[0]
        prediction_text = "Low Severity Accident" if prediction == 0 else "Very severe accident"
        st.success(f"Prediction: **{prediction_text}**")
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {e}")

st.write("Ensure all inputs are correctly filled before submitting.")

