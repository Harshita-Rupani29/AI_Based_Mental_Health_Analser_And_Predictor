import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt

# Load Data
@st.cache_data
def load_data():
    # Replace with the path to your dataset
    data = pd.read_csv(r'C:\Users\USER\Desktop\AI\dff_output.csv')
    return data

data = load_data()

# Load Random Forest Model
rf = joblib.load(r'C:\Users\USER\Desktop\AI\rf_model.pkl')


# Sidebar
st.sidebar.title("AI Mental Health Dashboard")
option = st.sidebar.radio("Choose an option:", ["Mental Fitness Score", "Forecast Future Trends"])

# Encode the Country column
@st.cache_data
def get_country_mapping(data):
    unique_countries = data['Country'].unique()
    country_mapping = {country: idx for idx, country in enumerate(unique_countries)}
    return country_mapping, {idx: country for country, idx in country_mapping.items()}

# Load the country mapping
country_mapping, reverse_mapping = get_country_mapping(data)
data['Country'] = data['Country'].map(country_mapping)

# Option 1: Mental Fitness Score
if option == "Mental Fitness Score":
    st.title("Mental Fitness Score")
    st.write("Fill in your details to check your mental fitness score.")

    # User inputs
    country_name = st.selectbox(
        "Country",
        options=list(country_mapping.keys()),
        index=0,
        help="Select your country."
    )

    # Get the current year
    current_year = datetime.now().year

    # Year input
    year = st.number_input(
        "Year",
        min_value=int(data['Year'].min()),  # Minimum year from dataset
        max_value=current_year,  # Current year dynamically
        value=min(current_year, int(data['Year'].mean())),  # Default value: dataset mean or current year
        step=1,
        help="Enter the year for prediction (e.g., 2023)."
    )

    schi = st.number_input(
        "Schizophrenia rate (%)",
        min_value=0.0, max_value=10.0, value=0.2,
        help="Enter the schizophrenia rate as a percentage."
    )
    bipo = st.number_input(
        "Bipolar disorder rate (%)",
        min_value=0.0, max_value=10.0, value=0.7,
        help="Enter the bipolar disorder rate as a percentage."
    )
    eat = st.number_input(
        "Eating disorder rate (%)",
        min_value=0.0, max_value=10.0, value=0.1,
        help="Enter the eating disorder rate as a percentage."
    )
    anx = st.number_input(
        "Anxiety rate (%)",
        min_value=0.0, max_value=20.0, value=5.0,
        help="Enter the anxiety rate as a percentage."
    )
    drug = st.number_input(
        "Drug usage rate (%)",
        min_value=0.0, max_value=20.0, value=0.5,
        help="Enter the drug usage rate as a percentage."
    )
    depr = st.number_input(
        "Depression rate (%)",
        min_value=0.0, max_value=20.0, value=5.0,
        help="Enter the depression rate as a percentage."
    )
    alco = st.number_input(
        "Alcohol consumption rate (%)",
        min_value=0.0, max_value=20.0, value=0.5,
        help="Enter the alcohol consumption rate as a percentage."
    )

    # Convert country name to encoded value
    country_encoded = country_mapping[country_name]

    # Prediction button
    if st.button("Predict"):
        # Prepare input for RF model
        inputs = np.array([[country_encoded, year, schi, bipo, eat, anx, drug, depr, alco]])

        try:
            # Predict mental fitness
            prediction = rf.predict(inputs)[0]
            st.success(f"Your predicted mental fitness score: {100 - prediction:.2f}%")
            st.info(f"Disability Adjusted Life Years (DALYs): {prediction:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Option 2: Forecast Future Mental Fitness Trends
else:
    st.title("Forecast Future Mental Fitness Trends")
    st.write("Analyze mental fitness trends for a selected country and forecast future trends.")

    # Select a country
    country_list = reverse_mapping.values()
    country_input = st.selectbox("Select a Country:", list(country_list))

    # Start and End Year Input
    start_year = st.number_input(
        "Start Year",
        min_value=int(data['Year'].min()),
        max_value=datetime.now().year,
        value=int(data['Year'].min()),
        step=1,
        help="Enter the start year for viewing historical trends."
    )

    end_year = st.number_input(
        "End Year (Forecasting)",
        min_value=start_year,
        max_value=2050,
        value=2038,  
        step=1,
        help="Enter the end year for forecasting trends."
    )

    # Get encoded country value
    country_encoded = country_mapping[country_input]

    # Filter data for the selected country
    country_data = data[data['Country'] == country_encoded][['Year', 'Mental_fitness']].rename(columns={'Year': 'ds', 'Mental_fitness': 'y'})

    # Convert 'ds' column to datetime
    country_data['ds'] = pd.to_datetime(country_data['ds'], format='%Y')

    if len(country_data) > 0:
        # Train the Prophet model
        model = Prophet()
        model.fit(country_data)

        # Generate a future dataframe with years up to the specified end year
        periods_to_forecast = end_year - int(country_data['ds'].dt.year.max())
        future = model.make_future_dataframe(periods=periods_to_forecast, freq='Y')
        forecast = model.predict(future)

        # Filter forecast data to include only years from start_year to end_year
        forecast_filtered = forecast[(forecast['ds'].dt.year >= start_year) & (forecast['ds'].dt.year <= end_year)]

        # Plotting with confidence intervals
        st.write(f"Mental Fitness Trend for {country_input} ({start_year} to {end_year})")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual data
        ax.plot(country_data['ds'], country_data['y'], 'ko', label="Observed Data")

        # Plot forecast
        ax.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label="Forecasted Trend", color="blue", linewidth=2)

        # Add confidence intervals
        ax.fill_between(forecast_filtered['ds'],
                        forecast_filtered['yhat_lower'],
                        forecast_filtered['yhat_upper'],
                        color='blue', alpha=0.2, label="Confidence Interval")

        
        ax.set_title(f"Mental Fitness Trend for {country_input}", fontsize=16)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Mental Fitness", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=12)

        tick_interval = 5  
        years = range(start_year, end_year + 1, tick_interval)  
        ax.set_xticks(pd.to_datetime([str(year) for year in years])) 
        ax.set_xticklabels([str(year) for year in years], rotation=45, fontsize=10)  

        
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected country within the specified range.")
