"""
🎣 Sri Lankan Fishing - Fish Catch Prediction System
====================================================
Streamlit Web Application

This application helps Sri Lankan fishermen predict their potential fish catch
based on location, environmental conditions, and fishing parameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fish Catch Predictor - Sri Lanka",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .info-box {
        background-color: #FFF9C4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FBC02D;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #C8E6C9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #388E3C;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFCCBC;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #E64A19;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts"""
    try:
        with open('models/fish_catch_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, feature_columns, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# Regional data
REGIONS_INFO = {
    'Negombo': {'lat': 7.20, 'lon': 79.88, 'avg_catch': 375},
    'Puttalam': {'lat': 8.05, 'lon': 79.83, 'avg_catch': 420},
    'Kalutara': {'lat': 6.58, 'lon': 79.96, 'avg_catch': 310},
    'Galle': {'lat': 6.05, 'lon': 80.22, 'avg_catch': 350},
    'Matara': {'lat': 5.95, 'lon': 80.55, 'avg_catch': 290},
    'Hambantota': {'lat': 6.12, 'lon': 81.13, 'avg_catch': 260},
    'Trincomalee': {'lat': 8.59, 'lon': 81.23, 'avg_catch': 305},
    'Batticaloa': {'lat': 7.72, 'lon': 81.70, 'avg_catch': 340},
    'Kalmunai': {'lat': 7.41, 'lon': 81.84, 'avg_catch': 250},
    'Jaffna': {'lat': 9.66, 'lon': 80.01, 'avg_catch': 220}
}

def main():
    # Header
    st.markdown('<p class="main-header">🎣 Sri Lankan Fishing Catch Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Fish Catch Prediction for Smart Fishing</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_columns, metadata = load_model_artifacts()
    
    if model is None:
        st.error("⚠️ Model files not found! Please ensure the model is trained and saved.")
        return
    
    # Sidebar - Model Info
    # with st.sidebar:
    #     st.image("https://via.placeholder.com/300x150/1E88E5/FFFFFF?text=Fish+Catch+AI", use_column_width=True)
    #     st.markdown("### 📊 Model Information")
    #     st.info(f"""
    #     **Model Type:** {metadata['model_name']}  
    #     **Accuracy (R²):** {metadata['test_r2_score']:.2%}  
    #     **Avg Error:** ±{metadata['mae']:.0f} kg  
    #     **Training Samples:** {metadata['training_samples']:,}
    #     """)
    
    
        
        st.markdown("---")
        st.markdown("### 🌊 About This System")
        st.markdown("""
        This AI system predicts fish catch amounts based on:
        - 🗺️ **Location** (10 regions)
        - 🌡️ **Sea conditions**
        - 🌙 **Moon phase**
        - ⛵ **Boat type**
        - 🎣 **Target species**
        - 👨‍🏭 **Fisherman experience**
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        1. Select your fishing region
        2. Enter trip details
        3. Specify environmental conditions
        4. Click **Predict Catch** button
        5. View your prediction & recommendations
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎣 Predict Catch", "📊 Model Performance", "ℹ️ User Guide"])
    
    # Tab 1: Prediction Interface
    with tab1:
        st.markdown("## Enter Your Fishing Trip Details")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            # Column 1: Location & Geography
            with col1:
                st.markdown("### 🗺️ Location Details")
                
                region = st.selectbox(
                    "Fishing Region",
                    options=list(REGIONS_INFO.keys()),
                    help="Select your fishing region"
                )
                
                # Auto-populate lat/lon based on region
                default_lat = REGIONS_INFO[region]['lat']
                default_lon = REGIONS_INFO[region]['lon']
                
                latitude = st.number_input(
                    "Latitude",
                    min_value=5.0,
                    max_value=10.0,
                    value=float(default_lat),
                    step=0.01,
                    format="%.4f"
                )
                
                longitude = st.number_input(
                    "Longitude",
                    min_value=79.0,
                    max_value=82.0,
                    value=float(default_lon),
                    step=0.01,
                    format="%.4f"
                )
                
                distance = st.slider(
                    "Distance from Shore (km)",
                    min_value=2.0,
                    max_value=50.0,
                    value=15.0,
                    step=1.0,
                    help="How far from shore will you fish?"
                )
                
                depth = st.slider(
                    "Water Depth (m)",
                    min_value=10.0,
                    max_value=200.0,
                    value=45.0,
                    step=5.0
                )
                
                fishing_zone = st.selectbox(
                    "Fishing Zone",
                    options=['Nearshore', 'Continental_Shelf', 'Offshore'],
                    index=1
                )
            
            # Column 2: Environmental Conditions
            with col2:
                st.markdown("### 🌊 Environmental Conditions")
                
                season = st.selectbox(
                    "Season",
                    options=['Inter_Monsoon', 'NE_Monsoon', 'SW_Monsoon'],
                    help="Current monsoon season"
                )
                
                sst = st.slider(
                    "Sea Surface Temperature (°C)",
                    min_value=25.0,
                    max_value=31.0,
                    value=28.0,
                    step=0.5
                )
                
                chlorophyll = st.slider(
                    "Chlorophyll Concentration (mg/m³)",
                    min_value=0.10,
                    max_value=0.50,
                    value=0.30,
                    step=0.05,
                    help="Indicator of plankton/fish food"
                )
                
                wind_speed = st.slider(
                    "Wind Speed (m/s)",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=1.0
                )
                
                wave_height = st.slider(
                    "Wave Height (m)",
                    min_value=0.5,
                    max_value=4.0,
                    value=1.2,
                    step=0.1
                )
            
            # Column 3: Fishing Details
            with col3:
                st.markdown("### ⛵ Fishing Details")
                
                moon_phase = st.selectbox(
                    "Moon Phase",
                    options=['New_Moon', 'First_Quarter', 'Full_Moon', 'Last_Quarter'],
                    index=2
                )
                
                time_of_day = st.selectbox(
                    "Time of Day",
                    options=['Morning', 'Midday', 'Afternoon'],
                    index=0
                )
                
                boat_type = st.selectbox(
                    "Boat Type",
                    options=['Traditional', 'One_Day', 'Multi_Day'],
                    index=2
                )
                
                fish_species = st.selectbox(
                    "Target Fish Species",
                    options=['Tuna', 'Sardine', 'Mackerel', 'Herring', 'Prawn'],
                    index=0
                )
                
                experience = st.slider(
                    "Fisherman Experience (years)",
                    min_value=1,
                    max_value=40,
                    value=20,
                    step=1
                )
                
                # Date selection
                trip_date = st.date_input(
                    "Trip Date",
                    value=datetime.now(),
                    help="Select your fishing trip date"
                )
            
            # Submit button
            submitted = st.form_submit_button("🎯 Predict Fish Catch", use_container_width=True)
        
        # Make prediction when form is submitted
        if submitted:
            # Prepare input data
            input_data = prepare_input_data(
                region, latitude, longitude, distance, depth, fishing_zone,
                season, sst, chlorophyll, wind_speed, wave_height,
                moon_phase, time_of_day, boat_type, fish_species, experience,
                trip_date
            )
            
            # Make prediction
            try:
                prediction = make_prediction(input_data, model, scaler, feature_columns, metadata)
                display_prediction_results(prediction, input_data, region)
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
    
    # Tab 2: Model Performance
    with tab2:
        st.markdown("## 📊 Model Performance Metrics")
        
        # col1, col2, col3, col4 = st.columns(4)
        
        # with col1:
        #     st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        #     st.metric("R² Score", f"{metadata['test_r2_score']:.2%}", 
        #              help="Proportion of variance explained by the model")
        #     st.markdown('</div>', unsafe_allow_html=True)
        
        # with col2:
        #     st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        #     st.metric("RMSE", f"{metadata['rmse']:.0f} kg",
        #              help="Root Mean Squared Error")
        #     st.markdown('</div>', unsafe_allow_html=True)
        
        # with col3:
        #     st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        #     st.metric("MAE", f"{metadata['mae']:.0f} kg",
        #              help="Mean Absolute Error")
        #     st.markdown('</div>', unsafe_allow_html=True)
        
        # with col4:
        #     st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        #     st.metric("Training Samples", f"{metadata['training_samples']:,}",
        #              help="Number of records used for training")
        #     st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model interpretation
        st.markdown("### 📈 What These Metrics Mean")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **R² Score: {metadata['test_r2_score']:.2%}**
            
            This means the model can explain {metadata['test_r2_score']*100:.1f}% of the variation 
            in fish catch amounts. This is considered {"excellent" if metadata['test_r2_score'] > 0.8 else "good" if metadata['test_r2_score'] > 0.7 else "fair"} 
            performance for real-world fishing predictions.
            """)
        
        with col2:
            st.info(f"""
            **Average Error: ±{metadata['mae']:.0f} kg**
            
            On average, predictions are within {metadata['mae']:.0f} kg of the actual catch. 
            For a typical catch of {REGIONS_INFO[list(REGIONS_INFO.keys())[0]]['avg_catch']} kg, 
            this represents a {(metadata['mae']/REGIONS_INFO[list(REGIONS_INFO.keys())[0]]['avg_catch']*100):.1f}% error margin.
            """)
        
        # Regional performance (simulated)
        st.markdown("### 🗺️ Average Catch by Region")
        
        regions_df = pd.DataFrame([
            {'Region': k, 'Average Catch (kg)': v['avg_catch']} 
            for k, v in REGIONS_INFO.items()
        ]).sort_values('Average Catch (kg)', ascending=False)
        
        fig = px.bar(regions_df, x='Region', y='Average Catch (kg)',
                    color='Average Catch (kg)',
                    color_continuous_scale='Blues',
                    title='Average Fish Catch by Region')
        
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: User Guide
    with tab3:
        st.markdown("## 📖 User Guide & Tips")
        
        st.markdown("### 🎯 How to Get the Best Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### ✅ Best Practices
            
            1. **Accurate Location Data**
               - Use GPS coordinates when possible
               - Select the correct region
            
            2. **Check Weather Forecast**
               - Update wind speed before trip
               - Monitor wave height predictions
            
            3. **Know Your Waters**
               - Estimate water depth accurately
               - Choose correct fishing zone
            
            4. **Seasonal Awareness**
               - Be aware of monsoon seasons
               - Factor in moon phases
            """)
        
        with col2:
            st.markdown("""
            #### 💡 Understanding Results
            
            1. **Prediction Range**
               - High (>500 kg): Excellent conditions
               - Medium (300-500 kg): Good conditions
               - Low (<300 kg): Fair/Poor conditions
            
            2. **Environmental Factors**
               - High chlorophyll = More fish
               - Calm seas = Better fishing
               - Full moon = Often better for some species
            
            3. **Experience Matters**
               - More experienced fishermen tend to catch more
               - Boat type significantly affects capacity
            """)
        
        st.markdown("---")
        
        st.markdown("### 🌊 Environmental Factors Explained")
        
        with st.expander("🌡️ Sea Surface Temperature (SST)"):
            st.markdown("""
            - **Optimal Range:** 27-29°C
            - **Effect:** Different species prefer different temperatures
            - **Tip:** Warmer waters during inter-monsoon periods often mean better fishing
            """)
        
        with st.expander("🌿 Chlorophyll Concentration"):
            st.markdown("""
            - **What is it:** Measure of plankton in water
            - **Effect:** High chlorophyll = More fish food = More fish
            - **Optimal:** 0.25-0.40 mg/m³
            - **Tip:** Green-tinted water often indicates good fishing
            """)
        
        with st.expander("💨 Wind Speed & Wave Height"):
            st.markdown("""
            - **Calm conditions:** <5 m/s wind, <1.5m waves = Best fishing
            - **Moderate:** 5-10 m/s wind = Still fishable
            - **Rough:** >15 m/s wind, >2.5m waves = Dangerous, avoid
            - **Tip:** Check marine forecast before departure
            """)
        
        with st.expander("🌙 Moon Phase"):
            st.markdown("""
            - **Full Moon:** Often best for pelagic fish (Tuna, Mackerel)
            - **New Moon:** Good for bottom fishing
            - **Quarter Moons:** Moderate activity
            - **Tip:** Many fishermen swear by full moon fishing!
            """)
        
        st.markdown("---")
        
        st.markdown("### ⚠️ Important Safety Notes")
        
        st.warning("""
        **⚠️ This tool is for guidance only!**
        
        - Always check official weather forecasts
        - Never venture out in dangerous conditions
        - Ensure boat is seaworthy and properly equipped
        - Follow all maritime safety regulations
        - Inform coastal authorities of your trip
        - Carry emergency communication devices
        
        **Your safety is more important than any catch!**
        """)
        
        st.markdown("---")
        
        st.success("""
        ### ✅ Benefits of Using This System
        
        1. **Better Planning** - Know expected catch before you go
        2. **Fuel Efficiency** - Plan fuel needs based on distance and expected catch
        3. **Time Optimization** - Fish when conditions are best
        4. **Resource Management** - Bring appropriate ice and storage
        5. **Sustainability** - Avoid trips during poor conditions, reducing waste
        """)

def prepare_input_data(region, latitude, longitude, distance, depth, fishing_zone,
                      season, sst, chlorophyll, wind_speed, wave_height,
                      moon_phase, time_of_day, boat_type, fish_species, experience,
                      trip_date):
    """Prepare input data for prediction"""
    
    # Extract date features
    year = trip_date.year
    month = trip_date.month
    day_of_week = trip_date.weekday()
    
    # Calculate derived features
    distance_depth_ratio = distance / (depth + 1)
    wind_wave_ratio = wind_speed / (wave_height + 0.1)
    is_monsoon = 1 if season in ['NE_Monsoon', 'SW_Monsoon'] else 0
    
    # Create input dictionary
    input_dict = {
        'Latitude': latitude,
        'Longitude': longitude,
        'Distance_From_Shore_km': distance,
        'Water_Depth_m': depth,
        'SST_Celsius': sst,
        'Chlorophyll_mg_m3': chlorophyll,
        'Wind_Speed_ms': wind_speed,
        'Wave_Height_m': wave_height,
        'Fisherman_Experience_Years': float(experience),
        'Year': year,
        'Month': month,
        'Day_of_Week': day_of_week,
        'Distance_Depth_Ratio': distance_depth_ratio,
        'Wind_Wave_Ratio': wind_wave_ratio,
        'Is_Monsoon': is_monsoon,
        # Categorical
        'Region': region,
        'Season': season,
        'Moon_Phase': moon_phase,
        'Time_Of_Day': time_of_day,
        'Boat_Type': boat_type,
        'Fish_Species': fish_species,
        'Fishing_Zone': fishing_zone
    }
    
    return input_dict

def make_prediction(input_data, model, scaler, feature_columns, metadata):
    """Make prediction using the trained model"""
    
    # Create dataframe
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    categorical_features = metadata['categorical_features']
    input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
    
    # Align with training features
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_columns]
    
    # Scale features
    input_scaled = scaler.transform(input_encoded)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    return prediction

def display_prediction_results(prediction, input_data, region):
    """Display prediction results with visualizations"""
    
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    # Main prediction box
    st.markdown(f"""
    <div class="prediction-box">
        <h2 style="color: #1E88E5; margin-bottom: 1rem;">🎣 Predicted Fish Catch</h2>
        <h1 style="color: #0D47A1; font-size: 4rem; margin: 0;">{prediction:.1f} kg</h1>
        <p style="color: #616161; margin-top: 0.5rem; font-size: 1.2rem;">Expected catch for your trip</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation & Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        # Condition assessment
        if prediction > 500:
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Excellent Fishing Conditions!</h3>
                <p><strong>Expected Range:</strong> {prediction*0.8:.0f} - {prediction*1.2:.0f} kg</p>
                <p><strong>Recommendation:</strong> Perfect conditions for a successful trip! 
                Make sure to bring sufficient ice and storage.</p>
            </div>
            """, unsafe_allow_html=True)
        elif prediction > 300:
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Good Fishing Conditions</h3>
                <p><strong>Expected Range:</strong> {prediction*0.8:.0f} - {prediction*1.2:.0f} kg</p>
                <p><strong>Recommendation:</strong> Good catch expected. Plan your trip accordingly.</p>
            </div>
            """, unsafe_allow_html=True)
        elif prediction > 150:
            st.markdown(f"""
            <div class="info-box">
                <h3>⚠️ Fair Fishing Conditions</h3>
                <p><strong>Expected Range:</strong> {prediction*0.8:.0f} - {prediction*1.2:.0f} kg</p>
                <p><strong>Recommendation:</strong> Moderate catch expected. Consider if fuel costs justify the trip.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ Poor Fishing Conditions</h3>
                <p><strong>Expected Range:</strong> {prediction*0.8:.0f} - {prediction*1.2:.0f} kg</p>
                <p><strong>Recommendation:</strong> Low catch expected. Consider rescheduling or choosing a different location.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Trip planning assistance
        st.markdown("### 📋 Trip Planning Guide")
        
        # Calculate requirements
        fuel_estimate = input_data['Distance_From_Shore_km'] * 2 * 2  # Round trip + reserve
        ice_needed = prediction * 0.5  # Rough estimate: 0.5kg ice per kg fish
        storage_capacity = prediction * 1.2  # 20% extra capacity
        
        st.info(f"""
        **Based on {prediction:.0f} kg expected catch:**
        
        ⛽ **Fuel Needed:** ~{fuel_estimate:.0f} liters  
        (Distance: {input_data['Distance_From_Shore_km']:.0f} km round trip)
        
        🧊 **Ice Required:** ~{ice_needed:.0f} kg  
        (For proper fish preservation)
        
        📦 **Storage Capacity:** ~{storage_capacity:.0f} kg  
        (Recommended boat capacity)
        
        👨‍👨‍👦 **Crew Size:** {3 if prediction > 400 else 2 if prediction > 200 else 1}-{4 if prediction > 400 else 3 if prediction > 200 else 2} persons  
        (For efficient handling)
        """)
    
    # Detailed condition summary
    st.markdown("---")
    st.markdown("### 📊 Trip Conditions Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🗺️ Location**")
        st.write(f"Region: {input_data['Region']}")
        st.write(f"Zone: {input_data['Fishing_Zone']}")
        st.write(f"Distance: {input_data['Distance_From_Shore_km']:.1f} km")
        st.write(f"Depth: {input_data['Water_Depth_m']:.0f} m")
    
    with col2:
        st.markdown("**🌊 Environment**")
        st.write(f"Season: {input_data['Season']}")
        st.write(f"Temperature: {input_data['SST_Celsius']:.1f}°C")
        st.write(f"Wind: {input_data['Wind_Speed_ms']:.1f} m/s")
        st.write(f"Waves: {input_data['Wave_Height_m']:.1f} m")
    
    with col3:
        st.markdown("**⛵ Details**")
        st.write(f"Boat: {input_data['Boat_Type']}")
        st.write(f"Species: {input_data['Fish_Species']}")
        st.write(f"Time: {input_data['Time_Of_Day']}")
        st.write(f"Moon: {input_data['Moon_Phase']}")
    
    # Gauge chart for catch potential
    st.markdown("---")
    st.markdown("### 📈 Catch Potential Gauge")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Catch (kg)", 'font': {'size': 24}},
        delta={'reference': REGIONS_INFO[region]['avg_catch'], 
               'valueformat': '.0f'},
        gauge={
            'axis': {'range': [None, 1000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 200], 'color': '#FFCDD2'},
                {'range': [200, 400], 'color': '#FFF9C4'},
                {'range': [400, 700], 'color': '#C8E6C9'},
                {'range': [700, 1000], 'color': '#81C784'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': REGIONS_INFO[region]['avg_catch']
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Red line indicates regional average ({REGIONS_INFO[region]['avg_catch']} kg for {region})")

if __name__ == "__main__":
    main()
