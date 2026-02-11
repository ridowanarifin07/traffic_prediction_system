import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

# Page config
st.set_page_config(page_title="Smart Traffic Prediction System - Chattogram", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #0066cc; color: white; border-radius: 8px;}
    h1, h2, h3 {color: #0066cc;}
    .alert-high {background-color: #ff4b4b; color: white; padding: 10px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ------------------- Login System -------------------
if not st.session_state.get("authenticated", False):
    st.title("🚦 Smart Traffic Prediction System")
    st.markdown("### Login to continue")
    col1, col2 = st.columns([1,1])
    with col1:
        username = st.text_input("Username")
    with col2:
        password = st.text_input("Password", type="password")
    if st.button("Login"):
        if (username == "user" and password == "user123") or (username == "admin" and password == "admin123"):
            st.session_state.authenticated = True
            st.session_state.role = "admin" if username == "admin" else "user"
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Wrong credentials. Try user/user123 or admin/admin123")
    st.stop()

# Logged in
role = st.session_state.role
username = st.session_state.username
st.sidebar.success(f"Welcome {username}! ({role.capitalize()})")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

st.title(f"🚦 Smart Traffic Prediction System - Chattogram")
st.markdown(f"**Personalized Dashboard | Hello {username}!**")

# ------------------- Locations (Chattogram Real Junctions) -------------------
locations = [
    {"name": "GEC Circle", "lat": 22.3600, "lon": 91.8331},
    {"name": "Agrabad", "lat": 22.3236, "lon": 91.8119},
    {"name": "Tiger Pass", "lat": 22.3428, "lon": 91.8206},
    {"name": "2 No. Gate", "lat": 22.3372, "lon": 91.8325},
    {"name": "Bahaddarhat", "lat": 22.3708, "lon": 91.8514},
    {"name": "Oxygen More", "lat": 22.3819, "lon": 91.8556},
    {"name": "Muradpur", "lat": 22.3678, "lon": 91.8422},
    {"name": "Probartak Circle", "lat": 22.3542, "lon": 91.8264},
]
location_names = [loc["name"] for loc in locations]
location_boost_dict = {
    "GEC Circle": 400, "Agrabad": 600, "Tiger Pass": 300, "2 No. Gate": 350,
    "Bahaddarhat": 450, "Oxygen More": 300, "Muradpur": 400, "Probartak Circle": 350
}

# ------------------- Data Input -------------------
st.sidebar.header("📊 Data Input")
data_option = st.sidebar.radio("Data source", ["Sample Synthetic Data (Chattogram)", "Upload CSV"])

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        st.success("CSV uploaded!")
    else:
        st.info("CSV must have columns: timestamp, traffic_volume, temperature, rain, is_holiday etc.")
        st.stop()
else:
    @st.cache_data
    def generate_data():
        dates = pd.date_range("2024-01-01", periods=8000, freq='15T')
        np.random.seed(42)
        n = len(dates)
        base_volume = 400 + 350 * np.sin(np.arange(n) * 2 * np.pi / 96)
        peak_boost = np.where((dates.hour >= 7) & (dates.hour <= 9), 700, np.where((dates.hour >= 17) & (dates.hour <= 19), 600, 0))
        weekend_reduction = np.where(dates.dayofweek >= 5, -250, 0)
        rain_effect = np.random.choice([0, -350], n, p=[0.8, 0.2])
        holiday_effect = np.random.choice([0, -200], n, p=[0.95, 0.05])
        
        df = pd.DataFrame({'timestamp': dates})
        df['location_id'] = np.random.randint(0, len(locations), n)
        df['location_name'] = df['location_id'].map(lambda i: locations[i]['name'])
        df['lat'] = df['location_id'].map(lambda i: locations[i]['lat'])
        df['lon'] = df['location_id'].map(lambda i: locations[i]['lon'])
        df['location_boost'] = df['location_name'].map(location_boost_dict).fillna(200)
        
        df['accident'] = np.random.choice([0, 1], n, p=[0.98, 0.02])
        accident_effect = np.where(df['accident'] == 1, -400, 0)
        
        df['traffic_volume'] = np.clip(base_volume + peak_boost + weekend_reduction + rain_effect + holiday_effect 
                                       + df['location_boost'] + accident_effect + np.random.normal(0, 60, n), 50, 1600)
        
        df['temperature'] = np.random.normal(28, 6, n)
        df['rain'] = np.random.choice([0, 1], n, p=[0.8, 0.2])
        df['is_weekend'] = (dates.dayofweek >= 5).astype(int)
        df['is_holiday'] = np.random.choice([0, 1], n, p=[0.95, 0.05])
        df['hour'] = dates.hour
        df['day_of_week'] = dates.dayofweek
        df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
        
        return df.drop(columns=['location_boost'])
    
    data = generate_data()

# Preprocessing
data['congestion_level'] = pd.cut(data['traffic_volume'], bins=[0, 400, 800, 2000], labels=["Low", "Medium", "High"])
mean_vol = data['traffic_volume'].mean()
std_vol = data['traffic_volume'].std()
data['anomaly'] = (data['traffic_volume'] > mean_vol + 3 * std_vol) | (data['traffic_volume'] < mean_vol - 3 * std_vol)

# ------------------- Model Training -------------------
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=150, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=150, random_state=42)
    }
    
    results = {}
    best_mae = float('inf')
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2, "model": model}
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name
    
    return results, best_model, best_name

features = ['hour', 'day_of_week', 'temperature', 'rain', 'is_weekend', 'is_holiday', 'is_peak_hour', 'location_id', 'accident']
X = data[features]
y = data['traffic_volume']

results, best_model, best_name = train_models(X, y)

# ------------------- Tabs -------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Prediction", "Visualizations", "3D Map", "Route Planner"])

with tab1:
    st.subheader("🏆 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best Model", best_name)
        st.metric("MAE", f"{results[best_name]['MAE']:.1f}")
        st.metric("R²", f"{results[best_name]['R²']:.3f}")
        st.metric("Anomalies Detected", data['anomaly'].sum())
    with col2:
        metrics_df = pd.DataFrame({k: v for k, v in results.items() if k != "model"}).T
        st.dataframe(metrics_df.style.highlight_min(subset=["MAE", "RMSE"], color='lightgreen'))
    
    # Feature Importance (if tree model)
    if hasattr(best_model, "feature_importances_"):
        imp_df = pd.DataFrame({"Feature": features, "Importance": best_model.feature_importances_}).sort_values("Importance", ascending=False)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="Feature Importance (Explainable AI)")
        st.plotly_chart(fig_imp, use_container_width=True)
    
    st.subheader("Location-wise Average Traffic")
    loc_vol = data.groupby('location_name')['traffic_volume'].mean().sort_values(ascending=False)
    fig_loc = px.bar(loc_vol, title="Average Traffic Volume by Junction")
    st.plotly_chart(fig_loc, use_container_width=True)

with tab2:
    st.subheader("🔮 One-Click Traffic Prediction")
    pred_location_name = st.selectbox("Select Location", location_names)
    pred_location_id = location_names.index(pred_location_name)
    pred_hour = st.slider("Hour", 0, 23, 12)
    pred_day = st.selectbox("Day of week (0=Mon)", options=list(range(7)), index=1)
    pred_temp = st.number_input("Temperature (°C)", 15, 40, 30)
    pred_rain = st.selectbox("Raining?", [0, 1])
    pred_weekend = st.selectbox("Weekend?", [0, 1])
    pred_holiday = st.selectbox("Holiday?", [0, 1])
    pred_accident = st.selectbox("Accident Reported?", [0, 1], index=0)
    is_peak = 1 if pred_hour in [7,8,9,17,18,19] else 0
    
    input_feat = np.array([[pred_hour, pred_day, pred_temp, pred_rain, pred_weekend, pred_holiday, is_peak, pred_location_id, pred_accident]])
    prediction = best_model.predict(input_feat)[0]
    pred_level = pd.cut([prediction], bins=[0, 400, 800, 2000], labels=["Low", "Medium", "High"])[0]
    confidence = "High" if results[best_name]['MAE'] < 80 else "Medium" if results[best_name]['MAE'] < 150 else "Low"
    
    st.success(f"**Predicted Volume: {prediction:.0f} vehicles/hour** at **{pred_location_name}**")
    st.info(f"**Congestion: {pred_level} | Confidence: {confidence}**")
    if pred_level == "High":
        st.markdown('<div class="alert-high">🚨 SMART ALERT: Severe congestion predicted!</div>', unsafe_allow_html=True)
    if pred_accident:
        st.warning("Accident flag active – expect delays")

with tab3:
    st.subheader("📈 Traffic Visualizations")
    min_time = data['timestamp'].min().to_pydatetime()
    max_time = data['timestamp'].max().to_pydatetime()
    start_t, end_t = st.slider("Time Range Slider (Replay)", min_value=min_time, max_value=max_time, value=(min_time, max_time))
    filtered = data[(data['timestamp'] >= start_t) & (data['timestamp'] <= end_t)]
    
    fig_trend = px.line(filtered.sort_values('timestamp'), x='timestamp', y='traffic_volume', color='location_name',
                        title="Traffic Trend (Interactive Replay)")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.subheader("Congestion Distribution")
    st.bar_chart(data['congestion_level'].value_counts())

with tab4:
    st.subheader("🗺️ 3D Interactive Traffic Heatmap (Chattogram)")
    heatmap_data = data.groupby(['lat', 'lon', 'location_name'])['traffic_volume'].mean().reset_index()
    
    layer = pdk.Layer(
        "ColumnLayer",
        data=heatmap_data,
        get_position=["lon", "lat"],
        get_elevation="traffic_volume",
        elevation_scale=20,
        radius=300,
        get_fill_color="[255, 255 - (traffic_volume / 10), 0, 180]",
        pickable=True,
        auto_highlight=True
    )
    
    view = pdk.ViewState(latitude=22.35, longitude=91.83, zoom=11, pitch=50, bearing=0)
    deck = pdk.Deck(layers=[layer], initial_view_state=view,
                    tooltip={"text": "{location_name}\nAvg Volume: {traffic_volume:.0f} vehicles"})
    st.pydeck_chart(deck)

with tab5:
    st.subheader("🛣️ Alternative Route Suggestion")
    col1, col2 = st.columns(2)
    with col1:
        start_loc = st.selectbox("Start", location_names)
    with col2:
        end_loc = st.selectbox("End", location_names)
    
    if start_loc != end_loc:
        # Hardcoded sample routes (expand as needed)
        routes_dict = {
            ("GEC Circle", "Agrabad"): {"direct": 25, "alt": "via Tiger Pass", "alt_time": 32},
            ("Agrabad", "GEC Circle"): {"direct": 25, "alt": "via Probartak", "alt_time": 30},
            ("Tiger Pass", "Bahaddarhat"): {"direct": 20, "alt": "via Oxygen", "alt_time": 28},
            ("Bahaddarhat", "Tiger Pass"): {"direct": 20, "alt": "via Muradpur", "alt_time": 25},
            # Add more as needed
        }
        key = (start_loc, end_loc)
        rev_key = (end_loc, start_loc)
        route_info = routes_dict.get(key) or routes_dict.get(rev_key)
        
        if route_info:
            start_vol = data[data['location_name']==start_loc]['traffic_volume'].mean()
            end_vol = data[data['location_name']==end_loc]['traffic_volume'].mean()
            avg_vol = (start_vol + end_vol) / 2
            congestion_factor = min(avg_vol / 1000, 1.5)
            est_time = route_info["direct"] * (1 + congestion_factor)
            
            st.success(f"**Direct Route Estimated: {est_time:.0f} minutes**")
            if congestion_factor > 0.8:
                alt_est = route_info["alt_time"] * (1 + congestion_factor * 0.8)
                st.warning(f"🚨 High congestion! Alternative route: {route_info['alt']} → {alt_est:.0f} min")
        else:
            st.info("No specific route data – general estimate: 30–45 min based on current traffic")

# ------------------- Admin Panel -------------------
if role == "admin":
    with st.expander("🔧 Admin Panel – Data Management & Retraining", expanded=False):
        st.subheader("Edit Traffic Data")
        edited_data = st.data_editor(data, num_rows="dynamic")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save & Download Updated Data"):
                csv = edited_data.to_csv(index=False)
                st.download_button("Download CSV", csv, "updated_traffic_data.csv", "text/csv")
        with col2:
            if st.button("🔄 Retrain All Models"):
                train_models.clear()
                st.success("Models retrained successfully!")
                st.rerun()

st.markdown("---")
st.caption("Thesis-Ready Traffic Prediction System | Built with Streamlit | Next: Real API / LSTM / Full Registration?")
