import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="EnergyGuard AI", page_icon="⚡", layout="wide")

st.markdown('''
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1E88E5; }
    .alert-box { background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f44336; }
</style>
''', unsafe_allow_html=True)

st.markdown('<p class="main-header">⚡ EnergyGuard AI</p>', unsafe_allow_html=True)
st.markdown("### Intelligent Anomaly Detection for Commercial Buildings")
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload your chilledwater.csv file", type=['csv'])

if uploaded_file is None:
    st.info("👆 Please upload your CSV file to begin analysis")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=['timestamp'])
    return df

df = load_data(uploaded_file)

# Preprocessing
missing_pct = df.isnull().sum() / len(df)
cols_to_keep = missing_pct[missing_pct < 0.95].index
df_clean = df[cols_to_keep].copy()
building_cols = [c for c in df_clean.columns if c != 'timestamp']
completeness = df_clean[building_cols].count().sort_values(ascending=False)
buildings = completeness.head(5).index.tolist()

st.sidebar.header("⚙️ Control Panel")
selected_building = st.sidebar.selectbox(
    "Select Building", 
    buildings, 
    format_func=lambda x: x.replace('_', ' ').title()
)

cost_per_unit = st.sidebar.slider("Energy Cost ($/unit)", 0.01, 0.20, 0.05, 0.01)
sensitivity = st.sidebar.slider("Detection Sensitivity", 1, 10, 5)

if st.button("🚀 Run Anomaly Detection", type="primary"):
    with st.spinner("Analyzing building data... Please wait"):

        # Prepare data
        df_analysis = df_clean[['timestamp', selected_building]].copy()
        df_analysis = df_analysis.dropna()
        df_analysis.set_index('timestamp', inplace=True)
        df_analysis[selected_building] = df_analysis[selected_building].interpolate(method='time')

        # Feature engineering
        df_analysis['hour'] = df_analysis.index.hour
        df_analysis['day_of_week'] = df_analysis.index.dayofweek
        df_analysis['month'] = df_analysis.index.month
        df_analysis['rolling_mean_7d'] = df_analysis[selected_building].rolling(168, min_periods=24).mean()
        df_analysis['rolling_std_7d'] = df_analysis[selected_building].rolling(168, min_periods=24).std()
        df_analysis = df_analysis.dropna()

        # Detection 1: Rolling Z-Score
        z_scores = np.abs((df_analysis[selected_building] - df_analysis['rolling_mean_7d']) / df_analysis['rolling_std_7d'])
        df_analysis['z_anomaly'] = (z_scores > 3.5) & (df_analysis[selected_building] > df_analysis['rolling_mean_7d'] * 1.5)

        # Detection 2: Isolation Forest
        feature_cols = ['hour', 'day_of_week', 'month', 'rolling_mean_7d']
        X = StandardScaler().fit_transform(df_analysis[feature_cols])
        iso = IsolationForest(contamination=0.02 * sensitivity/5, random_state=42)
        df_analysis['ml_anomaly'] = iso.fit_predict(X) == -1

        # Detection 3: Seasonal baseline
        seasonal_med = df_analysis.groupby([df_analysis.index.month, df_analysis.index.hour])[selected_building].transform('median')
        deviation = np.abs(df_analysis[selected_building] - seasonal_med) / seasonal_med
        df_analysis['seasonal_anomaly'] = deviation > 3.0

        # Combine (2+ methods)
        method_count = df_analysis['z_anomaly'].astype(int) + df_analysis['seasonal_anomaly'].astype(int)
        df_analysis['is_anomaly'] = (method_count >= 2) | (df_analysis['ml_anomaly'] & (method_count >= 1))

        # Metrics
        total_anomalies = df_analysis['is_anomaly'].sum()
        anomaly_rate = (total_anomalies / len(df_analysis)) * 100

        if total_anomalies > 0:
            normal_median = df_analysis[~df_analysis['is_anomaly']][selected_building].median()
            anomaly_vals = df_analysis[df_analysis['is_anomaly']][selected_building]
            cost = ((anomaly_vals - 2*normal_median).clip(lower=0).sum() * cost_per_unit)
        else:
            cost = 0

        health_score = max(0, 100 - anomaly_rate * 5)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🚨 Anomalies", int(total_anomalies))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Anomaly Rate", f"{anomaly_rate:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💚 Health Score", f"{health_score:.0f}/100")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Cost Impact", f"${cost:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Visualization
        st.subheader("📈 Consumption Timeline")

        fig = go.Figure()
        normal_data = df_analysis[~df_analysis['is_anomaly']]
        anomaly_data = df_analysis[df_analysis['is_anomaly']]

        fig.add_trace(go.Scatter(
            x=normal_data.index, 
            y=normal_data[selected_building],
            mode='lines',
            name='Normal Usage',
            line=dict(color='#1E88E5', width=1),
            opacity=0.7
        ))

        if len(anomaly_data) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data[selected_building],
                mode='markers',
                name='Anomaly',
                marker=dict(color='#f44336', size=10, symbol='x', line=dict(width=2)),
                hovertemplate='Date: %{x}<br>Usage: %{y}<br>Type: Anomaly'
            ))

        fig.update_layout(
            title=f"{selected_building.replace('_', ' ').title()} - Energy Consumption Analysis",
            xaxis_title="Date",
            yaxis_title="Consumption",
            hovermode='x unified',
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Anomaly list
        if total_anomalies > 0:
            st.subheader("📋 Detected Anomalies")
            anomaly_table = anomaly_data[[selected_building]].copy()
            anomaly_table['Severity'] = ['High' if v > normal_median * 3 else 'Medium' 
                                        for v in anomaly_data[selected_building]]
            anomaly_table.columns = ['Consumption', 'Severity']
            st.dataframe(anomaly_table.sort_index(ascending=False), use_container_width=True)

            csv = anomaly_table.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Anomaly Report (CSV)",
                data=csv,
                file_name=f'anomalies_{selected_building}.csv'
            )
        else:
            st.success("✅ No anomalies detected! Building is operating normally.")

        # Alert
        if anomaly_rate > 5:
            st.markdown(
                "<div class='alert-box'><strong>⚠️ High Anomaly Alert!</strong><br>" +
                f"{selected_building.replace('_', ' ').title()} shows {anomaly_rate:.1f}% anomaly rate. " +
                "Recommend immediate HVAC inspection.</div>", 
                unsafe_allow_html=True
            )

st.sidebar.markdown("---")
st.sidebar.info("""
How to use:
1. Upload chilledwater.csv
2. Select a building
3. Adjust sensitivity if needed
4. Click 'Run Anomaly Detection'
5. Download reports
""")
