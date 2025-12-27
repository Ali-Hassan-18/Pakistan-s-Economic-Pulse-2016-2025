import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import re
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Climate & Trade Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS FOR COLORFUL KPIs ----------------
st.markdown("""
<style>
    .kpi-card {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-blue { border-top: 5px solid #007bff; background-color: #f0f7ff; }
    .kpi-green { border-top: 5px solid #28a745; background-color: #f1fdf4; }
    .kpi-orange { border-top: 5px solid #fd7e14; background-color: #fff9f0; }
    .kpi-red { border-top: 5px solid #dc3545; background-color: #fff5f5; }
    
    .kpi-title { font-size: 13px; color: #666; font-weight: bold; text-transform: uppercase; }
    .kpi-value { font-size: 24px; color: #222; font-weight: 800; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

def colorful_metric(label, value, color_class="kpi-blue"):
    st.markdown(f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-title">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    # Update these paths if running in a cloud environment
    trade = pd.read_csv("Cleaned_Trade_dataset.csv")
    climate = pd.read_csv("Cleaned_Climate_Dataset.csv")
    qim = pd.read_csv("Cleaned_QIM_dataset.csv")

    # Clean column names
    for df in [trade, climate, qim]:
        df.columns = [c.strip().lower().replace(" ", "_").rstrip("_") for c in df.columns]

    # Enhanced Category Name Cleaning for Trade
    def clean_category(name):
        name = re.sub(r'^S\s+Services\s+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+\(Fob\)$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+N\.I\.E\.$', '', name, flags=re.IGNORECASE)
        name = name.replace("  ", " ").strip()
        return name

    # Process Trade
    trade = trade[trade["trade_type"].isin(['Export', 'Import'])].copy()
    trade["service_type"] = trade["service_type"].apply(clean_category)
    trade["observation_date"] = pd.to_datetime(trade["observation_date"])
    trade["year"] = trade["observation_date"].dt.year
    trade["month_year"] = trade["observation_date"].dt.strftime('%Y-%m')

    # Process Climate
    climate = climate.dropna(subset=['date']).copy()
    climate["date"] = pd.to_datetime(climate["date"])
    climate["year"] = climate["year"].astype(int)
    climate["month_year"] = climate["date"].dt.strftime('%Y-%m')
    climate["precipitation_mm"] = climate["total_precipitation_m"] * 1000

    # Process QIM
    qim["date"] = pd.to_datetime(qim["date"])
    qim["year"] = qim["date"].dt.year
    qim["month_year"] = qim["date"].dt.strftime('%Y-%m')

    return trade, climate, qim

trade_df, climate_df, qim_df = load_data()

# Cache for top industries calculation
@st.cache_data
def get_top_industries(_qim_df):
    industry_stats = _qim_df.groupby('industry')['index_value'].mean().reset_index()
    return industry_stats.sort_values('index_value', ascending=False)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 IDS Analytics")
page = st.sidebar.radio("Navigation", ["📈 Trade Analysis", "🌦️ Climate Trends", "🏭 Industry (QIM)", "🔗 Correlation", "🔮 Future Forecast"])

# ==================================================
# 📈 TRADE PAGE
# ==================================================
if page == "📈 Trade Analysis":
    st.title("📈 Trade Analysis")
    st.subheader("🔎 Trade Filters")
    years = sorted(trade_df["year"].unique())
    year_f = st.multiselect("Select Years", years, default=years[-5:])
    filtered = trade_df[trade_df["year"].isin(year_f)]

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: colorful_metric("Total Value", f"${filtered['observation_value'].sum():,.0f}M", "kpi-blue")
    with c2: colorful_metric("Export Total", f"${filtered[filtered['trade_type']=='Export']['observation_value'].sum():,.0f}M", "kpi-green")
    with c3: colorful_metric("Import Total", f"${filtered[filtered['trade_type']=='Import']['observation_value'].sum():,.0f}M", "kpi-red")
    with c4: colorful_metric("Active Categories", filtered["service_type"].nunique(), "kpi-orange")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Import vs Export: Top 5 Leading Sectors")
    exclude_list = ['Goods And Services Balance', 'Total', 'Goods And Services Total', 'S Goods And Services Total']
    detailed_trade = filtered[~filtered['service_type'].isin(exclude_list)]
    
    top_5 = detailed_trade.groupby('service_type')['observation_value'].sum().nlargest(5).index
    comp_df = detailed_trade[detailed_trade['service_type'].isin(top_5)]
    
    fig_comp = px.bar(
        comp_df.groupby(['service_type', 'trade_type'])['observation_value'].sum().reset_index(),
        x="observation_value", y="service_type", color="trade_type", barmode="group",
        orientation="h", color_discrete_map={'Export': '#2ecc71', 'Import': '#e74c3c'},
        labels={'observation_value': 'Value (Millions USD)', 'service_type': 'Category'}
    )
    fig_comp.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
    st.plotly_chart(fig_comp, use_container_width=True, key="trade_top5")

    st.subheader("Monthly Performance Trend")
    trend_data = filtered.groupby(['month_year', 'trade_type'])['observation_value'].sum().reset_index().sort_values('month_year')
    fig_line = px.line(trend_data, x='month_year', y='observation_value', color='trade_type', 
                        markers=True, color_discrete_map={'Export': '#2ecc71', 'Import': '#e74c3c'})
    fig_line.update_layout(template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True, key="trade_trend")

# ==================================================
# 🌦️ CLIMATE PAGE
# ==================================================
elif page == "🌦️ Climate Trends":
    st.title("🌦️ Climate Trends")

    c1, c2, c3 = st.columns(3)
    with c1: colorful_metric("Avg Temp", f"{climate_df['avgtemp_c'].mean():.1f}°C", "kpi-red")
    with c2: colorful_metric("Max Rain", f"{climate_df['max5dayrainfall_mm'].max():.1f} mm", "kpi-blue")
    with c3: colorful_metric("Hot Days (>35°C)", int(climate_df['hotdays_over35c'].sum()), "kpi-orange")

    st.markdown("---")
    
    st.subheader("🌡️ Monthly Temperature Heatmap")
    heat_data = climate_df.pivot_table(index='month', columns='year', values='avgtemp_c', aggfunc='mean')
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    heat_data = heat_data.reindex(month_order)
    fig_heat = px.imshow(heat_data, color_continuous_scale='RdYlBu_r', text_auto=".1f")
    st.plotly_chart(fig_heat, use_container_width=True, key="temp_heatmap")

    st.subheader("📅 Timeline: Rainfall vs. Temperature Trends")
    clim_timeline = climate_df.sort_values('date')
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(
        go.Scatter(x=clim_timeline['date'], y=clim_timeline['precipitation_mm'], 
                    name="Rainfall (mm)", fill='tozeroy', mode='lines',
                    line=dict(color='rgba(0, 123, 255, 0.4)')),
        secondary_y=False,
    )
    fig_dual.add_trace(
        go.Scatter(x=clim_timeline['date'], y=clim_timeline['avgtemp_c'], 
                    name="Temperature (°C)", mode='lines+markers',
                    line=dict(color='#dc3545', width=2)),
        secondary_y=True,
    )
    fig_dual.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_dual, use_container_width=True, key="climate_timeline")

# ==================================================
# 🏭 QIM PAGE
# ==================================================
elif page == "🏭 Industry (QIM)":
    st.title("🏭 Industry Analysis")
    c1, c2, c3 = st.columns(3)
    with c1: colorful_metric("Sectors", qim_df['industry'].nunique(), "kpi-blue")
    with c2: colorful_metric("Peak Index", f"{qim_df['index_value'].max():.1f}", "kpi-green")
    with c3: colorful_metric("Avg Index", f"{qim_df['index_value'].mean():.1f}", "kpi-orange")

    # Get top industries
    industry_stats = get_top_industries(qim_df)
    top_10_industries = industry_stats.head(10)
    
    # Bar chart for top industries
    st.subheader("Top 10 Industries by Average Index")
    fig_bar = px.bar(
        top_10_industries,
        x='index_value',
        y='industry',
        orientation='h',
        color='index_value',
        color_continuous_scale='viridis',
        title='Top 10 Industries by Average Index Value',
        labels={'index_value': 'Average Index', 'industry': 'Industry'}
    )
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True, key="industry_bar")
    
    # Industry selection
    st.subheader("Industry Trends")
    sel = st.multiselect("Select Industries", sorted(qim_df['industry'].unique()), default=['Overall', 'Automobiles'])
    
    if sel:
        filtered_data = qim_df[qim_df['industry'].isin(sel)]
        fig_line = px.line(filtered_data, x='date', y='index_value', color='industry')
        fig_line.update_layout(template="plotly_white")
        st.plotly_chart(fig_line, use_container_width=True, key="industry_line")

# ==================================================
# 🔗 CORRELATION PAGE
# ==================================================
elif page == "🔗 Correlation":
    st.title("🔗 Data Correlation")
    t_m = trade_df.groupby('month_year')['observation_value'].sum().rename('Trade')
    c_m = climate_df.groupby('month_year')[['avgtemp_c', 'hotdays_over35c']].mean().rename(columns={'avgtemp_c':'Temp', 'hotdays_over35c':'Heat Days'})
    q_m = qim_df[qim_df['industry'] == 'Overall'].groupby('month_year')['index_value'].mean().rename('QIM Index')
    merged = pd.concat([t_m, c_m, q_m], axis=1).dropna()
    
    st.subheader("Correlation Matrix")
    st.plotly_chart(px.imshow(merged.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True, key="correlation_matrix")
    
    st.subheader("Relationship Explorer")
    f_col1, f_col2 = st.columns(2)
    with f_col1: x_var = st.selectbox("X Axis (Predictor)", merged.columns, index=1)
    with f_col2: y_var = st.selectbox("Y Axis (Outcome)", merged.columns, index=0)
    st.plotly_chart(px.scatter(merged, x=x_var, y=y_var, hover_name=merged.index, template="plotly_white"), 
                    use_container_width=True, key="scatter_plot")

# ==================================================
# 🔮 FUTURE FORECAST PAGE
# ==================================================
else:
    st.title("🔮 Future Trade Forecast (2016 - 2026)")
    st.markdown("""
    This page uses a **SARIMAX model** to predict Imports and Exports. 
    **Note:** The visualization and model training are focused on data from **2016 onwards**.
    """)
    
    # Add 6-12 month prediction input
    forecast_months = st.selectbox(
        "Select forecast period (months):",
        options=[6, 7, 8, 9, 10, 11, 12],
        index=6  # Default to 12 months
    )

    # --- 1. PREPARE & FILTER DATA ---
    # Aggregate to monthly level
    trade_pivot = trade_df.pivot_table(index='observation_date', columns='trade_type', values='observation_value', aggfunc='sum').reset_index()
    trade_pivot = trade_pivot.rename(columns={'Export': 'Total_Exports', 'Import': 'Total_Imports', 'observation_date': 'Month'})
    
    # NEW: Filter data to start from 2016
    trade_pivot = trade_pivot[trade_pivot['Month'] >= '2016-01-01'].dropna().sort_values('Month')
    
    # Exogenous feature: month of the year
    trade_pivot['month_idx'] = trade_pivot['Month'].dt.month
    X_exog = pd.get_dummies(trade_pivot['month_idx'], prefix='m').astype(float)

    # --- 2. FORECASTING FUNCTION (WITHOUT CACHE) ---
    def generate_plotly_forecast(series, name, color, forecast_months):
        # Target Scaling
        scaler = MinMaxScaler().fit(series.values.reshape(-1, 1))
        scaled_data = scaler.transform(series.values.reshape(-1, 1)).flatten()
        
        # Define and Fit Model
        model = SARIMAX(scaled_data, exog=X_exog, order=(1,1,1), seasonal_order=(0,1,1,12))
        res = model.fit(disp=False)
        
        # Generate forecast months Forecast
        if len(X_exog) >= forecast_months:
            exog_forecast = X_exog.tail(forecast_months)
        else:
            exog_forecast = X_exog.iloc[-forecast_months:]
        
        fcast_obj = res.get_forecast(steps=forecast_months, exog=exog_forecast)
        mean_scaled = fcast_obj.predicted_mean
        conf_scaled = fcast_obj.conf_int(alpha=0.05)
        
        # Inverse Scale
        mean = scaler.inverse_transform(mean_scaled.values.reshape(-1, 1)).flatten()
        low = scaler.inverse_transform(conf_scaled.iloc[:, 0].values.reshape(-1, 1)).flatten()
        up = scaler.inverse_transform(conf_scaled.iloc[:, 1].values.reshape(-1, 1)).flatten()
        
        # Create Future Dates
        future_dates = pd.date_range(start=trade_pivot['Month'].max(), periods=forecast_months+1, freq='MS')[1:]
        
        # Plotly Construction
        fig = go.Figure()

        # Shaded Confidence Interval
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(future_dates), pd.Series(future_dates)[::-1]]),
            y=pd.concat([pd.Series(up), pd.Series(low)[::-1]]),
            fill='toself',
            fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval"
        ))

        # Actual Historical Line
        fig.add_trace(go.Scatter(x=trade_pivot['Month'], y=series, name=f"Actual {name}", line=dict(color='#34495e', width=2)))

        # Forecasted Line
        fig.add_trace(go.Scatter(x=future_dates, y=mean, name=f"Forecast {name}", line=dict(color=color, width=3, dash='dash')))

        fig.update_layout(
            title=f"Predictive Trend for {name} (Post-2016) - {forecast_months} Month Forecast",
            xaxis_title="Timeline",
            yaxis_title="Value (Millions USD)",
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(range=['2016-01-01', future_dates.max()]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    # --- 3. LAYOUT ---
    tab_exp, tab_imp = st.tabs(["📊 Exports Forecast", "📊 Imports Forecast"])
    
    with tab_exp:
        fig_exp = generate_plotly_forecast(trade_pivot['Total_Exports'], "Exports", "#2ecc71", forecast_months)
        st.plotly_chart(fig_exp, use_container_width=True, key=f"export_forecast_{forecast_months}")
    
    with tab_imp:
        fig_imp = generate_plotly_forecast(trade_pivot['Total_Imports'], "Imports", "#e74c3c", forecast_months)
        st.plotly_chart(fig_imp, use_container_width=True, key=f"import_forecast_{forecast_months}")
    
    st.markdown("---")
    st.info("💡 **Insights:** The model shows seasonal fluctuations based on historical trade patterns. The widening of the shaded region in the future represents increasing mathematical uncertainty over time.")