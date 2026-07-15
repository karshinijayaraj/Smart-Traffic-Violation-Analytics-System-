"""
Smart Traffic Violation Analytics Dashboard
--------------------------------------------
Run with:  streamlit run dashboard/app.py   (from the project root)

Tabs:
  1. Overview        - KPIs + trend charts
  2. Hotspot Map      - geographic violation hotspots
  3. Violation Explorer - filterable raw data table
  4. Risk Prediction   - live ML inference on user-entered violation details
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "traffic_violations.csv")
MODEL_PATH = os.path.join(ROOT, "outputs", "models", "risk_model.joblib")

st.set_page_config(page_title="Smart Traffic Violation Analytics", layout="wide", page_icon="🚦")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


df = load_data()
model = load_model()

st.title("🚦 Smart Traffic Violation Analytics System")
st.caption("AI + Computer Vision + Machine Learning powered smart-city traffic violation monitoring dashboard.")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
locations = st.sidebar.multiselect("Location", sorted(df["location"].unique()),
                                    default=sorted(df["location"].unique()))
vtypes = st.sidebar.multiselect("Vehicle Type", sorted(df["vehicle_type"].unique()),
                                 default=sorted(df["vehicle_type"].unique()))
vio_types = st.sidebar.multiselect("Violation Type", sorted(df["violation_type"].unique()),
                                    default=sorted(df["violation_type"].unique()))
date_range = st.sidebar.date_input(
    "Date range",
    value=(df["date"].min() if "date" in df.columns else df["timestamp"].min().date(),
           df["date"].max() if "date" in df.columns else df["timestamp"].max().date()),
)

fdf = df[
    df["location"].isin(locations)
    & df["vehicle_type"].isin(vtypes)
    & df["violation_type"].isin(vio_types)
]
if isinstance(date_range, tuple) and len(date_range) == 2:
    fdf = fdf[(pd.to_datetime(fdf["date"]) >= pd.to_datetime(date_range[0]))
              & (pd.to_datetime(fdf["date"]) <= pd.to_datetime(date_range[1]))]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Hotspot Map", "🔍 Violation Explorer", "🤖 Risk Prediction"])

# ================= TAB 1: OVERVIEW =================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Violations", f"{len(fdf):,}")
    c2.metric("Total Fines Issued", f"₹{fdf['fine_amount_inr'].sum():,.0f}")
    c3.metric("Repeat Offenders", f"{fdf['is_repeat_offender'].sum():,}")
    unpaid_pct = (1 - fdf["fine_paid"].mean()) * 100 if len(fdf) else 0
    c4.metric("Unpaid Fine Rate", f"{unpaid_pct:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        hourly = fdf.groupby("hour").size().reset_index(name="count")
        fig = px.bar(hourly, x="hour", y="count", title="Violations by Hour of Day",
                     color_discrete_sequence=["#3b82f6"])
        st.plotly_chart(fig, use_container_width=True)

        vt = fdf["violation_type"].value_counts().reset_index()
        vt.columns = ["violation_type", "count"]
        fig3 = px.bar(vt, x="count", y="violation_type", orientation="h",
                       title="Violations by Type", color_discrete_sequence=["#ef4444"])
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        daily = fdf.groupby("date").size().reset_index(name="count")
        fig2 = px.line(daily, x="date", y="count", title="Daily Violation Trend")
        st.plotly_chart(fig2, use_container_width=True)

        risk_counts = fdf["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig4 = px.pie(risk_counts, names="risk_level", values="count",
                       title="Risk Level Distribution",
                       color="risk_level",
                       color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"})
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Top Violation Hotspots")
    loc_counts = fdf["location"].value_counts().reset_index()
    loc_counts.columns = ["location", "violations"]
    fig5 = px.bar(loc_counts, x="violations", y="location", orientation="h",
                  color="violations", color_continuous_scale="Oranges")
    st.plotly_chart(fig5, use_container_width=True)

# ================= TAB 2: MAP =================
with tab2:
    st.subheader("Geographic Violation Hotspots")
    if len(fdf):
        fig_map = px.density_mapbox(
            fdf, lat="latitude", lon="longitude", radius=18,
            center=dict(lat=fdf["latitude"].mean(), lon=fdf["longitude"].mean()),
            zoom=11, mapbox_style="open-street-map",
            hover_data=["location", "violation_type"],
        )
        fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No data matches the current filters.")

# ================= TAB 3: EXPLORER =================
with tab3:
    st.subheader("Violation Records")
    st.write(f"Showing {len(fdf):,} of {len(df):,} total records.")
    st.dataframe(
        fdf.sort_values("timestamp", ascending=False)[
            ["violation_id", "timestamp", "location", "vehicle_type", "vehicle_plate",
             "violation_type", "speed_kmph", "speed_limit_kmph", "signal_status",
             "risk_level", "fine_amount_inr", "fine_paid"]
        ],
        use_container_width=True,
        height=500,
    )
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", csv, "filtered_violations.csv", "text/csv")

# ================= TAB 4: RISK PREDICTION =================
with tab4:
    st.subheader("Predict Violation Risk Level (Live ML Inference)")
    st.caption("Uses the trained RandomForest model (src/ml_model.py) to classify a new violation's risk.")

    if model is None:
        st.warning("Model file not found. Run `python3 src/ml_model.py` first to train and save it.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            vehicle_type = st.selectbox("Vehicle Type", sorted(df["vehicle_type"].unique()))
            violation_type = st.selectbox("Violation Type", sorted(df["violation_type"].unique()))
            location = st.selectbox("Location", sorted(df["location"].unique()))
        with c2:
            speed_kmph = st.slider("Speed (km/h)", 0, 150, 65)
            speed_limit_kmph = st.slider("Speed Limit (km/h)", 30, 80, 50)
            hour = st.slider("Hour of Day", 0, 23, 18)
        with c3:
            weather = st.selectbox("Weather", sorted(df["weather"].unique()))
            signal_status = st.selectbox("Signal Status", sorted(df["signal_status"].dropna().unique()))
            day_of_week = st.selectbox("Day of Week",
                                        ["Monday", "Tuesday", "Wednesday", "Thursday",
                                         "Friday", "Saturday", "Sunday"])
            vehicle_age_years = st.slider("Vehicle Age (years)", 0, 20, 5)
            prior_violations = st.slider("Prior Violations", 0, 10, 1)

        if st.button("Predict Risk Level", type="primary"):
            input_df = pd.DataFrame([{
                "speed_kmph": speed_kmph,
                "speed_limit_kmph": speed_limit_kmph,
                "vehicle_age_years": vehicle_age_years,
                "prior_violations": prior_violations,
                "hour": hour,
                "vehicle_type": vehicle_type,
                "violation_type": violation_type,
                "weather": weather,
                "signal_status": signal_status,
                "day_of_week": day_of_week,
                "location": location,
            }])
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            classes = model.named_steps["clf"].classes_

            color = {"Low": "green", "Medium": "orange", "High": "red"}[pred]
            st.markdown(f"### Predicted Risk Level: :{color}[{pred}]")

            proba_df = pd.DataFrame({"risk_level": classes, "probability": proba})
            fig = px.bar(proba_df, x="risk_level", y="probability", color="risk_level",
                         color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
                         range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Model Performance (held-out test set)")
    col1, col2 = st.columns(2)
    cm_path = os.path.join(ROOT, "outputs", "plots", "confusion_matrix.png")
    fi_path = os.path.join(ROOT, "outputs", "plots", "feature_importance.png")
    if os.path.exists(cm_path):
        col1.image(cm_path, caption="Confusion Matrix")
    if os.path.exists(fi_path):
        col2.image(fi_path, caption="Feature Importance")

st.markdown("---")
st.caption("Smart Traffic Violation Analytics System — Final Year Project | "
           "Computer Vision + ML + Data Science + Dashboard")
