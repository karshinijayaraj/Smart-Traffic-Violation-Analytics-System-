"""
eda.py
------
Exploratory Data Analysis for the traffic violations dataset.
Generates a set of PNG charts used both standalone and inside the
Streamlit dashboard.

Run: python3 src/eda.py
Outputs -> outputs/plots/*.png
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
DATA_PATH = "data/traffic_violations.csv"
PLOT_DIR = "outputs/plots"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    # 1. Violations by hour of day
    plt.figure(figsize=(8, 4.5))
    sns.countplot(data=df, x="hour", color="#3b82f6")
    plt.title("Violations by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/violations_by_hour.png", dpi=150)
    plt.close()

    # 2. Violations by type
    plt.figure(figsize=(8, 5))
    order = df["violation_type"].value_counts().index
    sns.countplot(data=df, y="violation_type", order=order, color="#ef4444")
    plt.title("Violations by Type")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/violations_by_type.png", dpi=150)
    plt.close()

    # 3. Violations by location (hotspots)
    plt.figure(figsize=(8, 5))
    order = df["location"].value_counts().index
    sns.countplot(data=df, y="location", order=order, color="#f59e0b")
    plt.title("Violation Hotspots by Location")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/violations_by_location.png", dpi=150)
    plt.close()

    # 4. Daily trend
    daily = df.groupby("date").size()
    plt.figure(figsize=(9, 4))
    daily.plot()
    plt.title("Daily Violation Trend")
    plt.xlabel("Date")
    plt.ylabel("Violations")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/daily_trend.png", dpi=150)
    plt.close()

    # 5. Speed distribution vs speed limit
    plt.figure(figsize=(8, 4.5))
    sns.histplot(data=df, x="speed_kmph", hue="risk_level", bins=40, multiple="stack")
    plt.title("Speed Distribution by Risk Level")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/speed_distribution.png", dpi=150)
    plt.close()

    # 6. Vehicle type mix
    plt.figure(figsize=(6, 6))
    df["vehicle_type"].value_counts().plot.pie(autopct="%1.1f%%", ylabel="")
    plt.title("Violations by Vehicle Type")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/vehicle_type_mix.png", dpi=150)
    plt.close()

    # 7. Fine collection: paid vs unpaid, by risk level
    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x="risk_level", hue="fine_paid",
                  order=["Low", "Medium", "High"], palette=["#ef4444", "#22c55e"])
    plt.title("Fine Payment Status by Risk Level")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fine_payment_status.png", dpi=150)
    plt.close()

    # 8. Correlation heatmap (numeric features)
    numeric_cols = ["speed_kmph", "speed_limit_kmph", "vehicle_age_years",
                     "prior_violations", "fine_amount_inr", "hour"]
    plt.figure(figsize=(7, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    print("EDA charts saved to outputs/plots/")
    for f in sorted(pd.Series(
        ["violations_by_hour.png", "violations_by_type.png", "violations_by_location.png",
         "daily_trend.png", "speed_distribution.png", "vehicle_type_mix.png",
         "fine_payment_status.png", "correlation_heatmap.png"])):
        print(f" - {f}")


if __name__ == "__main__":
    main()
