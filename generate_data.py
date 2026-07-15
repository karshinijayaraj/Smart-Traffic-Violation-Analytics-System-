"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC traffic-violation dataset for the
Smart Traffic Violation Analytics System.

In a real deployment this table would be populated automatically by:
  - src/detection.py (computer-vision pipeline reading live/CCTV feeds)
  - ANPR (Automatic Number Plate Recognition) events
  - Traffic-signal controller logs

Since no live camera feed is available in this environment, this script
creates a statistically realistic dataset (seasonality, rush-hour peaks,
location hot-spots, vehicle-type mix, weather effects) so every other
component (ML model, EDA, dashboard) can be fully built and tested
end-to-end. Swap this CSV for real pipeline output in production.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_RECORDS = 6000

LOCATIONS = [
    ("MG Road Junction", 9.9816, 76.2999),
    ("Marine Drive Signal", 9.9750, 76.2790),
    ("Kaloor Flyover", 10.0016, 76.2999),
    ("Vytilla Junction", 9.9668, 76.3193),
    ("Palarivattom Junction", 10.0028, 76.3086),
    ("Edappally Toll", 10.0270, 76.3082),
    ("Aluva Bypass", 10.1076, 76.3516),
    ("Thrippunithura Signal", 9.9450, 76.3450),
]

VIOLATION_TYPES = [
    "Overspeeding",
    "Red Light Jump",
    "Wrong Lane / Wrong Side",
    "No Helmet",
    "Signal Jump + No Helmet",
    "Illegal Parking",
    "Mobile Phone Usage",
    "Triple Riding",
]

VEHICLE_TYPES = ["Car", "Motorcycle", "Bus", "Truck", "Auto-rickshaw"]
WEATHER = ["Clear", "Rain", "Fog", "Overcast"]

SPEED_LIMITS = {"Car": 60, "Motorcycle": 60, "Bus": 50, "Truck": 50, "Auto-rickshaw": 45}

FINE_TABLE = {
    "Overspeeding": (1000, 2000),
    "Red Light Jump": (1000, 5000),
    "Wrong Lane / Wrong Side": (500, 1500),
    "No Helmet": (500, 1000),
    "Signal Jump + No Helmet": (1500, 5500),
    "Illegal Parking": (300, 1000),
    "Mobile Phone Usage": (1000, 5000),
    "Triple Riding": (1000, 2000),
}


def random_timestamp():
    """Generate timestamps with rush-hour (8-10am, 5-8pm) bias over the last 180 days."""
    day_offset = RNG.integers(0, 180)
    base_date = datetime(2026, 1, 1) + timedelta(days=int(day_offset))

    # 60% chance of rush hour, 40% uniform across the day
    if RNG.random() < 0.6:
        hour = int(RNG.choice([8, 9, 17, 18, 19]))
    else:
        hour = int(RNG.integers(0, 24))
    minute = int(RNG.integers(0, 60))
    second = int(RNG.integers(0, 60))
    return base_date.replace(hour=hour, minute=minute, second=second)


def generate_row(i):
    loc_name, lat, lon = LOCATIONS[RNG.integers(0, len(LOCATIONS))]
    # small jitter so points don't all overlap on the map
    lat_j = lat + RNG.normal(0, 0.002)
    lon_j = lon + RNG.normal(0, 0.002)

    vehicle_type = RNG.choice(VEHICLE_TYPES, p=[0.38, 0.33, 0.08, 0.09, 0.12])
    violation_type = RNG.choice(
        VIOLATION_TYPES, p=[0.28, 0.18, 0.12, 0.16, 0.06, 0.08, 0.07, 0.05]
    )
    weather = RNG.choice(WEATHER, p=[0.6, 0.2, 0.1, 0.1])
    ts = random_timestamp()

    speed_limit = SPEED_LIMITS[vehicle_type]
    if violation_type in ("Overspeeding", "Signal Jump + No Helmet"):
        speed = speed_limit + abs(RNG.normal(20, 12))
    else:
        speed = max(5, RNG.normal(speed_limit * 0.75, 10))
    speed = round(float(np.clip(speed, 5, 160)), 1)

    signal_status = "Red" if "Red Light" in violation_type or "Signal Jump" in violation_type else RNG.choice(
        ["Green", "Yellow", "N/A"]
    )

    fine_lo, fine_hi = FINE_TABLE[violation_type]
    fine_amount = int(RNG.integers(fine_lo, fine_hi + 1))

    vehicle_age_years = int(RNG.integers(0, 18))
    prior_violations = int(RNG.poisson(1.2))
    is_repeat_offender = prior_violations >= 2

    # penalty payment behaviour (target-ish, correlated with fine size & repeat offences)
    pay_prob = 0.85 - 0.05 * is_repeat_offender - (fine_amount / 20000)
    fine_paid = RNG.random() < np.clip(pay_prob, 0.15, 0.95)

    plate = f"KL{RNG.integers(1,15):02d}{chr(65+RNG.integers(0,26))}{chr(65+RNG.integers(0,26))}{RNG.integers(1000,9999)}"

    return {
        "violation_id": f"TV{100000+i}",
        "timestamp": ts,
        "date": ts.date(),
        "hour": ts.hour,
        "day_of_week": ts.strftime("%A"),
        "location": loc_name,
        "latitude": round(lat_j, 6),
        "longitude": round(lon_j, 6),
        "vehicle_type": vehicle_type,
        "vehicle_plate": plate,
        "vehicle_age_years": vehicle_age_years,
        "violation_type": violation_type,
        "speed_kmph": speed,
        "speed_limit_kmph": speed_limit,
        "signal_status": signal_status,
        "weather": weather,
        "prior_violations": prior_violations,
        "is_repeat_offender": is_repeat_offender,
        "fine_amount_inr": fine_amount,
        "fine_paid": fine_paid,
    }


def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a 'risk_level' label used as the ML target.
    Rule-of-thumb scoring (mirrors how a traffic authority might triage cases),
    with noise added so the classification task isn't trivially linear.
    """
    score = np.zeros(len(df))
    score += (df["speed_kmph"] - df["speed_limit_kmph"]).clip(lower=0) * 0.6
    score += df["is_repeat_offender"].astype(int) * 15
    score += df["prior_violations"] * 4
    score += (df["violation_type"].isin(["Red Light Jump", "Signal Jump + No Helmet"])) * 12
    score += (df["weather"].isin(["Rain", "Fog"])) * 5
    score += RNG.normal(0, 6, size=len(df))

    q1, q2 = np.quantile(score, [0.6, 0.85])
    risk = np.where(score >= q2, "High", np.where(score >= q1, "Medium", "Low"))
    df["risk_score_raw"] = score.round(2)
    df["risk_level"] = risk
    return df


def main():
    rows = [generate_row(i) for i in range(N_RECORDS)]
    df = pd.DataFrame(rows)
    df = add_risk_label(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = "data/traffic_violations.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records -> {out_path}")
    print(df["risk_level"].value_counts())


if __name__ == "__main__":
    main()
