import gzip
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Schiphol Flight & Delay Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈️ Schiphol Vlucht, Weer & Vertraging Dashboard")
st.markdown(
    """*Een integraal dataplaatje waarin vluchtdata, KNMI-weeromstandigheden, """
    """wereldwijde luchthavencoördinaten en vluchttelemetrie samenkomen.*"""
)


# ==========================================
# 2. HELPER FUNCTIONS & HAVERSINE DISTANCE
# ==========================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Berekent de afstand in kilometers tussen twee geografische coördinaten."""
    R = 6371.0  # Straal van de aarde in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# Coördinaten Schiphol (AMS / EHAM)
AMS_LAT, AMS_LON = 52.3105, 4.7683


# ==========================================
# 3. DATA LOADING & PREPROCESSING PIPELINE
# ==========================================
@st.cache_data
def load_and_clean_data():
    # A. Schedule Airport Data (Eerst ZIP proberen)
    df_sched = pd.DataFrame()
    zip_candidates = [
        "schedule_airport.zip",
        "schedule_airport.csv.zip",
        "data/schedule_airport.zip",
    ]
    csv_candidates = ["schedule_airport.csv", "data/schedule_airport.csv"]

    loaded = False
    for zpath in zip_candidates:
        if os.path.exists(zpath):
            with zipfile.ZipFile(zpath, "r") as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if csv_files:
                    df_sched = pd.read_csv(z.open(csv_files[0]))
                    loaded = True
                    break

    if not loaded:
        for cpath in csv_candidates:
            if os.path.exists(cpath):
                df_sched = pd.read_csv(cpath)
                loaded = True
                break

    if not loaded or df_sched.empty:
        st.error("Bestand 'schedule_airport.zip' niet gevonden!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Datum en Tijd Conversies
    df_sched["STD_DT"] = pd.to_datetime(
        df_sched["STD"], format="%d/%m/%Y", errors="coerce"
    )
    df_sched["Sched_Time"] = pd.to_datetime(
        df_sched["STD"] + " " + df_sched["STA_STD_ltc"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df_sched["Act_Time"] = pd.to_datetime(
        df_sched["STD"] + " " + df_sched["ATA_ATD_ltc"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    # Vertraging berekenen in minuten + opschonen midnight rollover
    df_sched["Vertraging_Minuten"] = (
        df_sched["Act_Time"] - df_sched["Sched_Time"]
    ).dt.total_seconds() / 60.0
    df_sched.loc[df_sched["Vertraging_Minuten"] < -720, "Vertraging_Minuten"] += (
        1440
    )
    df_sched.loc[df_sched["Vertraging_Minuten"] > 720, "Vertraging_Minuten"] -= (
        1440
    )

    df_sched["Is_Vertraagd"] = df_sched["Vertraging_Minuten"] > 15
    df_sched["Uur"] = df_sched["Sched_Time"].dt.hour
    df_sched["Maand"] = df_sched["Sched_Time"].dt.strftime("%Y-%m")
    df_sched["DagVanWeek"] = df_sched["Sched_Time"].dt.day_name()
    df_sched["ICAO_clean"] = (
        df_sched["Org/Des"].astype(str).str.strip().str.upper()
    )

    # B. Airports Extended Data (Eerst ZIP proberen)
    df_airports = pd.DataFrame()
    airports_zip = "airports-extended.csv.zip"
    airports_csv = "airports-extended.csv"

    if os.path.exists(airports_zip):
        with zipfile.ZipFile(airports_zip, "r") as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if csv_files:
                df_airports = pd.read_csv(z.open(csv_files[0]), header=None)
    elif os.path.exists(airports_csv):
        df_airports = pd.read_csv(airports_csv, header=None)

    if not df_airports.empty:
        df_airports.columns = [
            "Airport_ID",
            "Name",
            "City",
            "Country",
            "IATA",
            "ICAO",
            "Latitude",
            "Longitude",
            "Altitude",
            "Timezone",
            "DST",
            "Tz_Database",
            "Type",
            "Source",
        ]
        df_airports["ICAO_clean"] = (
            df_airports["ICAO"].astype(str).str.strip().str.upper()
        )

        # Merge met luchthavendata
        df_merged = pd.merge(
            df_sched,
            df_airports[
                [
                    "ICAO_clean",
                    "Name",
                    "City",
                    "Country",
                    "Latitude",
                    "Longitude",
                ]
            ],
            on="ICAO_clean",
            how="left",
        )
        # Afstand berekenen tot Schiphol
        df_merged["Afstand_km"] = haversine_distance(
            AMS_LAT, AMS_LON, df_merged["Latitude"], df_merged["Longitude"]
        )
    else:
        df_merged = df_sched
        df_merged["Afstand_km"] = np.nan

    # C. KNMI Weather Data
    weather_path = "06670.csv.gz"
    df_weather = pd.DataFrame()
    if os.path.exists(weather_path):
        with gzip.open(weather_path, "rt") as f:
            df_weather = pd.read_csv(f, header=None)
            cols = [
                "Datum_Str",
                "TG",
                "TN",
                "TX",
                "SQ",
                "SP",
                "DD",
                "FH",
                "FF",
                "PX",
                "VN",
            ]
            df_weather = df_weather.iloc[:, : len(cols)]
            df_weather.columns = cols[: df_weather.shape[1]]

            df_weather["Weather_Date"] = pd.to_datetime(
                df_weather["Datum_Str"], errors="coerce"
            )
            df_weather["Temperatuur_C"] = (
                df_weather["TG"] / 10.0
            )  # KNMI is in 0.1 C
            df_weather["Windsnelheid_m_s"] = (
                df_weather["FH"] / 10.0
            )  # KNMI wind is in 0.1 m/s

            # Koppelen op datum
            df_merged = pd.merge(
                df_merged,
                df_weather[
                    ["Weather_Date", "Temperatuur_C", "Windsnelheid_m_s"]
                ],
                left_on="STD_DT",
                right_on="Weather_Date",
                how="left",
            )

    return df_merged, df_airports, df_weather


@st.cache_data
def load_telemetry_file(filename):
    zip_path = "flightdata.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            if filename in z.namelist():
                return pd.read_excel(z.open(filename))
    return pd.DataFrame()


df, df_airports, df_weather = load_and_clean_data()

# ==========================================
# 4. SIDEBAR INTERACTIVE FILTERS
# ==========================================
st.sidebar.header("🔍 Interactive Filters")

if not df.empty and "STD_DT" in df.columns:
    min_d, max_d = df["STD_DT"].min().date(), df["STD_DT"].max().date()
    date_range = st.sidebar.date_input("Selecteer Datum Range", [min_d, max_d])

    type_filter = st.sidebar.radio(
        "Vluchttype (LSV)",
        ["Alles", "Inbound (L - Landend)", "Outbound (S - Vertrekkend)"],
    )

    act_list = ["Alles"] + sorted([x for x in df["ACT"].dropna().unique()])
    act_filter = st.sidebar.selectbox("Vliegtuigtype (ACT)", act_list)

    rwy_list = ["Alles"] + sorted(
        [str(x) for x in df["RWY"].dropna().unique()]
    )
    rwy_filter = st.sidebar.selectbox("Start-/Landingsbaan (RWY)", rwy_list)

    min_delay = st.sidebar.slider(
        "Minimale Vertraging Filter (minuten)", -30, 180, 0
    )

    # Toepassen van filters
    filtered_df = df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["STD_DT"].dt.date >= date_range[0])
            & (filtered_df["STD_DT"].dt.date <= date_range[1])
        ]
    if type_filter != "Alles":
        code = "L" if "Inbound" in type_filter else "S"
        filtered_df = filtered_df[filtered_df["LSV"] == code]
    if act_filter != "Alles":
        filtered_df = filtered_df[filtered_df["ACT"] == act_filter]
    if rwy_filter != "Alles":
        filtered_df = filtered_df[filtered_df["RWY"].astype(str) == rwy_filter]

    filtered_df = filtered_df[filtered_df["Vertraging_Minuten"] >= min_delay]
else:
    filtered_df = df

# ==========================================
# 5. EXECUTIVE KPI METRICS
# ==========================================
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        label="📊 Geanalyseerde Vluchten", value=f"{len(filtered_df):,}"
    )
with k2:
    avg_d = (
        filtered_df["Vertraging_Minuten"].mean()
        if not filtered_df.empty
        else 0
    )
    st.metric(label="⏱️ Gem. Vertraging", value=f"{avg_d:.1f} min")
with k3:
    pct_d = (
        (filtered_df["Is_Vertraagd"].sum() / len(filtered_df) * 100)
        if len(filtered_df) > 0
        else 0
    )
    st.metric(label="🚨 Vertraagd (>15m)", value=f"{pct_d:.1f}%")
with k4:
    avg_dist = (
        filtered_df["Afstand_km"].mean() if not filtered_df.empty else 0
    )
    st.metric(label="🌐 Gem. Vluchtafstand", value=f"{avg_dist:.0f} km")
with k5:
    top_dest = (
        filtered_df["Org/Des"].mode()[0]
        if not filtered_df.empty and "Org/Des" in filtered_df.columns
        else "N/A"
    )
    st.metric(label="📍 Drukste Bestemming", value=top_dest)

st.divider()

# ==========================================
# 6. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Tijdsanalyse & Trends",
        "🗺️ 3D Kaart & Afstanden",
        "🌤️ Weerinvloed (KNMI)",
        "🤖 Machine Learning Model",
        "🛫 Vluchttelemetrie & Data Inspectie",
    ]
)

# --- TAB 1: TIJDSANALYSE ---
with tab1:
    st.subheader("📈 Verloop en Trends van Vertragingen")

    if not filtered_df.empty:
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            df_m = (
                filtered_df.groupby(["Maand", "LSV"])["Vertraging_Minuten"]
                .mean()
                .reset_index()
            )
            df_m["LSV_Name"] = df_m["LSV"].map(
                {"L": "Inbound (L)", "S": "Outbound (S)"}
            )
            fig_m = px.line(
                df_m,
                x="Maand",
                y="Vertraging_Minuten",
                color="LSV_Name",
                title="Gemiddelde Vertraging per Maand (Inbound vs Outbound)",
                markers=True,
                labels={
                    "Vertraging_Minuten": "Gemiddelde Vertraging (min)",
                    "Maand": "Maand",
                },
            )
            st.plotly_chart(fig_m, use_container_width=True)

        with col_t2:
            df_u = (
                filtered_df.groupby("Uur")["Vertraging_Minuten"]
                .mean()
                .reset_index()
            )
            fig_u = px.bar(
                df_u,
                x="Uur",
                y="Vertraging_Minuten",
                color="Vertraging_Minuten",
                title="Gemiddelde Vertraging per Uur van de Dag (Piekuren)",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_u, use_container_width=True)

        col_t3, col_t4 = st.columns(2)
        with col_t3:
            fig_rwy = px.box(
                filtered_df.dropna(subset=["RWY"]),
                x="RWY",
                y="Vertraging_Minuten",
                title="Spreiding van Vertraging per Start-/Landingsbaan",
                color="RWY",
            )
            st.plotly_chart(fig_rwy, use_container_width=True)

        with col_t4:
            df_dow = (
                filtered_df.groupby("DagVanWeek")["Vertraging_Minuten"]
                .mean()
                .reindex(
                    [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]
                )
                .reset_index()
            )
            fig_dow = px.bar(
                df_dow,
                x="DagVanWeek",
                y="Vertraging_Minuten",
                title="Gemiddelde Vertraging per Dag van de Week",
            )
            st.plotly_chart(fig_dow, use_container_width=True)

