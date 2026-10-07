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


def safe_read_csv_or_zip(file_path, header="infer"):
    """Probeert een bestand veilig in te lezen, of het nu een ZIP of CSV is."""
    if not os.path.exists(file_path):
        return None

    # Check of het bestand een echt ZIP-bestand is
    if zipfile.is_zipfile(file_path):
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                csv_files = [
                    f
                    for f in z.namelist()
                    if f.endswith(".csv") and not f.startswith("__MACOSX")
                ]
                if csv_files:
                    return pd.read_csv(z.open(csv_files[0]), header=header)
        except Exception:
            pass

    # Direct inlezen met Pandas (voor normale CSV's of direct door Pandas ondersteunde formaten)
    try:
        return pd.read_csv(file_path, header=header)
    except Exception:
        return None


# ==========================================
# 3. DATA LOADING & PREPROCESSING PIPELINE
# ==========================================
@st.cache_data
def load_and_clean_data():
    # A. Schedule Airport Data
    df_sched = None
    candidate_paths = [
        "schedule_airport.zip",
        "schedule_airport.csv.zip",
        "data/schedule_airport.zip",
        "schedule_airport.csv",
        "data/schedule_airport.csv",
    ]

    for path in candidate_paths:
        df_sched = safe_read_csv_or_zip(path)
        if df_sched is not None and not df_sched.empty:
            break

    if df_sched is None or df_sched.empty:
        st.error(
            "⚠️ Kan 'schedule_airport' data niet inlezen. "
            "Controleer of het bestand aanwezig is op GitHub/Streamlit Cloud en geen beschadigde Git LFS pointer is."
        )
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

    # B. Airports Extended Data
    df_airports = None
    airport_paths = [
        "airports-extended.csv.zip",
        "airports-extended.csv",
        "data/airports-extended.csv.zip",
        "data/airports-extended.csv",
    ]

    for path in airport_paths:
        df_airports = safe_read_csv_or_zip(path, header=None)
        if df_airports is not None and not df_airports.empty:
            break

    if df_airports is not None and not df_airports.empty:
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
        df_airports = pd.DataFrame()
        df_merged = df_sched
        df_merged["Afstand_km"] = np.nan

    # C. KNMI Weather Data
    df_weather = pd.DataFrame()
    weather_paths = ["06670.csv.gz", "06670.csv", "data/06670.csv.gz"]
    for wpath in weather_paths:
        if os.path.exists(wpath):
            try:
                if wpath.endswith(".gz"):
                    with gzip.open(wpath, "rt") as f:
                        df_weather = pd.read_csv(f, header=None)
                else:
                    df_weather = pd.read_csv(wpath, header=None)
                break
            except Exception:
                continue

    if not df_weather.empty:
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
            df_weather[["Weather_Date", "Temperatuur_C", "Windsnelheid_m_s"]],
            left_on="STD_DT",
            right_on="Weather_Date",
            how="left",
        )

    return df_merged, df_airports, df_weather


@st.cache_data
def load_telemetry_file(filename):
    zip_paths = ["flightdata.zip", "data/flightdata.zip"]
    for zpath in zip_paths:
        if os.path.exists(zpath) and zipfile.is_zipfile(zpath):
            try:
                with zipfile.ZipFile(zpath, "r") as z:
                    if filename in z.namelist():
                        return pd.read_excel(z.open(filename))
            except Exception:
                pass
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

# --- TAB 2: 3D KAART & AFSTANDEN ---
with tab2:
    st.subheader("🗺️ Geografische Spreiding & Berekende Afstanden")
    st.markdown(
        "De afstanden worden berekend met behulp van de **Haversine formule** tussen Schiphol en buitenlandse luchthavens."
    )

    df_map = filtered_df.dropna(subset=["Latitude", "Longitude"]).copy()

    if not df_map.empty:
        df_geo = (
            df_map.groupby(
                [
                    "Org/Des",
                    "Name",
                    "City",
                    "Country",
                    "Latitude",
                    "Longitude",
                    "Afstand_km",
                ]
            )
            .agg(
                Aantal_Vluchten=("FLT", "count"),
                Gem_Vertraging=("Vertraging_Minuten", "mean"),
            )
            .reset_index()
        )

        view_state = pdk.ViewState(
            latitude=52.3105, longitude=4.7683, zoom=3, pitch=40
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_geo,
            get_position="[Longitude, Latitude]",
            get_color="[220, 50, 30, 180]",
            get_radius="Aantal_Vluchten * 150 + 20000",
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "text": "Luchthaven: {Name} ({Org/Des})\nStad/Land: {City}, {Country}\nBerekende Afstand: {Afstand_km:.0f} km\nAantal Vluchten: {Aantal_Vluchten}\nGem. Vertraging: {Gem_Vertraging:.1f} min"
                },
            )
        )

        fig_dist = px.scatter(
            df_geo,
            x="Afstand_km",
            y="Gem_Vertraging",
            size="Aantal_Vluchten",
            color="Country",
            hover_name="Name",
            title="Relatie tussen Vluchtafstand (km) en Vertraging (minuten)",
            trendline="ols",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# --- TAB 3: WEERINVLOED (KNMI) ---
with tab3:
    st.subheader("🌤️ Invloed van KNMI Weergegevens op Vertragingen")
    st.markdown(
        "Analyse van dagelijkse temperatuur en windsnelheid gemeten op Schiphol (`06670.csv.gz`)."
    )

    if (
        "Temperatuur_C" in filtered_df.columns
        and "Windsnelheid_m_s" in filtered_df.columns
    ):
        df_w_daily = (
            filtered_df.groupby("STD_DT")
            .agg(
                Gem_Vertraging=("Vertraging_Minuten", "mean"),
                Temperatuur=("Temperatuur_C", "first"),
                Windsnelheid=("Windsnelheid_m_s", "first"),
            )
            .dropna()
            .reset_index()
        )

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            fig_w1 = px.scatter(
                df_w_daily,
                x="Windsnelheid",
                y="Gem_Vertraging",
                title="Windsnelheid (m/s) vs. Dagelijkse Vertraging",
                trendline="ols",
                labels={
                    "Windsnelheid": "Windsnelheid (m/s)",
                    "Gem_Vertraging": "Gem. Vertraging (min)",
                },
            )
            st.plotly_chart(fig_w1, use_container_width=True)

        with col_w2:
            fig_w2 = px.scatter(
                df_w_daily,
                x="Temperatuur",
                y="Gem_Vertraging",
                title="Temperatuur (°C) vs. Dagelijkse Vertraging",
                trendline="ols",
                labels={
                    "Temperatuur": "Temperatuur (°C)",
                    "Gem_Vertraging": "Gem. Vertraging (min)",
                },
            )
            st.plotly_chart(fig_w2, use_container_width=True)

# --- TAB 4: MACHINE LEARNING MODEL ---
with tab4:
    st.subheader("🤖 Machine Learning Vertraging Voorspellen")
    st.markdown(
        "Een **Scikit-Learn Gradient Boosting** model getraind op de werkelijke historische vlucht- en weerdata."
    )

    @st.cache_resource
    def train_ml_model(data):
        ml_data = (
            data[
                [
                    "Vertraging_Minuten",
                    "LSV",
                    "ACT",
                    "RWY",
                    "Afstand_km",
                    "Temperatuur_C",
                    "Windsnelheid_m_s",
                    "Uur",
                ]
            ]
            .dropna()
            .copy()
        )
        if len(ml_data) > 40000:
            ml_data = ml_data.sample(n=40000, random_state=42)

        cat_cols = ["LSV", "ACT", "RWY"]
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        ml_data[cat_cols] = encoder.fit_transform(ml_data[cat_cols].astype(str))

        X = ml_data.drop("Vertraging_Minuten", axis=1)
        y = ml_data["Vertraging_Minuten"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        return model, encoder, mae, rmse, r2, list(X.columns)

    if not df.empty:
        with st.spinner(
            "Model aan het trainen op echte vlucht- en weerdata..."
        ):
            model, encoder, mae, rmse, r2, feature_names = train_ml_model(df)

        m1, m2, m3 = st.columns(3)
        m1.metric("Model MAE (Mean Absolute Error)", f"{mae:.2f} min")
        m2.metric("Model RMSE (Root Mean Sq. Error)", f"{rmse:.2f} min")
        m3.metric("Model R² Score", f"{r2:.4f}")

        st.divider()
        st.markdown("#### 🔮 Doe een Interactieve Voorspelling")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            in_lsv = st.selectbox(
                "Vluchttype", ["L (Inbound)", "S (Outbound)"]
            )
            in_act = st.selectbox(
                "Vliegtuigtype",
                sorted([str(x) for x in df["ACT"].dropna().unique()]),
            )
            in_uur = st.slider("Vertrekuur", 0, 23, 14)
        with col_p2:
            in_rwy = st.selectbox(
                "Landings/Startbaan",
                sorted([str(x) for x in df["RWY"].dropna().unique()]),
            )
            in_dist = st.number_input(
                "Afstand tot Bestemming (km)", 100, 15000, 1500
            )
        with col_p3:
            in_wind = st.slider("Verwachte Windsnelheid (m/s)", 0, 30, 8)
            in_temp = st.slider("Verwachte Temperatuur (°C)", -10, 35, 18)

        if st.button("🚀 Bereken Voorspelde Vertraging"):
            lsv_code = "L" if "Inbound" in in_lsv else "S"
            input_row = pd.DataFrame(
                [
                    [
                        lsv_code,
                        in_act,
                        in_rwy,
                        in_dist,
                        in_temp,
                        in_wind,
                        in_uur,
                    ]
                ],
                columns=feature_names,
            )
            input_row[["LSV", "ACT", "RWY"]] = encoder.transform(
                input_row[["LSV", "ACT", "RWY"]].astype(str)
            )

            pred_delay = model.predict(input_row)[0]
            st.success(
                f"Voorspelde Vertraging: **{pred_delay:.1f} minuten**"
            )

# --- TAB 5: TELEMETRIE & DATA INSPECTIE ---
with tab5:
    st.subheader("🛫 Vluchttelemetrie & Data Inspectie")

    zip_paths = ["flightdata.zip", "data/flightdata.zip"]
    found_telemetry = False
    for zpath in zip_paths:
        if os.path.exists(zpath) and zipfile.is_zipfile(zpath):
            st.markdown("#### Vluchttelemetrie Analyse")
            try:
                with zipfile.ZipFile(zpath, "r") as z:
                    f_list = sorted([f for f in z.namelist() if f.endswith(".xlsx")])

                sel_f = st.selectbox("Selecteer Telemetrie Bestand", f_list)
                if sel_f:
                    df_tel = load_telemetry_file(sel_f)
                    st.write(f"Datapunten in gekozen vlucht: **{len(df_tel):,}**")

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        if "[3d Altitude Ft]" in df_tel.columns:
                            fig_alt = px.line(
                                df_tel,
                                x="Time (secs)",
                                y="[3d Altitude Ft]",
                                title="Hoogteverloop (Feet) over Tijd",
                            )
                            st.plotly_chart(fig_alt, use_container_width=True)
                    with col_f2:
                        if "TRUE AIRSPEED (derived)" in df_tel.columns:
                            fig_sp = px.line(
                                df_tel,
                                x="Time (secs)",
                                y="TRUE AIRSPEED (derived)",
                                title="Luchtsnelheid over Tijd",
                            )
                            st.plotly_chart(fig_sp, use_container_width=True)
                found_telemetry = True
                break
            except Exception:
                pass

    st.divider()
    st.markdown("#### 📋 Opschonde Dataset Inspectie & Exporteren")
    st.dataframe(filtered_df.head(500), use_container_width=True)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Gefilterde Dataset als CSV",
        data=csv_bytes,
        file_name="opschonde_schiphol_vluchtdata.csv",
        mime="text/csv",
    )