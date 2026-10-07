import math
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PAGINA CONFIGURATIE & STYLING
# ==========================================
st.set_page_config(
    page_title="Schiphol Vlucht, Weer & Vertraging Dashboard",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="expanded",
)

# Custom CSS voor Schiphol Dashboard styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #002244;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        font-style: italic;
        color: #555555;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #0055a5;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("✈️ Schiphol Vlucht, Weer & Vertraging Dashboard")
st.write(
    "Een integraal dataplaatje waarin vluchtdata, KNMI-weeromstandigheden, "
    "wereldwijde luchthavencoördinaten en vluchttelemetrie samenkomen."
)


# ==========================================
# 2. HELPER FUNCTIES (HAVERSINE & DATA CLEANING)
# ==========================================
def haversine_distance(lat1, lon1, lat2=52.3105, lon2=4.7683):
    """Berekent de afstand in kilometers vanaf Schiphol (52.3105, 4.7683)."""
    try:
        r = 6371.0  # Aardstraal in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
    except Exception:
        return np.nan


# ==========================================
# 3. ROBUUST DATA INLADEN (.ZIP SUPPORT)
# ==========================================
@st.cache_data
def load_data():
    candidate_files = [
        "schedule_airport.zip",
        "data/schedule_airport.zip",
        "schedule_airport.csv.gz",
        "data/schedule_airport.csv.gz",
        "schedule_airport.csv",
        "data/schedule_airport.csv",
        "schedule_airport.parquet",
    ]

    target_file = None
    for f in candidate_files:
        if os.path.exists(f):
            target_file = f
            break

    if target_file is None:
        return None, "Bestand 'schedule_airport.zip' niet gevonden!"

    try:
        # Pandas leest een zip met 1 CSV-bestand direct uit
        if target_file.endswith(".parquet"):
            df = pd.read_parquet(target_file)
        else:
            df = pd.read_csv(target_file)

        # Hulpfunctie om kolommen te hernoemen naar standaard namen
        rename_dict = {}
        for col in df.columns:
            c_low = col.strip().lower()
            if c_low in ["latitude", "lat", "breedtegraad"]:
                rename_dict[col] = "Latitude"
            elif c_low in ["longitude", "lon", "lng", "lengtegraad"]:
                rename_dict[col] = "Longitude"
            elif c_low in [
                "delay",
                "delay_minutes",
                "vertraging",
                "vertraging_min",
            ]:
                rename_dict[col] = "Delay"
            elif c_low in ["destination", "bestemming", "dest"]:
                rename_dict[col] = "Destination"
            elif c_low in ["distance", "distance_km", "afstand"]:
                rename_dict[col] = "Distance"
            elif c_low in ["airline", "luchtvaartmaatschappij", "carrier"]:
                rename_dict[col] = "Airline"
            elif c_low in ["date", "datum", "time", "tijd", "datetime"]:
                rename_dict[col] = "DateTime"
            elif c_low in ["wind", "wind_speed", "windsnelheid"]:
                rename_dict[col] = "Wind_Speed"
            elif c_low in ["temp", "temperature", "temperatuur"]:
                rename_dict[col] = "Temperature"
            elif c_low in ["rain", "neerslag", "precipitation"]:
                rename_dict[col] = "Rain"
            elif c_low in ["visibility", "zicht"]:
                rename_dict[col] = "Visibility"

        df = df.rename(columns=rename_dict)

        # Datetime conversie
        if "DateTime" in df.columns:
            df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
            df["Hour"] = df["DateTime"].dt.hour
            df["DayOfWeek"] = df["DateTime"].dt.day_name()

        # Afstand berekenen met Haversine als Latitude/Longitude aanwezig zijn
        if "Latitude" in df.columns and "Longitude" in df.columns:
            df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
            df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
            if "Distance" not in df.columns:
                df["Distance"] = df.apply(
                    lambda row: haversine_distance(
                        row["Latitude"], row["Longitude"]
                    ),
                    axis=1,
                )

        if "Delay" in df.columns:
            df["Delay"] = pd.to_numeric(df["Delay"], errors="coerce").fillna(0)

        return df, None
    except Exception as e:
        return None, f"Fout bij openen van data: {str(e)}"


# Laad de data
df, error_msg = load_data()

# Als bestand niet aanwezig is: toon melding en stop de uitvoering veilig
if df is None:
    st.error(f"❌ {error_msg}")
    st.info(
        "💡 **Oplossing:** Upload het gecomprimeerde bestand `schedule_airport.zip` "
        "naar de hoofdmap van je GitHub repository en herstart de app."
    )
    st.stop()  # Voorkomt KeyErrors in onderstaande regels code!


# ==========================================
# 4. INTERACTIVE FILTERS (SIDEBAR)
# ==========================================
st.sidebar.markdown("## 🔍 Interactive Filters")

filtered_df = df.copy()

# Filter: Luchtvaartmaatschappij
if "Airline" in filtered_df.columns:
    airlines = ["Alles"] + sorted(
        filtered_df["Airline"].dropna().unique().tolist()
    )
    sel_airline = st.sidebar.selectbox("Luchtvaartmaatschappij:", airlines)
    if sel_airline != "Alles":
        filtered_df = filtered_df[filtered_df["Airline"] == sel_airline]

# Filter: Bestemming
if "Destination" in filtered_df.columns:
    destinations = ["Alles"] + sorted(
        filtered_df["Destination"].dropna().unique().tolist()
    )
    sel_dest = st.sidebar.selectbox("Bestemming:", destinations)
    if sel_dest != "Alles":
        filtered_df = filtered_df[filtered_df["Destination"] == sel_dest]

# Filter: Vertragingsstatus
if "Delay" in filtered_df.columns:
    delay_option = st.sidebar.radio(
        "Vertragingsstatus:",
        ["Alle Vluchten", "Alleen Vertraagd (>15m)", "Op Tijd (≤15m)"],
    )
    if delay_option == "Alleen Vertraagd (>15m)":
        filtered_df = filtered_df[filtered_df["Delay"] > 15]
    elif delay_option == "Op Tijd (≤15m)":
        filtered_df = filtered_df[filtered_df["Delay"] <= 15]


# ==========================================
# 5. METRICS / KPI BEREKENINGEN
# ==========================================
totaal_vluchten = len(filtered_df)
avg_delay = (
    filtered_df["Delay"].mean() if "Delay" in filtered_df.columns else 0.0
)
delayed_pct = (
    (filtered_df["Delay"] > 15).mean() * 100
    if "Delay" in filtered_df.columns
    else 0.0
)
avg_distance = (
    filtered_df["Distance"].mean()
    if "Distance" in filtered_df.columns
    else 0.0
)

if (
    "Destination" in filtered_df.columns
    and not filtered_df["Destination"].empty
):
    top_dest_mode = filtered_df["Destination"].mode()
    busiest_dest = top_dest_mode[0] if not top_dest_mode.empty else "N/A"
else:
    busiest_dest = "N/A"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📊 Geanalyseerde Vluchten", f"{totaal_vluchten:,}")
col2.metric(
    "⏱️ Gem. Vertraging",
    f"{avg_delay:.1f} min" if not np.isnan(avg_delay) else "0.0 min",
)
col3.metric(
    "🚨 Vertraagd (>15m)",
    f"{delayed_pct:.1f}%" if not np.isnan(delayed_pct) else "0.0%",
)
col4.metric(
    "🌐 Gem. Vluchtafstand",
    f"{int(avg_distance)} km" if not np.isnan(avg_distance) else "0 km",
)
col5.metric("📍 Drukste Bestemming", str(busiest_dest))

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
        "📡 Vluchttelemetrie & Data Inspectie",
    ]
)

# ------------------------------------------
# TAB 1: TIJDSANALYSE & TRENDS
# ------------------------------------------
with tab1:
    st.subheader("📈 Verloop en Trends van Vertragingen")

    if "Delay" in filtered_df.columns:
        c1, c2 = st.columns(2)

        with c1:
            fig_hist = px.histogram(
                filtered_df,
                x="Delay",
                nbins=35,
                title="Verdeling van Vluchtvertragingen (in Minuten)",
                labels={"Delay": "Vertraging (minuten)"},
                color_discrete_sequence=["#0055a5"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            if "Destination" in filtered_df.columns:
                top_delays = (
                    filtered_df.groupby("Destination")["Delay"]
                    .mean()
                    .reset_index()
                    .sort_values(by="Delay", ascending=False)
                    .head(10)
                )

                fig_bar = px.bar(
                    top_delays,
                    x="Destination",
                    y="Delay",
                    title="Top 10 Bestemmingen met Hoogste Gemiddelde Vertraging",
                    labels={
                        "Destination": "Bestemming",
                        "Delay": "Gem. Vertraging (min)",
                    },
                    color="Delay",
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        if "Hour" in filtered_df.columns:
            st.markdown("### Gemiddelde Vertraging per Uur van de Dag")
            hourly_delay = (
                filtered_df.groupby("Hour")["Delay"].mean().reset_index()
            )
            fig_line = px.line(
                hourly_delay,
                x="Hour",
                y="Delay",
                markers=True,
                title="Vertragingsverloop over het Etmaal",
                labels={
                    "Hour": "Uur van de Dag (0-23)",
                    "Delay": "Gem. Vertraging (min)",
                },
            )
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Geen vertragingsgegevens beschikbaar voor visualisatie.")


# ------------------------------------------
# TAB 2: 3D KAART & AFSTANDEN
# ------------------------------------------
with tab2:
    st.subheader("🗺️ Geografische Spreiding & Berekende Afstanden")
    st.write(
        "De afstanden worden berekend met behulp van de **Haversine formule** "
        "tussen Schiphol (52.3105° N, 4.7683° E) en buitenlandse luchthavens."
    )

    # VEREIST VOOR REGEL 254: Veilige controle op Latitude & Longitude
    if (
        "Latitude" in filtered_df.columns
        and "Longitude" in filtered_df.columns
    ):
        df_map = filtered_df.dropna(subset=["Latitude", "Longitude"]).copy()

        if not df_map.empty:
            st.markdown("### 3D Vluchtroutes & Bestemmingen")

            # Schiphol coördinaten
            schiphol_lat, schiphol_lon = 52.3105, 4.7683

            # Maak arc data voor Pydeck 3D kaart
            arc_data = df_map.drop_duplicates(subset=["Latitude", "Longitude"])[
                ["Latitude", "Longitude", "Destination"]
            ].copy()
            arc_data["from_lat"] = schiphol_lat
            arc_data["from_lon"] = schiphol_lon

            layer_arcs = pdk.Layer(
                "ArcLayer",
                data=arc_data,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["Longitude", "Latitude"],
                get_source_color=[0, 85, 165, 160],
                get_target_color=[230, 50, 50, 160],
                get_width=2,
            )

            layer_scatter = pdk.Layer(
                "ScatterplotLayer",
                data=arc_data,
                get_position=["Longitude", "Latitude"],
                get_color=[230, 50, 50, 200],
                get_radius=30000,
                pickable=True,
            )

            view_state = pdk.ViewState(
                latitude=50.0, longitude=10.0, zoom=3, pitch=45
            )

            deck = pdk.Deck(
                layers=[layer_arcs, layer_scatter],
                initial_view_state=view_state,
                tooltip={
                    "text": "Bestemming: {Destination}\nLat: {Latitude}, Lon: {Longitude}"
                },
            )

            st.pydeck_chart(deck)

            # Standaard platte kaart
            st.markdown("### Platte Kaartweergave")
            st.map(df_map[["Latitude", "Longitude"]])
        else:
            st.warning("Geen geldige coördinaten gevonden in de gefilterde dataset.")
    else:
        st.info(
            "Kolommen 'Latitude' en 'Longitude' zijn niet aanwezig om de kaart op te bouwen."
        )


# ------------------------------------------
# TAB 3: WEERINVLOED (KNMI)
# ------------------------------------------
with tab3:
    st.subheader("🌤️ Weerinvloed op Vluchtvertragingen (KNMI)")
    st.write(
        "Analyse van de relatie tussen KNMI-weeromstandigheden en vluchtvertragingen."
    )

    weather_cols = [
        c
        for c in filtered_df.columns
        if c in ["Wind_Speed", "Temperature", "Rain", "Visibility"]
    ]

    if weather_cols and "Delay" in filtered_df.columns:
        c1, c2 = st.columns([1, 2])

        with c1:
            sel_weather = st.selectbox(
                "Kies een weer-variabele:", weather_cols
            )

            st.markdown("**Correlatie-overzicht:**")
            corr_df = (
                filtered_df[weather_cols + ["Delay"]]
                .corr()["Delay"]
                .drop("Delay")
                .reset_index()
            )
            corr_df.columns = ["Weer Variabele", "Correlatie met Vertraging"]
            st.dataframe(corr_df, use_container_width=True)

        with c2:
            fig_scatter = px.scatter(
                filtered_df,
                x=sel_weather,
                y="Delay",
                trendline="ols",
                title=f"Invloed van {sel_weather} op Vertraging",
                labels={
                    sel_weather: sel_weather,
                    "Delay": "Vertraging (minuten)",
                },
                opacity=0.5,
                color_discrete_sequence=["#e63232"],
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info(
            "KNMI-weerkolommen (zoals Wind_Speed, Temperature, Rain) of 'Delay' niet gevonden."
        )


# ------------------------------------------
# TAB 4: MACHINE LEARNING MODEL
# ------------------------------------------
with tab4:
    st.subheader("🤖 Machine Learning Model: Vertraging Voorspellen")
    st.write(
        "Voorspel de verwachte vertraging op basis van afstand, vertrektijd en weersomstandigheden."
    )

    feature_cols = [
        c
        for c in ["Distance", "Hour", "Wind_Speed", "Temperature", "Rain"]
        if c in filtered_df.columns
    ]

    if len(feature_cols) >= 2 and "Delay" in filtered_df.columns:
        ml_df = filtered_df[feature_cols + ["Delay"]].dropna()

        if len(ml_df) > 50:
            X = ml_df[feature_cols]
            y = ml_df["Delay"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X_train, y_train)

            st.success(
                f"✅ Model succesvol getraind op {len(X_train)} vluchten!"
            )

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### 🎯 Doe een Voorspelling")
                input_data = {}
                for col in feature_cols:
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    mean_val = float(X[col].mean())
                    input_data[col] = st.slider(
                        f"{col}:", min_val, max_val, mean_val
                    )

                input_df = pd.DataFrame([input_data])
                prediction = rf_model.predict(input_df)[0]
                st.markdown(
                    f"### ⏱️ Voorspelde Vertraging: **{max(0, prediction):.1f} minuten**"
                )

            with col_right:
                st.markdown("### 📊 Feature Importance")
                importance_df = pd.DataFrame(
                    {
                        "Feature": feature_cols,
                        "Importance": rf_model.feature_importances_,
                    }
                ).sort_values(by="Importance", ascending=True)

                fig_imp = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Belangrijkste Factoren voor Vertraging",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning(
                "Onvoldoende schone data om het Machine Learning model te trainen."
            )
    else:
        st.info(
            "Onvoldoende feature kolommen in de dataset aanwezig voor Machine Learning."
        )


# ------------------------------------------
# TAB 5: VLUCHTTELEMETRIE & DATA INSPECTIE
# ------------------------------------------
with tab5:
    st.subheader("📡 Vluchttelemetrie & Data Inspectie")
    st.write(
        "Bekijk, filter en exporteer de ruwe data van het Schiphol dashboard."
    )

    search_term = st.text_input("🔍 Zoek in dataset (bijv. bestemming of airline):")

    display_df = filtered_df
    if search_term:
        match_mask = display_df.astype(str).apply(
            lambda row: row.str.contains(search_term, case=False).any(), axis=1
        )
        display_df = display_df[match_mask]

    st.dataframe(display_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Dataset Statistieken")
        st.write(display_df.describe())

    with c2:
        st.markdown("### Exporteer Data")
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Gefilterde Data als CSV",
            data=csv_data,
            file_name="schiphol_gefilterde_data.csv",
            mime="text/csv",
        )