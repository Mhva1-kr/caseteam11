import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
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

st.title("✈️ Schiphol Vlucht, Weer & Vertraging Dashboard")
st.write(
    "Een integraal dataplaatje waarin vluchtdata, KNMI-weeromstandigheden, "
    "wereldwijde luchthavencoördinaten en vluchttelemetrie samenkomen."
)


# ==========================================
# 2. HELPER FUNCTIES (HAVERSINE DISTANCE)
# ==========================================
def haversine_distance(lat1, lon1, lat2=52.3105, lon2=4.7683):
    """Berekent afstand in km vanaf Schiphol (52.3105, 4.7683)."""
    try:
        if pd.isna(lat1) or pd.isna(lon1):
            return np.nan
        r = 6371.0
        phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlambda = math.radians(float(lon2) - float(lon1))
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
    except Exception:
        return np.nan


# ==========================================
# 3. ROBUUST DATA INLADEN & SLIMME KOLOM MATCHING
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
        if target_file.endswith(".parquet"):
            df = pd.read_parquet(target_file)
        else:
            df = pd.read_csv(target_file)

        # Verwijder dubbele kolommen uit het originele bestand
        df = df.loc[:, ~df.columns.duplicated()]

        rename_dict = {}
        used_targets = set()

        for col in df.columns:
            c_clean = (
                col.strip()
                .lower()
                .replace("_", "")
                .replace(" ", "")
                .replace("-", "")
                .replace(".", "")
            )

            target = None

            # 1. Delay / Vertraging
            if any(
                k in c_clean
                for k in [
                    "delay",
                    "vertraging",
                    "vertraagd",
                    "depdelay",
                    "arrdelay",
                    "delayminutes",
                ]
            ):
                target = "Delay"

            # 2. Destination / Bestemming
            elif any(
                k in c_clean
                for k in [
                    "destination",
                    "bestemming",
                    "arrival",
                    "destiata",
                    "orgdes",
                    "airport",
                    "dest",
                    "iata",
                ]
            ):
                target = "Destination"

            # 3. Latitude
            elif any(
                k in c_clean
                for k in ["latitude", "lat", "breedtegraad", "destlat"]
            ):
                target = "Latitude"

            # 4. Longitude
            elif any(
                k in c_clean
                for k in [
                    "longitude",
                    "lon",
                    "lng",
                    "lengtegraad",
                    "destlon",
                ]
            ):
                target = "Longitude"

            # 5. Distance / Afstand
            elif any(k in c_clean for k in ["distance", "afstand", "dist"]):
                target = "Distance"

            # 6. Airline / Luchtvaartmaatschappij
            elif any(
                k in c_clean
                for k in [
                    "airline",
                    "carrier",
                    "luchtvaartmaatschappij",
                    "operator",
                ]
            ):
                target = "Airline"

            # 7. DateTime / Datum / Tijd
            elif any(
                k in c_clean
                for k in [
                    "datetime",
                    "scheduledatetime",
                    "actualdatetime",
                    "datum",
                    "tijd",
                    "std",
                    "sta",
                ]
            ):
                target = "DateTime"

            # 8. KNMI Weer Variabelen
            elif any(
                k in c_clean for k in ["windspeed", "windsnelheid", "wind"]
            ):
                target = "Wind_Speed"
            elif any(
                k in c_clean for k in ["temperature", "temperatuur", "temp"]
            ):
                target = "Temperature"
            elif any(
                k in c_clean for k in ["rain", "neerslag", "precipitation"]
            ):
                target = "Rain"
            elif any(k in c_clean for k in ["visibility", "zicht"]):
                target = "Visibility"

            if target and target not in used_targets:
                rename_dict[col] = target
                used_targets.add(target)

        df = df.rename(columns=rename_dict)
        df = df.loc[:, ~df.columns.duplicated()]

        # automatische vertragingsberekening als er geplande vs werkelijke tijden zijn
        if "Delay" not in df.columns:
            time_cols = [
                c
                for c in df.columns
                if any(
                    k in c.lower()
                    for k in [
                        "time",
                        "tijd",
                        "date",
                        "std",
                        "sta",
                        "etd",
                        "eta",
                        "schedule",
                        "actual",
                    ]
                )
            ]
            if len(time_cols) >= 2:
                try:
                    t1 = pd.to_datetime(df[time_cols[0]], errors="coerce")
                    t2 = pd.to_datetime(df[time_cols[1]], errors="coerce")
                    diff_min = (t2 - t1).dt.total_seconds() / 60.0
                    if diff_min.notna().sum() > 0:
                        df["Delay"] = diff_min.clip(lower=0)
                except Exception:
                    pass

        # Converteer en verwerk types
        if "Delay" in df.columns:
            df["Delay"] = pd.to_numeric(df["Delay"], errors="coerce").fillna(0)

        if "Latitude" in df.columns and "Longitude" in df.columns:
            df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
            df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
            if "Distance" not in df.columns:
                df["Distance"] = df.apply(
                    lambda r: haversine_distance(r["Latitude"], r["Longitude"]),
                    axis=1,
                )

        if "DateTime" in df.columns:
            df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
            df["Hour"] = df["DateTime"].dt.hour

        return df, None
    except Exception as e:
        return None, f"Fout bij het verwerken van de data: {str(e)}"


# Data inladen
df, error_msg = load_data()

if df is None:
    st.error(f"❌ {error_msg}")
    st.stop()


# ==========================================
# 4. HANDMATIGE KOLOM KOPPELING (SIDEBAR FALLBACK)
# ==========================================
st.sidebar.markdown("## 🔍 Interactive Filters")

# Als 'Delay' of 'Destination' niet automatisch is gekoppeld, bieden we een handmatige selector
if "Delay" not in df.columns or "Destination" not in df.columns:
    with st.sidebar.expander("⚙️ Kolommen handmatig toewijzen", expanded=True):
        st.write("Koppel de kolommen uit je CSV aan het dashboard:")

        all_cols = ["-- Selecteer Kolom --"] + list(df.columns)

        if "Delay" not in df.columns:
            sel_delay = st.selectbox("Vertragingskolom (minuten):", all_cols)
            if sel_delay != "-- Selecteer Kolom --":
                df["Delay"] = pd.to_numeric(
                    df[sel_delay], errors="coerce"
                ).fillna(0)

        if "Destination" not in df.columns:
            sel_dest = st.selectbox("Bestemmingskolom:", all_cols)
            if sel_dest != "-- Selecteer Kolom --":
                df["Destination"] = df[sel_dest]

filtered_df = df.copy()

# Filter: Luchtvaartmaatschappij
if "Airline" in filtered_df.columns:
    airlines = ["Alles"] + sorted(
        filtered_df["Airline"].dropna().astype(str).unique().tolist()
    )
    sel_airline = st.sidebar.selectbox("Luchtvaartmaatschappij:", airlines)
    if sel_airline != "Alles":
        filtered_df = filtered_df[filtered_df["Airline"] == sel_airline]

# Filter: Bestemming
if "Destination" in filtered_df.columns:
    destinations = ["Alles"] + sorted(
        filtered_df["Destination"].dropna().astype(str).unique().tolist()
    )
    sel_dest = st.sidebar.selectbox("Bestemming:", destinations)
    if sel_dest != "Alles":
        filtered_df = filtered_df[filtered_df["Destination"] == sel_dest]

# Filter: Vertraging
if "Delay" in filtered_df.columns:
    delay_option = st.sidebar.radio(
        "Vertragingsfilter:",
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
    mode_res = filtered_df["Destination"].mode()
    busiest_dest = mode_res[0] if not mode_res.empty else "N/A"
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

    if "Delay" in filtered_df.columns and filtered_df["Delay"].sum() > 0:
        c1, c2 = st.columns(2)

        with c1:
            fig_hist = px.histogram(
                filtered_df,
                x="Delay",
                nbins=35,
                title="Verdeling van Vluchtvertragingen (in Minuten)",
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
                    title="Top 10 Bestemmingen met Meeste Vertraging",
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
            )
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning(
            "⚠️ Vertragingskolom nog niet gekoppeld. Open de sidebar links onder "
            "**'⚙️ Kolommen handmatig toewijzen'** om de juiste vertragingskolom te selecteren."
        )


# ------------------------------------------
# TAB 2: 3D KAART & AFSTANDEN
# ------------------------------------------
with tab2:
    st.subheader("🗺️ Geografische Spreiding & Berekende Afstanden")
    st.write(
        "De afstanden worden berekend via de **Haversine formule** "
        "tussen Schiphol (52.3105° N, 4.7683° E) en de bestemmingen."
    )

    if (
        "Latitude" in filtered_df.columns
        and "Longitude" in filtered_df.columns
    ):
        df_map = filtered_df.dropna(subset=["Latitude", "Longitude"]).copy()

        if not df_map.empty:
            schiphol_lat, schiphol_lon = 52.3105, 4.7683

            arc_data = df_map.drop_duplicates(subset=["Latitude", "Longitude"])[
                [
                    "Latitude",
                    "Longitude",
                    "Destination"
                    if "Destination" in df_map.columns
                    else "Latitude",
                ]
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
            )

            st.pydeck_chart(deck)
            st.map(df_map[["Latitude", "Longitude"]])
        else:
            st.warning("Geen geldige Latitude/Longitude coördinaten gevonden.")
    else:
        st.info(
            "Voeg Latitude en Longitude kolommen toe aan de dataset om de kaart te tonen."
        )


# ------------------------------------------
# TAB 3: WEERINVLOED (KNMI)
# ------------------------------------------
with tab3:
    st.subheader("🌤️ Weerinvloed op Vluchtvertragingen (KNMI)")

    weather_cols = [
        c
        for c in filtered_df.columns
        if c in ["Wind_Speed", "Temperature", "Rain", "Visibility"]
    ]

    if weather_cols and "Delay" in filtered_df.columns:
        sel_weather = st.selectbox("Kies een weer-variabele:", weather_cols)
        fig_scatter = px.scatter(
            filtered_df,
            x=sel_weather,
            y="Delay",
            title=f"Invloed van {sel_weather} op Vertraging",
            opacity=0.5,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info(
            "Geen KNMI weerkolommen (zoals wind, temperatuur, neerslag) gekoppeld aan de dataset."
        )


# ------------------------------------------
# TAB 4: MACHINE LEARNING MODEL
# ------------------------------------------
with tab4:
    st.subheader("🤖 Machine Learning Model")

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

            st.success("✅ Machine Learning Model succesvol getraind!")

            input_data = {}
            for col in feature_cols:
                input_data[col] = st.slider(
                    f"{col}:",
                    float(X[col].min()),
                    float(X[col].max()),
                    float(X[col].mean()),
                )

            pred = rf_model.predict(pd.DataFrame([input_data]))[0]
            st.metric("⏱️ Voorspelde Vertraging", f"{max(0, pred):.1f} min")
        else:
            st.warning("Onvoldoende schone data om het model te trainen.")
    else:
        st.info("Onvoldoende feature kolommen aanwezig voor voorspellingen.")


# ------------------------------------------
# TAB 5: VLUCHTTELEMETRIE & DATA INSPECTIE
# ------------------------------------------
with tab5:
    st.subheader("📡 Vluchttelemetrie & Data Inspectie")

    st.markdown("### 📋 Alle Kolommen in jouw CSV-bestand:")
    st.write(list(df.columns))

    st.markdown("### 🔍 Eerste 500 rijen van de dataset:")
    st.dataframe(filtered_df.head(500), use_container_width=True)
    