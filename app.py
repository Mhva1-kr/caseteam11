import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as bg
import pydeck as pdk

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Vluchten & Vertraging Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ Dashboard Vluchten & Vertragingen")
st.markdown(" Inzichten in vluchttijden, vertragingen, geografische spreiding en voorspellingen.")

# ==========================================
# 2. DATA LOADING & PREPROCESSING (MOCK / DEMO DATA EN ENGINE)
# ==========================================
@st.cache_data
def load_data():
    # Vervang dit deel door het inladen van je eigen bestanden:
    # df_schedule = pd.read_csv('schedule_airport.csv')
    # df_airports = pd.read_csv('airports-extended.csv', header=None)
    
    # Voorbeeld Mock Data structuur gebaseerd op de opdracht:
    n = 1000
    dates = pd.date_range(start='2023-01-01', periods=n, freq='2h')
    
    data = {
        'Datum': dates,
        'Vluchtnummer': [f'HV{100 + i%50}' for i in range(n)],
        'Geplande_Aankomst': dates,
        'Werkelijke_Aankomst': dates + pd.to_timedelta(np.random.normal(15, 30, n), unit='m'),
        'Inbound_Outbound': np.random.choice(['L', 'S'], n),
        'Geplande_Gate': np.random.choice(['D1', 'D2', 'E4', 'E8', 'F2', 'G3'], n),
        'Werkelijke_Gate': np.random.choice(['D1', 'D2', 'E4', 'E8', 'F2', 'G3'], n),
        'Vliegtuigtype': np.random.choice(['B737-800', 'A320', 'B787-9', 'E190'], n),
        'Start_Landingsbaan': np.random.choice(['18R/36L', '06/24', '18C/36C', '24/06'], n),
        'Bestemming': np.random.choice(['BCN', 'LHR', 'JFK', 'CDG', 'AGP', 'ALC'], n),
        'Lat': np.random.uniform(36.0, 55.0, n),
        'Lon': np.random.uniform(-5.0, 15.0, n)
    }
    
    df = pd.DataFrame(data)
    # Bereken vertraging in minuten
    df['Vertraging_Minuten'] = (df['Werkelijke_Aankomst'] - df['Geplande_Aankomst']).dt.total_seconds() / 60
    df['Is_Vertraagd'] = df['Vertraging_Minuten'] > 15
    df['Uur'] = df['Geplande_Aankomst'].dt.hour
    df['Maand'] = df['Geplande_Aankomst'].dt.strftime('%Y-%m')
    return df

df = load_data()

# ==========================================
# 3. SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🔍 Filter Opties")

# Datum Filter
min_date = df['Datum'].min().date()
max_date = df['Datum'].max().date()
date_range = st.sidebar.date_input("Selecteer Datum Periode", [min_date, max_date])

# Vliegtuigtype Filter
vliegtuig_options = ['Alles'] + list(df['Vliegtuigtype'].unique())
selected_vliegtuig = st.sidebar.selectbox("Vliegtuigtype", vliegtuig_options)

# Start/Landingsbaan Filter
baan_options = ['Alles'] + list(df['Start_Landingsbaan'].unique())
selected_baan = st.sidebar.selectbox("Start-/Landingsbaan", baan_options)

# Vertragingsdrempel
min_delay = st.sidebar.slider("Minimale Vertraging (minuten)", -30, 120, 15)

# Toepassen van filters op de dataset
filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['Datum'].dt.date >= date_range[0]) & (filtered_df['Datum'].dt.date <= date_range[1])]
if selected_vliegtuig != 'Alles':
    filtered_df = filtered_df[filtered_df['Vliegtuigtype'] == selected_vliegtuig]
if selected_baan != 'Alles':
    filtered_df = filtered_df[filtered_df['Start_Landingsbaan'] == selected_baan]

filtered_df = filtered_df[filtered_df['Vertraging_Minuten'] >= min_delay]

# ==========================================
# 4. KPI METRICS HEADER
# ==========================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Totaal Geanalyseerde Vluchten", value=f"{len(filtered_df):,}")
with kpi2:
    avg_delay = filtered_df['Vertraging_Minuten'].mean()
    st.metric(label="Gemiddelde Vertraging", value=f"{avg_delay:.1f} min" if not np.isnan(avg_delay) else "N/A")
with kpi3:
    vertraagd_pct = (filtered_df['Is_Vertraagd'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric(label="Percentage Vertraagd (>15m)", value=f"{vertraagd_pct:.1f}%")
with kpi4:
    top_bestemming = filtered_df['Bestemming'].mode()[0] if len(filtered_df) > 0 else "N/A"
    st.metric(label="Meest Frequente Bestemming", value=top_bestemming)

st.divider()

# ==========================================
# 5. TABBLADEN VOOR VERDELING VAN ANALYSES
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Tijdsanalyse & Trends", "🗺️ Geografische Kaart", "🤖 Vertraging Voorspellen", "📋 Data Inspectie"])

# --- TAB 1: TIJDSANALYSE (LIJNGRAFIEK & PATRONEN) ---
with tab1:
    st.subheader("Verloop van Vertragingen over de Tijd")
    
    # Aggergeren per dag/uur
    df_trend = filtered_df.groupby('Maand')['Vertraging_Minuten'].mean().reset_index()
    
    fig_line = px.line(
        df_trend, 
        x='Maand', 
        y='Vertraging_Minuten',
        title="Gemiddelde Vertraging per Maand (Lijngrafiek)",
        markers=True,
        labels={'Vertraging_Minuten': 'Gemiddelde Vertraging (min)', 'Maand': 'Maand'}
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Vertraging per Uur van de Dag")
        df_uur = filtered_df.groupby('Uur')['Vertraging_Minuten'].mean().reset_index()
        fig_bar = px.bar(df_uur, x='Uur', y='Vertraging_Minuten', color='Vertraging_Minuten', title="Drukte & Piekuren")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_b:
        st.markdown("#### Vertraging per Start-/Landingsbaan")
        fig_box = px.box(filtered_df, x='Start_Landingsbaan', y='Vertraging_Minuten', title="Spreiding per Baan")
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 2: GEOGRAFISCHE KAART ---
with tab2:
    st.subheader("Geografische Spreiding & Vluchtroutes")
    st.markdown("Visualisatie van bestemmingen en gerelateerde vertragingen.")
    
    # Pydeck Kaart
    view_state = pdk.ViewState(latitude=52.3676, longitude=4.9041, zoom=4, pitch=30)
    
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=filtered_df,
        get_position='[Lon, Lat]',
        get_color='[200, 30, 0, 160]',
        get_radius=50000,
        pickable=True
    )
    
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Bestemming: {Bestemming}
Vertraging: {Vertraging_Minuten} min"}))

# --- TAB 3: VOORSPELMODEL (MACHINE LEARNING DEMO) ---
with tab3:
    st.subheader("🤖 Vertraging Voorspellen (Machine Learning)")
    st.markdown("Vul de kenmerken van een geplande vlucht in om de verwachte vertraging te berekenen.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        input_vliegtuig = st.selectbox("Selecteer Vliegtuigtype", df['Vliegtuigtype'].unique())
        input_uur = st.slider("Gepland Uur van Vertrek", 0, 23, 14)
    with col_input2:
        input_baan = st.selectbox("Selecteer Start/Landingsbaan", df['Start_Landingsbaan'].unique())
        input_inbound = st.radio("Vluchttype", ["Inbound (L)", "Outbound (S)"])
    with col_input3:
        input_gate = st.selectbox("Geplande Gate", df['Geplande_Gate'].unique())
    
    if st.button("Calculate Predicted Delay 🚀"):
        # Eenvoudige demonstratieve formule (vervangen door trained scikit-learn model object)
        base_delay = 5.0
        if input_uur in [8, 9, 17, 18, 19]:
            base_delay += 12.5 # Piekuren extra vertraging
        if input_vliegtuig in ['B787-9']:
            base_delay += 5.0
            
        st.success(f" Voorspelde Vertraging: **{base_delay:.1f} minuten**")
        st.info("💡 **Inzicht:** Vertragingen pieken voornamelijk tijdens spitsuren (08:00 - 10:00 & 17:00 - 19:00).")

# --- TAB 4: DATA INSPECTIE & EXPORT ---
with tab4:
    st.subheader("Data Inspectie & Exporteren")
    st.markdown("Bekijk de gefilterde opschonde dataset en download deze als CSV.")
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download knop
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Gefilterde Data als CSV",
        data=csv_data,
        file_name="gefilterde_vluchtdata.csv",
        mime="text/csv"
    )
