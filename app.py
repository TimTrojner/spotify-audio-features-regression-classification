import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Nastavitve strani
st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")

st.title("🎵 Spotify Hit Predictor & Optimizer")
st.markdown("Aplikacija za napovedovanje in simulacijo uspešnosti glasbenih del.")

# Naložimo modele in skaler (poskrbi, da so datoteke v isti mapi!)
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.joblib')
    # Če nimaš vseh datotek še shranjenih, bo tukaj javilo napako
    m_xgb = joblib.load('model_xgb.joblib')
    m_rf = joblib.load('model_rf.joblib')
    m_knn = joblib.load('model_knn.joblib')
    return scaler, m_xgb, m_rf, m_knn

try:
    scaler, model_xgb, model_rf, model_knn = load_assets()
except Exception as e:
    st.error(f"Napaka pri nalaganju modelov: {e}")
    st.stop()

# --- 4.1 Izbira modela ---
st.sidebar.header("Nastavitve")
izbran_ime = st.sidebar.selectbox("Izberi model", ["XGBoost", "Random Forest", "k-NN"])
modeli_dict = {"XGBoost": model_xgb, "Random Forest": model_rf, "k-NN": model_knn}
izbran_model = modeli_dict[izbran_ime]

# --- 4.2 Vnos vrednosti (9 parametrov) ---
st.sidebar.subheader("Parametri pesmi")

# Vrednosti v drsnikih nastavi na povprečja tvojih podatkov ali 0.5
v_valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)
v_tempo = st.sidebar.slider("Tempo", 50.0, 250.0, 120.0)
v_acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.2)
v_energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.7)
v_speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.1)
v_liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.2)
v_loudness = st.sidebar.slider("Loudness", -60.0, 0.0, -8.0)
v_popularity = st.sidebar.slider("Popularity", 0.0, 100.0, 50.0)
v_instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0)

# Priprava podatkov (Tukaj smo popravili imena, da ni več Type Error)
stolpci = ['valence', 'tempo', 'acousticness', 'energy', 'speechiness', 
           'liveness', 'loudness', 'popularity', 'instrumentalness']

input_df = pd.DataFrame([[v_valence, v_tempo, v_acousticness, v_energy, v_speechiness, 
                          v_liveness, v_loudness, v_popularity, v_instrumentalness]], 
                        columns=stolpci)

# Skaliranje
input_scaled = pd.DataFrame(
    scaler.transform(input_df),
    columns=stolpci
)

# --- NAPOVED ---
proba = izbran_model.predict_proba(input_scaled)[0][1]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Rezultat napovedi")
    st.metric(label="Verjetnost za HIT", value=f"{proba:.2%}")
    
    if proba > 0.5:
        st.success("Model napoveduje: TA PESEM JE HIT! 🚀")
    else:
        st.warning("Model napoveduje: Pesem verjetno ne bo hit. 📉")

# --- 4.3 Simulacija sprememb ---
with col2:
    st.subheader("Simulacija optimizacije")
    
    # Uporabnik sam izbere, katero spremenljivko bi rad optimiziral
    sim_feature = st.selectbox("Izberi spremenljivko za simulacijo vpliva:", stolpci)
    
    # Ročno določimo smiselne meje (ker df v aplikaciji ni naložen)
    if sim_feature == 'tempo':
        min_val, max_val = 50.0, 250.0
    elif sim_feature == 'loudness':
        min_val, max_val = -60.0, 0.0
    elif sim_feature == 'popularity':
        min_val, max_val = 0.0, 100.0
    else:
        min_val, max_val = 0.0, 1.0
    
    sim_values = np.linspace(min_val, max_val, 25)
    sim_results = []
    
    # Izračunamo vpliv izbrane spremenljivke na verjetnost hita
    for val in sim_values:
        temp_df = input_df.copy()
        temp_df[sim_feature] = val
        # Pazimo, da stolpci ostanejo v pravilnem vrstnem redu za skaler
        temp_scaled = pd.DataFrame(
            scaler.transform(temp_df[stolpci]),
            columns=stolpci
        )
        sim_results.append(izbran_model.predict_proba(temp_scaled)[0][1])
    
    # Izris grafa
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sim_values, sim_results, color='#1DB954', lw=3)
    ax.set_xlabel(sim_feature.capitalize())
    ax.set_ylabel('Verjetnost hita')
    ax.set_ylim(-0.05, 1.05) # Fiksna skala za lažjo primerjavo
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.info(f"Graf prikazuje, kako bi se spreminjala verjetnost hita, če bi spreminjali le '{sim_feature}', "
            "vse ostale parametre pa pustili tako, kot so nastavljeni na drsnikih.")

# --- Izvoz v CSV ---
st.divider()
if st.button("Izvozi rezultate v CSV"):
    input_df['Prediction_Probability'] = proba
    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button("Prenesi CSV", data=csv, file_name="simulacija_rezultat.csv", mime="text/csv")