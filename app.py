import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Nastavitve strani
st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")

st.title("🎵 Spotify Hit Predictor & Optimizer")
st.markdown("Aplikacija za napovedovanje in simulacijo uspešnosti glasbenih del.")

# Definiraj stolpce za vsak model
CLASSIFICATION_FEATURES = ['valence', 'loudness', 'acousticness', 'speechiness',
                           'popularity', 'instrumentalness', 'liveness']

REGRESSION_BASE_FEATURES = ['duration_ms', 'danceability', 'energy', 'loudness',
                            'speechiness', 'acousticness', 'instrumentalness',
                            'liveness', 'tempo']

# Možne vrednosti za kategorične spremenljivke (prilagodi glede na tvoje podatke)
TRACK_GENRES = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal',
                'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop',
                'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country',
                'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco',
                'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic',
                'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth',
                'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore',
                'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house',
                'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance',
                'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino',
                'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore', 'minimal-techno',
                'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party',
                'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop',
                'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day',
                'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 'rockabilly',
                'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter',
                'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish', 'study',
                'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop',
                'turkish', 'work-out', 'world-music']

KEY_MAPPING = {
    0: 'C',
    1: 'C♯/D♭',
    2: 'D',
    3: 'D♯/E♭',
    4: 'E',
    5: 'F',
    6: 'F♯/G♭',
    7: 'G',
    8: 'G♯/A♭',
    9: 'A',
    10: 'A♯/B♭',
    11: 'B'
}

KEYS = list(KEY_MAPPING.keys())


# Naložimo modele in skaler
@st.cache_resource
def load_assets():
    assets = {'classification': {}, 'regression': {}}

    # Poskusi naložiti klasifikacijske modele
    try:
        scaler_class = joblib.load('./klasifikacija/modeli/scaler.joblib')
        m_xgb_class = joblib.load('./klasifikacija/modeli/model_xgb.joblib')
        m_rf_class = joblib.load('./klasifikacija/modeli/model_rf.joblib')
        m_knn_class = joblib.load('./klasifikacija/modeli/model_knn.joblib')

        assets['classification'] = {
            'scaler': scaler_class,
            'XGBoost': m_xgb_class,
            'Random Forest': m_rf_class,
            'k-NN': m_knn_class
        }
    except FileNotFoundError:
        pass

    # Poskusi naložiti regresijske modele
    try:
        scaler_reg = joblib.load('./regresija/models/scaler_reg.joblib')
        m_xgb_reg = joblib.load('./regresija/models/model_xgb_reg.joblib')
        m_ridge_reg = joblib.load('./regresija/models/model_ridge_reg.joblib')
        m_knn_reg = joblib.load('./regresija/models/model_knn_reg.joblib')

        assets['regression'] = {
            'scaler': scaler_reg,
            'XGBoost': m_xgb_reg,
            'Ridge': m_ridge_reg,
            'KNN': m_knn_reg
        }
    except FileNotFoundError:
        pass

    return assets


assets = load_assets()

# Preveri razpoložljivost modelov
classification_available = len(assets['classification']) > 0
regression_available = len(assets['regression']) > 0

if not classification_available and not regression_available:
    st.error("❌ Nobeni modeli niso na voljo. Preveri poti do modelov.")
    st.stop()

# --- Izbira modela ---
st.sidebar.header("Nastavitve")

tip_opcije = []
if classification_available:
    tip_opcije.append("Klasifikacija (HIT/ne-HIT)")
if regression_available:
    tip_opcije.append("Regresija (popularnost 0-100)")

if len(tip_opcije) == 1:
    tip_modela = tip_opcije[0]
    st.sidebar.info(f"Aktiven tip: **{tip_modela}**")
else:
    tip_modela = st.sidebar.radio("Tip modela", tip_opcije)

# Določi model_type in ustrezne stolpce
if tip_modela.startswith("Klasifikacija"):
    model_type = 'classification'
    modeli_options = [k for k in assets['classification'].keys() if k != 'scaler']
    stolpci = CLASSIFICATION_FEATURES
else:
    model_type = 'regression'
    modeli_options = [k for k in assets['regression'].keys() if k != 'scaler']
    stolpci = REGRESSION_BASE_FEATURES

izbran_ime = st.sidebar.selectbox("Izberi model", modeli_options)
izbran_model = assets[model_type][izbran_ime]
scaler = assets[model_type]['scaler']

# --- Način vnosa ---
st.sidebar.subheader("Način vnosa")
input_mode = st.sidebar.radio("Izberi način vnosa:", ["Posamezen vzorec (drsniki)", "Množični vnos (CSV)"])

# ========== NAČIN 1: Posamezen vnos ==========
if input_mode == "Posamezen vzorec (drsniki)":
    st.sidebar.subheader("Parametri pesmi")

    # Dinamični drsniki glede na izbran tip modela
    input_values = {}

    if model_type == 'classification':
        input_values['valence'] = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)
        input_values['loudness'] = st.sidebar.slider("Loudness", -60.0, 0.0, -8.0)
        input_values['acousticness'] = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.2)
        input_values['speechiness'] = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.1)
        input_values['popularity'] = st.sidebar.slider("Popularity", 0.0, 100.0, 50.0)
        input_values['instrumentalness'] = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0)
        input_values['liveness'] = st.sidebar.slider("Liveness", 0.0, 1.0, 0.2)
    else:  # regression
        input_values['duration_ms'] = st.sidebar.slider("Duration (ms)", 30000, 600000, 210000)
        input_values['danceability'] = st.sidebar.slider("Danceability", 0.0, 1.0, 0.6)
        input_values['energy'] = st.sidebar.slider("Energy", 0.0, 1.0, 0.7)
        input_values['loudness'] = st.sidebar.slider("Loudness", -60.0, 0.0, -8.0)
        input_values['speechiness'] = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.1)
        input_values['acousticness'] = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.2)
        input_values['instrumentalness'] = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0)
        input_values['liveness'] = st.sidebar.slider("Liveness", 0.0, 1.0, 0.2)
        input_values['tempo'] = st.sidebar.slider("Tempo", 50.0, 250.0, 120.0)

        # Kategorične spremenljivke
        input_values['track_genre'] = st.sidebar.selectbox("Track Genre", TRACK_GENRES)
        selected_key_num = st.sidebar.selectbox("Key (tonaliteta)", KEYS, format_func=lambda x: KEY_MAPPING[x])
        input_values['key'] = selected_key_num

    # Priprava podatkov
    input_df = pd.DataFrame([input_values])

    # One-hot encoding za regresijo
    if model_type == 'regression':
        input_df = pd.get_dummies(input_df, columns=['track_genre', 'key'], drop_first=True)

        # Dodaj manjkajoče stolpce z reindex (optimizirano)
        expected_cols = scaler.feature_names_in_
        input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    # Skaliranje
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    # --- NAPOVED ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rezultat napovedi")

        if model_type == 'classification':
            proba = izbran_model.predict_proba(input_scaled)[0][1]
            st.metric(label="Verjetnost za HIT", value=f"{proba:.2%}")

            fig_gauge, ax_gauge = plt.subplots(figsize=(6, 1.5))
            color = '#FF4444' if proba < 0.3 else ('#FFB347' if proba < 0.6 else '#1DB954')
            ax_gauge.barh([0], [proba], color=color, height=0.5, edgecolor='black', linewidth=1.5)
            ax_gauge.set_xlim(0, 1)
            ax_gauge.set_ylim(-0.5, 0.5)
            ax_gauge.set_yticks([])
            ax_gauge.set_xlabel("Verjetnost")
            ax_gauge.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_gauge.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            st.pyplot(fig_gauge)

            if proba > 0.5:
                st.success("Model napoveduje: TA PESEM JE HIT! 🚀")
            else:
                st.warning("Model napoveduje: Pesem verjetno ne bo hit. 📉")
        else:
            napoved_pop = float(izbran_model.predict(input_scaled)[0])
            st.metric(label="Napovedana popularnost", value=f"{napoved_pop:.1f} / 100")
            st.progress(min(max(napoved_pop / 100, 0), 1))

            if napoved_pop >= 70:
                st.success(f"Visoka popularnost: {napoved_pop:.1f}/100 🎉")
            elif napoved_pop >= 40:
                st.info(f"Srednja popularnost: {napoved_pop:.1f}/100")
            else:
                st.warning(f"Nizka popularnost: {napoved_pop:.1f}/100")

    # --- Simulacija sprememb (samo za numerične značke) ---
    with col2:
        st.subheader("Simulacija optimizacije")
        sim_feature = st.selectbox("Izberi spremenljivko za simulacijo vpliva:", stolpci)

        # Določi meje
        if sim_feature == 'tempo':
            min_val, max_val = 50.0, 250.0
        elif sim_feature == 'loudness':
            min_val, max_val = -60.0, 0.0
        elif sim_feature == 'popularity':
            min_val, max_val = 0.0, 100.0
        elif sim_feature == 'duration_ms':
            min_val, max_val = 30000, 600000
        else:
            min_val, max_val = 0.0, 1.0

        sim_values = np.linspace(min_val, max_val, 30)
        sim_results = []

        for val in sim_values:
            temp_values = input_values.copy()
            temp_values[sim_feature] = val
            temp_df = pd.DataFrame([temp_values])

            # One-hot encoding za regresijo
            if model_type == 'regression':
                temp_df = pd.get_dummies(temp_df, columns=['track_genre', 'key'], drop_first=True)
                # Uporabi reindex namesto zanke (optimizirano)
                temp_df = temp_df.reindex(columns=expected_cols, fill_value=0)

            temp_scaled = pd.DataFrame(scaler.transform(temp_df), columns=temp_df.columns)

            if model_type == 'classification':
                sim_results.append(izbran_model.predict_proba(temp_scaled)[0][1])
            else:
                sim_results.append(float(izbran_model.predict(temp_scaled)[0]))

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(sim_values, sim_results, color='#1DB954', lw=3, marker='o', markersize=4, alpha=0.7)

        current_val = input_values[sim_feature]
        current_pred = sim_results[np.argmin(np.abs(sim_values - current_val))]
        ax.scatter([current_val], [current_pred], color='red', s=150, zorder=5,
                   label='Trenutna vrednost', edgecolors='black', linewidth=2)

        ax.set_xlabel(sim_feature.capitalize(), fontsize=12, fontweight='bold')

        if model_type == 'classification':
            ax.set_ylabel('Verjetnost hita', fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Prag (50%)')
        else:
            ax.set_ylabel('Napovedana popularnost', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 100)

        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)  # --- Izvoz ---
    st.divider()
    if st.button("Izvozi rezultate v CSV"):
        export_df = pd.DataFrame([input_values])
        if model_type == 'classification':
            export_df['Prediction_Probability'] = proba
            export_df['Prediction_HIT'] = int(proba > 0.5)
        else:
            export_df['Predicted_Popularity'] = napoved_pop

        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Prenesi CSV", data=csv, file_name="simulacija_rezultat.csv", mime="text/csv")

# ========== NAČIN 2: Množični vnos CSV ==========
else:
    st.subheader("📂 Množična napoved (CSV upload)")

    if model_type == 'regression':
        required_cols = REGRESSION_BASE_FEATURES + ['track_genre', 'key']
    else:
        required_cols = CLASSIFICATION_FEATURES

    st.markdown(f"**Naloži CSV datoteko, ki vsebuje stolpce:** `{', '.join(required_cols)}`")

    uploaded_file = st.file_uploader("Izberi CSV datoteko", type=['csv'])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"**Naloženih vzorcev:** {len(df_batch)}")

            missing_cols = [col for col in required_cols if col not in df_batch.columns]
            if missing_cols:
                st.error(f"❌ Manjkajoči stolpci: {', '.join(missing_cols)}")
            else:
                # One-hot encoding za regresijo
                if model_type == 'regression':
                    df_batch_encoded = pd.get_dummies(df_batch, columns=['track_genre', 'key'], drop_first=True)
                    expected_cols = scaler.feature_names_in_
                    # Uporabi reindex namesto zanke (optimizirano)
                    df_batch_encoded = df_batch_encoded.reindex(columns=expected_cols, fill_value=0)
                else:
                    df_batch_encoded = df_batch[required_cols]

                batch_scaled = pd.DataFrame(scaler.transform(df_batch_encoded), columns=df_batch_encoded.columns)

                if model_type == 'classification':
                    df_batch['Prediction_Probability'] = izbran_model.predict_proba(batch_scaled)[:, 1]
                    df_batch['Prediction_HIT'] = (df_batch['Prediction_Probability'] > 0.5).astype(int)
                    st.success(
                        f"✅ Napoved končana! **{df_batch['Prediction_HIT'].sum()}** od {len(df_batch)} pesmi napovedanih kot HIT.")
                else:
                    df_batch['Predicted_Popularity'] = izbran_model.predict(batch_scaled)
                    avg_pop = df_batch['Predicted_Popularity'].mean()
                    st.success(f"✅ Napoved končana! **Povprečna napovedana popularnost:** {avg_pop:.1f}/100")

                st.dataframe(df_batch, use_container_width=True)

                st.subheader("📊 Porazdelitev napovedi")
                fig_hist, ax_hist = plt.subplots(figsize=(8, 4))

                if model_type == 'classification':
                    ax_hist.hist(df_batch['Prediction_Probability'], bins=20, color='#1DB954', edgecolor='black',
                                 alpha=0.7)
                    ax_hist.set_xlabel('Verjetnost hita')
                    ax_hist.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Prag (50%)')
                else:
                    ax_hist.hist(df_batch['Predicted_Popularity'], bins=20, color='#1DB954', edgecolor='black',
                                 alpha=0.7)
                    ax_hist.set_xlabel('Napovedana popularnost')

                ax_hist.set_ylabel('Število pesmi')
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                st.pyplot(fig_hist)

                csv_output = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Prenesi rezultate", data=csv_output,
                                   file_name="batch_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Napaka pri obdelavi datoteke: {e}")
    else:
        st.info("👆 Naloži CSV datoteko za množično napovedovanje.")
