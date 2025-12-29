Razumem. Želite nov Markdown datoteko, ki bo služila kot osnutek za analizo Spotify podatkov, podobno obstoječemu `spotify_ANALysis.ipynb` z dodatnimi predlogi za regresijske in klasifikacijske modele, posebej za problem popularnosti in uspešnic. Poudarek bo na logistični in linearni regresiji ter drugih modelih, kljub omejeni velikosti podatkovne baze, a z veliko računske moči.

Tukaj je predlog za strukturo in vsebino nove Markdown datoteke:

```markdown
# Spotify Music Analysis & Machine Learning Pipeline - Razširjena Analiza

## Celovita analiza Spotify skladb z osredotočenostjo na napovedovanje popularnosti in uspešnic

Ta dokument razširja obstoječo analizo z dodatnimi modeli in optimizacijskimi pristopi za razumevanje in napovedovanje uspeha glasbe.

**Cilji:**
- Poglobljena raziskovalna analiza podatkov (EDA)
- Napredno inženirstvo značilk
- Razvoj in optimizacija modelov za napovedovanje popularnosti (regresija)
- Razvoj in optimizacija modelov za napovedovanje uspešnic (klasifikacija)
- Predlogi za optimizacijo zvočnih parametrov za maksimizacijo uspeha

**Dataset vključuje:** metapodatke skladb + zvočne značilnosti (danceability, energy, tempo, valence, itd.)

## 1. Uvoz knjižnic

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Sklearn moduli za predprocesiranje in modele
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Sklearn moduli za evalvacijo
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
                            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
```

## 2. Nalaganje in raziskovanje podatkov

```python
df = pd.read_csv('spotify_analysis_dataset.csv')

print(f"Naloženih {len(df)} skladb")
print(f"Število stolpcev: {len(df.columns)}")
print("\nPrvih 5 vrstic podatkov:")
print(df.head())
print("\nInformacije o podatkovnem okviru:")
df.info()
print("\nStatistični povzetek numeričnih stolpcev:")
print(df.describe())
```

## 3. Predprocesiranje podatkov

```python
# Pretvorba trajanja iz milisekund v minute
if 'duration_ms' in df.columns:
    df['duration_min'] = df['duration_ms'] / 60000
    print("Dodana značilnost 'duration_min'.")

# Ekstrakcija leta izdaje iz datuma izdaje
if 'release_date' in df.columns:
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    # Odstranimo vrstice, kjer je 'release_year' NaN po konverziji
    df.dropna(subset=['release_year'], inplace=True)
    df['release_year'] = df['release_year'].astype(int)
    print("Dodana značilnost 'release_year'.")

# Obravnava manjkajočih vrednosti (primer: zamenjava z mediano ali povprečjem)
# Zaenkrat bomo preprosto odstranili vrstice z manjkajočimi vrednostmi v ključnih numeričnih stolpcih
initial_rows = len(df)
df.dropna(subset=['popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'], inplace=True)
print(f"Odstranjenih {initial_rows - len(df)} vrstic z manjkajočimi vrednostmi v ključnih značilkah.")

# Preverjanje duplikatov
df.drop_duplicates(inplace=True)
print(f"Število vrstic po odstranitvi duplikatov: {len(df)}")
```

## 4. Inženirstvo značilk

```python
# Seznam avdio značilk, ki jih bomo uporabili
audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_min']
# Filtriramo, da zagotovimo prisotnost v DataFrame-u
audio_features = [f for f in audio_features if f in df.columns]

# Izračun 'mood_score' kot kombinacija valence, energy in danceability
if 'valence' in df.columns and 'energy' in df.columns and 'danceability' in df.columns:
    df['mood_score'] = df['valence'] * 0.4 + df['energy'] * 0.3 + df['danceability'] * 0.3
    audio_features.append('mood_score') # Dodamo novo značilko v seznam
    print("Dodana značilnost 'mood_score'.")

# Dodatne značilke:
# Interakcijske značilke (primer: energy * loudness)
df['energy_loudness'] = df['energy'] * df['loudness']
audio_features.append('energy_loudness')
print("Dodana značilnost 'energy_loudness'.")

# Binarna značilka za instrumentalne skladbe (instrumentalness > 0.5)
df['is_instrumental'] = (df['instrumentalness'] > 0.5).astype(int)
audio_features.append('is_instrumental')
print("Dodana značilnost 'is_instrumental'.")

# Logaritemska transformacija za poševne značilke (npr. loudness, speechiness, instrumentalness)
# Preverimo, če so vrednosti pozitivne, preden uporabimo logaritem
for feature in ['speechiness', 'instrumentalness']:
    if feature in df.columns and (df[feature] >= 0).all():
        df[f'log_{feature}'] = np.log1p(df[feature]) # log1p(x) = log(1+x) za obravnavo ničel
        audio_features.append(f'log_{feature}')
        print(f"Dodana značilnost 'log_{feature}'.")

# Odstranimo morebitne duplikate iz seznama značilk
audio_features = list(set(audio_features))
print(f"\nKončni seznam avdio značilk za modeliranje: {audio_features}")
```

## 5. Raziskovalna analiza podatkov (EDA)

```python
# Korelacijska matrika
plt.figure(figsize=(12, 10))
sns.heatmap(df[audio_features + ['popularity']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelacijska matrika avdio značilk in popularnosti')
plt.show()

# Porazdelitev popularnosti
plt.figure(figsize=(8, 6))
sns.histplot(df['popularity'], bins=30, kde=True)
plt.title('Porazdelitev popularnosti skladb')
plt.xlabel('Popularnost (0-100)')
plt.ylabel('Število skladb')
plt.show()

# Odnos med izbranimi značilkami in popularnostjo (scatter plots)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Odnos med avdio značilkami in popularnostjo', fontsize=16)

sns.scatterplot(x='danceability', y='popularity', data=df, ax=axes[0, 0], alpha=0.6)
axes[0, 0].set_title('Danceability vs. Popularnost')

sns.scatterplot(x='energy', y='popularity', data=df, ax=axes[0, 1], alpha=0.6)
axes[0, 1].set_title('Energy vs. Popularnost')

sns.scatterplot(x='valence', y='popularity', data=df, ax=axes[0, 2], alpha=0.6)
axes[0, 2].set_title('Valence vs. Popularnost')

sns.scatterplot(x='loudness', y='popularity', data=df, ax=axes[1, 0], alpha=0.6)
axes[1, 0].set_title('Loudness vs. Popularnost')

sns.scatterplot(x='tempo', y='popularity', data=df, ax=axes[1, 1], alpha=0.6)
axes[1, 1].set_title('Tempo vs. Popularnost')

sns.scatterplot(x='release_year', y='popularity', data=df, ax=axes[1, 2], alpha=0.6)
axes[1, 2].set_title('Leto izdaje vs. Popularnost')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Box plot za popularnost glede na 'is_instrumental'
plt.figure(figsize=(8, 6))
sns.boxplot(x='is_instrumental', y='popularity', data=df)
plt.title('Popularnost glede na to, ali je skladba instrumentalna')
plt.xlabel('Je instrumentalna (0=Ne, 1=Da)')
plt.ylabel('Popularnost')
plt.show()
```

## 6. A. Regresijski problem: Napovedovanje popularnosti

**Cilj (Y):** `popularity` (Indeks popularnosti od 0 do 100).
**Vhodni podatki (X):** Tehnične lastnosti skladbe (`audio_features`).
**Optimizacija (Simulacija):** Produkcija uspešnice. Iskanje optimalnih zvočnih parametrov za maksimizacijo napovedane popularnosti.

### Priprava podatkov za regresijo

```python
# Uporabimo očiščen DataFrame in izbrane značilke
X_reg = df[audio_features].copy()
y_reg = df['popularity'].copy()

# Odstranimo morebitne NaN vrednosti, ki so se pojavile po inženirstvu značilk
# To je pomembno, saj nekateri modeli ne delujejo z NaN
initial_rows_reg = len(X_reg)
combined_reg = pd.concat([X_reg, y_reg], axis=1).dropna()
X_reg = combined_reg[audio_features]
y_reg = combined_reg['popularity']
print(f"Odstranjenih {initial_rows_reg - len(X_reg)} vrstic z NaN vrednostmi za regresijo.")

# Razdelitev podatkov na učno in testno množico
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Skaliranje značilk
scaler_reg = StandardScaler()
X_train_scaled_reg = scaler_reg.fit_transform(X_train_reg)
X_test_scaled_reg = scaler_reg.transform(X_test_reg)

print(f"Velikost učne množice (regresija): {X_train_scaled_reg.shape}")
print(f"Velikost testne množice (regresija): {X_test_scaled_reg.shape}")
```

### Modeli za napovedovanje popularnosti

#### 6.A.1. Linearna regresija

```python
print("--- Linearna regresija ---")
linear_model = LinearRegression()
linear_model.fit(X_train_scaled_reg, y_train_reg)
y_pred_linear = linear_model.predict(X_test_scaled_reg)

mse_linear = mean_squared_error(y_test_reg, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test_reg, y_pred_linear)
r2_linear = r2_score(y_test_reg, y_pred_linear)

print(f"MSE: {mse_linear:.2f}")
print(f"RMSE: {rmse_linear:.2f}")
print(f"MAE: {mae_linear:.2f}")
print(f"R2 Score: {r2_linear:.2f}")

# Analiza koeficientov
coefficients = pd.DataFrame({'Feature': audio_features, 'Coefficient': linear_model.coef_})
print("\nKoeficienti linearne regresije:")
print(coefficients.sort_values(by='Coefficient', ascending=False))
```

#### 6.A.2. Ridge regresija (z regularizacijo L2)

```python
print("\n--- Ridge regresija ---")
ridge_model = Ridge(random_state=42)
# Optimizacija hiperparametrov z GridSearchCV
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]} # Parameter alpha nadzoruje moč regularizacije

grid_search_ridge = GridSearchCV(ridge_model, param_grid_ridge, cv=5, scoring='r2', n_jobs=-1)
grid_search_ridge.fit(X_train_scaled_reg, y_train_reg)

print(f"Najboljši parametri za Ridge: {grid_search_ridge.best_params_}")
best_ridge_model = grid_search_ridge.best_estimator_
y_pred_ridge = best_ridge_model.predict(X_test_scaled_reg)

mse_ridge = mean_squared_error(y_test_reg, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = mean_absolute_error(y_test_reg, y_pred_ridge)
r2_ridge = r2_score(y_test_reg, y_pred_ridge)

print(f"MSE: {mse_ridge:.2f}")
print(f"RMSE: {rmse_ridge:.2f}")
print(f"MAE: {mae_ridge:.2f}")
print(f"R2 Score: {r2_ridge:.2f}")
```

#### 6.A.3. Lasso regresija (z regularizacijo L1)

```python
print("\n--- Lasso regresija ---")
lasso_model = Lasso(random_state=42)
# Optimizacija hiperparametrov z GridSearchCV
param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10]} # Parameter alpha nadzoruje moč regularizacije

grid_search_lasso = GridSearchCV(lasso_model, param_grid_lasso, cv=5, scoring='r2', n_jobs=-1)
grid_search_lasso.fit(X_train_scaled_reg, y_train_reg)

print(f"Najboljši parametri za Lasso: {grid_search_lasso.best_params_}")
best_lasso_model = grid_search_lasso.best_estimator_
y_pred_lasso = best_lasso_model.predict(X_test_scaled_reg)

mse_lasso = mean_squared_error(y_test_reg, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test_reg, y_pred_lasso)
r2_lasso = r2_score(y_test_reg, y_pred_lasso)

print(f"MSE: {mse_lasso:.2f}")
print(f"RMSE: {rmse_lasso:.2f}")
print(f"MAE: {mae_lasso:.2f}")
print(f"R2 Score: {r2_lasso:.2f}")

# Lasso lahko izbere značilke (nastavi koeficiente na 0)
coefficients_lasso = pd.DataFrame({'Feature': audio_features, 'Coefficient': best_lasso_model.coef_})
print("\nKoeficienti Lasso regresije (pomembnost značilk):")
print(coefficients_lasso[coefficients_lasso['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False))
```

#### 6.A.4. Random Forest Regressor

```python
print("\n--- Random Forest Regressor ---")
rf_reg_model = RandomForestRegressor(random_state=42, n_jobs=-1)
# Optimizacija hiperparametrov (primer, lahko traja dolgo)
param_grid_rf_reg = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search_rf_reg = GridSearchCV(rf_reg_model, param_grid_rf_reg, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search_rf_reg.fit(X_train_scaled_reg, y_train_reg)

print(f"Najboljši parametri za Random Forest Regressor: {grid_search_rf_reg.best_params_}")
best_rf_reg_model = grid_search_rf_reg.best_estimator_
y_pred_rf_reg = best_rf_reg_model.predict(X_test_scaled_reg)

mse_rf_reg = mean_squared_error(y_test_reg, y_pred_rf_reg)
rmse_rf_reg = np.sqrt(mse_rf_reg)
mae_rf_reg = mean_absolute_error(y_test_reg, y_pred_rf_reg)
r2_rf_reg = r2_score(y_test_reg, y_pred_rf_reg)

print(f"MSE: {mse_rf_reg:.2f}")
print(f"RMSE: {rmse_rf_reg:.2f}")
print(f"MAE: {mae_rf_reg:.2f}")
print(f"R2 Score: {r2_rf_reg:.2f}")

# Pomembnost značilk
feature_importances_rf_reg = pd.DataFrame({'Feature': audio_features, 'Importance': best_rf_reg_model.feature_importances_})
print("\nPomembnost značilk (Random Forest Regressor):")
print(feature_importances_rf_reg.sort_values(by='Importance', ascending=False))
```

#### 6.A.5. Gradient Boosting Regressor

```python
print("\n--- Gradient Boosting Regressor ---")
gb_reg_model = GradientBoostingRegressor(random_state=42)
# Optimizacija hiperparametrov (primer)
param_grid_gb_reg = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

grid_search_gb_reg = GridSearchCV(gb_reg_model, param_grid_gb_reg, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search_gb_reg.fit(X_train_scaled_reg, y_train_reg)

print(f"Najboljši parametri za Gradient Boosting Regressor: {grid_search_gb_reg.best_params_}")
best_gb_reg_model = grid_search_gb_reg.best_estimator_
y_pred_gb_reg = best_gb_reg_model.predict(X_test_scaled_reg)

mse_gb_reg = mean_squared_error(y_test_reg, y_pred_gb_reg)
rmse_gb_reg = np.sqrt(mse_gb_reg)
mae_gb_reg = mean_absolute_error(y_test_reg, y_pred_gb_reg)
r2_gb_reg = r2_score(y_test_reg, y_pred_gb_reg)

print(f"MSE: {mse_gb_reg:.2f}")
print(f"RMSE: {rmse_gb_reg:.2f}")
print(f"MAE: {mae_gb_reg:.2f}")
print(f"R2 Score: {r2_gb_reg:.2f}")

# Pomembnost značilk
feature_importances_gb_reg = pd.DataFrame({'Feature': audio_features, 'Importance': best_gb_reg_model.feature_importances_})
print("\nPomembnost značilk (Gradient Boosting Regressor):")
print(feature_importances_gb_reg.sort_values(by='Importance', ascending=False))
```

### Optimizacija (Simulacija): Produkcija uspešnice

Na podlagi najboljšega regresijskega modela (npr. Random Forest ali Gradient Boosting, glede na R2 rezultat) lahko poskusimo optimizirati zvočne parametre. To je simulacijski problem, kjer želimo najti kombinacijo značilk, ki maksimizira napovedano popularnost.

```python
# Izberemo najboljši model (npr. best_rf_reg_model ali best_gb_reg_model)
# Za demonstracijo bomo uporabili Random Forest
best_reg_model = best_rf_reg_model # Ali best_gb_reg_model, odvisno od rezultatov

# Poiščemo povprečne vrednosti značilk kot izhodišče
avg_features = X_reg.mean().to_dict()

# Ustvarimo DataFrame za simulacijo
sim_df = pd.DataFrame([avg_features])
sim_df_scaled = scaler_reg.transform(sim_df[audio_features])
initial_popularity = best_reg_model.predict(sim_df_scaled)[0]
print(f"Izhodiščna napovedana popularnost (povprečne značilke): {initial_popularity:.2f}")

# Simulacija: Spreminjanje posameznih značilk in opazovanje vpliva na popularnost
# Povečamo danceability za 10%
sim_df_dance = sim_df.copy()
if 'danceability' in sim_df_dance.columns:
    sim_df_dance['danceability'] *= 1.1
    sim_df_dance_scaled = scaler_reg.transform(sim_df_dance[audio_features])
    new_popularity_dance = best_reg_model.predict(sim_df_dance_scaled)[0]
    print(f"Napovedana popularnost s +10% danceability: {new_popularity_dance:.2f} (Sprememba: {new_popularity_dance - initial_popularity:.2f})")

# Povečamo energy za 10%
sim_df_energy = sim_df.copy()
if 'energy' in sim_df_energy.columns:
    sim_df_energy['energy'] *= 1.1
    sim_df_energy_scaled = scaler_reg.transform(sim_df_energy[audio_features])
    new_popularity_energy = best_reg_model.predict(sim_df_energy_scaled)[0]
    print(f"Napovedana popularnost s +10% energy: {new_popularity_energy:.2f} (Sprememba: {new_popularity_energy - initial_popularity:.2f})")

# Zmanjšamo speechiness za 10%
sim_df_speech = sim_df.copy()
if 'speechiness' in sim_df_speech.columns:
    sim_df_speech['speechiness'] *= 0.9
    sim_df_speech_scaled = scaler_reg.transform(sim_df_speech[audio_features])
    new_popularity_speech = best_reg_model.predict(sim_df_speech_scaled)[0]
    print(f"Napovedana popularnost z -10% speechiness: {new_popularity_speech:.2f} (Sprememba: {new_popularity_speech - initial_popularity:.2f})")

# Bolj napredna optimizacija bi vključevala iterativno iskanje ali genetske algoritme
# za iskanje optimalne kombinacije značilk znotraj realističnih razponov.
# Za to bi potrebovali definirati meje za vsako značilko.
```

## 6. B. Klasifikacijski problem: Napovedovanje uspešnice (Hit vs. Ne-Hit)

**Cilj (Y):** `Is_Hit` (Nova binarna spremenljivka: Hit vs. Ne-Hit).
**Optimizacija (Simulacija):** Napoved verjetnosti uspeha. Določanje mejnih vrednosti za ključne parametre (npr. minimalna danceability), da verjetnost uspeha preseže 90 %.

### Priprava podatkov za klasifikacijo

```python
# Definiramo "uspešnico" (Hit). Lahko je npr. popularnost > 70.
# To mejo je mogoče prilagoditi glede na distribucijo popularnosti.
hit_threshold = 70
df['Is_Hit'] = (df['popularity'] >= hit_threshold).astype(int)
print(f"Ustvarjena binarna značilka 'Is_Hit' (1 če popularnost >= {hit_threshold}, sicer 0).")
print(f"Število uspešnic: {df['Is_Hit'].sum()} ({df['Is_Hit'].mean()*100:.2f}%)")

# Uporabimo iste avdio značilke kot za regresijo
X_clf = df[audio_features].copy()
y_clf = df['Is_Hit'].copy()

# Odstranimo morebitne NaN vrednosti
initial_rows_clf = len(X_clf)
combined_clf = pd.concat([X_clf, y_clf], axis=1).dropna()
X_clf = combined_clf[audio_features]
y_clf = combined_clf['Is_Hit']
print(f"Odstranjenih {initial_rows_clf - len(X_clf)} vrstic z NaN vrednostmi za klasifikacijo.")

# Razdelitev podatkov na učno in testno množico
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf) # stratify za ohranjanje razmerja razredov

# Skaliranje značilk
scaler_clf = StandardScaler()
X_train_scaled_clf = scaler_clf.fit_transform(X_train_clf)
X_test_scaled_clf = scaler_clf.transform(X_test_clf)

print(f"Velikost učne množice (klasifikacija): {X_train_scaled_clf.shape}")
print(f"Velikost testne množice (klasifikacija): {X_test_scaled_clf.shape}")
```

### Modeli za napovedovanje uspešnic

#### 6.B.1. Logistična regresija

```python
print("\n--- Logistična regresija ---")
logistic_model = LogisticRegression(random_state=42, solver='liblinear', n_jobs=-1) # liblinear je dober za manjše datasete
# Optimizacija hiperparametrov
param_grid_logreg = {
    'C': [0.01, 0.1, 1, 10, 100], # Inverzna moč regularizacije
    'penalty': ['l1', 'l2']
}

grid_search_logreg = GridSearchCV(logistic_model, param_grid_logreg, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search_logreg.fit(X_train_scaled_clf, y_train_clf)

print(f"Najboljši parametri za Logistično regresijo: {grid_search_logreg.best_params_}")
best_logistic_model = grid_search_logreg.best_estimator_
y_pred_logistic = best_logistic_model.predict(X_test_scaled_clf)
y_prob_logistic = best_logistic_model.predict_proba(X_test_scaled_clf)[:, 1] # Verjetnosti za razred 1 (Hit)

print(f"Točnost (Accuracy): {accuracy_score(y_test_clf, y_pred_logistic):.2f}")
print(f"Preciznost (Precision): {precision_score(y_test_clf, y_pred_logistic):.2f}")
print(f"Priklic (Recall): {recall_score(y_test_clf, y_pred_logistic):.2f}")
print(f"F1-Score: {f1_score(y_test_clf, y_pred_logistic):.2f}")
print(f"ROC AUC Score: {roc_auc_score(y_test_clf, y_prob_logistic):.2f}")

# Matrika zmede
cm_logistic = confusion_matrix(y_test_clf, y_pred_logistic)
print("\nMatrika zmede (Logistična regresija):")
print(cm_logistic)

# Koeficienti logistične regresije (pomembnost značilk)
coefficients_logistic = pd.DataFrame({'Feature': audio_features, 'Coefficient': best_logistic_model.coef_[0]})
print("\nKoeficienti logistične regresije:")
print(coefficients_logistic.sort_values(by='Coefficient', ascending=False))
```

#### 6.B.2. Random Forest Classifier

```python
print("\n--- Random Forest Classifier ---")
rf_clf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
# Optimizacija hiperparametrov
param_grid_rf_clf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced'] # Pomembno za neuravnotežene razrede
}

grid_search_rf_clf = GridSearchCV(rf_clf_model, param_grid_rf_clf, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search_rf_clf.fit(X_train_scaled_clf, y_train_clf)

print(f"Najboljši parametri za Random Forest Classifier: {grid_search_rf_clf.best_params_}")
best_rf_clf_model = grid_search_rf_clf.best_estimator_
y_pred_rf_clf = best_rf_clf_model.predict(X_test_scaled_clf)
y_prob_rf_clf = best_rf_clf_model.predict_proba(X_test_scaled_clf)[:, 1]

print(f"Točnost (Accuracy): {accuracy_score(y_test_clf, y_pred_rf_clf):.2f}")
print(f"Preciznost (Precision): {precision_score(y_test_clf, y_pred_rf_clf):.2f}")
print(f"Priklic (Recall): {recall_score(y_test_clf, y_pred_rf_clf):.2f}")
print(f"F1-Score: {f1_score(y_test_clf, y_pred_rf_clf):.2f}")
print(f"ROC AUC Score: {roc_auc_score(y_test_clf, y_prob_rf_clf):.2f}")

cm_rf_clf = confusion_matrix(y_test_clf, y_pred_rf_clf)
print("\nMatrika zmede (Random Forest Classifier):")
print(cm_rf_clf)

feature_importances_rf_clf = pd.DataFrame({'Feature': audio_features, 'Importance': best_rf_clf_model.feature_importances_})
print("\nPomembnost značilk (Random Forest Classifier):")
print(feature_importances_rf_clf.sort_values(by='Importance', ascending=False))
```

#### 6.B.3. Gradient Boosting Classifier

```python
print("\n--- Gradient Boosting Classifier ---")
gb_clf_model = GradientBoostingClassifier(random_state=42)
# Optimizacija hiperparametrov
param_grid_gb_clf = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

grid_search_gb_clf = GridSearchCV(gb_clf_model, param_grid_gb_clf, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search_gb_clf.fit(X_train_scaled_clf, y_train_clf)

print(f"Najboljši parametri za Gradient Boosting Classifier: {grid_search_gb_clf.best_params_}")
best_gb_clf_model = grid_search_gb_clf.best_estimator_
y_pred_gb_clf = best_gb_clf_model.predict(X_test_scaled_clf)
y_prob_gb_clf = best_gb_clf_model.predict_proba(X_test_scaled_clf)[:, 1]

print(f"Točnost (Accuracy): {accuracy_score(y_test_clf, y_pred_gb_clf):.2f}")
print(f"Preciznost (Precision): {precision_score(y_test_clf, y_pred_gb_clf):.2f}")
print(f"Priklic (Recall): {recall_score(y_test_clf, y_pred_gb_clf):.2f}")
print(f"F1-Score: {f1_score