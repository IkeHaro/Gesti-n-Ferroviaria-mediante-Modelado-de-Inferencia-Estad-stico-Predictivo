import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# --- 1. Cargar y Limpiar Datos ---
file_path = r'C:\Users\danny\OneDrive\Escritorio\ARyC\Mod_Reg\dataset de prueba\Regularities_by_liaisons_Trains_France.csv'
df = pd.read_csv(file_path)

# --- 2. Feature Engineering (Modelo 3.0) ---

# --- A. Creación Inicial de Ruta y Fecha ---
print("Paso 2A: Creando columnas 'Date' y 'Route'...")
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(int).astype(str))
df['Route'] = df['Departure station'] + ' - ' + df['Arrival station']
df = df.sort_values(by=['Route', 'Date'])

# --- B. Definición de Columnas ---
target_cols = [
    'Delay due to external causes',
    'Delay due to railway infrastructure',
    'Delay due to traffic management',
    'Delay due to rolling stock',
    'Delay due to station management and reuse of material',
    'Delay due to travellers taken into account'
]

predictor_cols = [
    'Number of expected circulations', 
    'Average travel time (min)',     
    'Number of cancelled trains',
    'Number of late trains at departure',
    'Average delay of late departing trains (min)',
    'Number of trains late on arrival',
    'Average delay of late arriving trains (min)',
    'Number of late trains > 15min',
    'Number of late trains > 30min',
    'Number of late trains > 60min'  
]

# 'cols_for_features' ahora incluirá las 6 targets + los 10 predictores
cols_for_features = target_cols + predictor_cols
# Llenamos NaNs en todas estas columnas
df[cols_for_features] = df[cols_for_features].fillna(0)

# --- C. Features de Tendencia (Promedios Móviles) ---
print(f"Paso 2C: Creando features de tendencia para {len(cols_for_features)} columnas...")
windows = [3, 6]
rolling_features_list = []
for col in cols_for_features:
    for w in windows:
        col_name = f'{col}_roll_mean_{w}_lag_1'
        df[col_name] = df.groupby('Route')[col].rolling(window=w, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
        rolling_features_list.append(col_name)

# --- D. Features de Inmediatez (Lags simples) ---
print(f"Paso 2D: Creando features de inmediatez para {len(cols_for_features)} columnas...")
lag_1_features_list = []
for col in cols_for_features:
    col_name = f'{col}_lag_1'
    df[col_name] = df.groupby('Route')[col].shift(1)
    lag_1_features_list.append(col_name)

# --- E. Features de Tiempo y Ruta ---
print("Paso 2E: Creando features de tiempo y codificación...")
df['month_sin'] = np.sin(2 * np.pi * df['Month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['Month']/12)
le = LabelEncoder()
df['Route_encoded'] = le.fit_transform(df['Route'])
base_features = ['Route_encoded', 'month_sin', 'month_cos']

# --- 3. Definir Features (X) y Targets (y) ---
feature_cols = list(set(base_features + rolling_features_list + lag_1_features_list))
print(f"\nEl modelo usará {len(feature_cols)} features (híbridas).") # Ahora serán muchas más

# --- 4. Limpieza Final de NaNs ---
df_model = df.dropna(subset=feature_cols)

# --- 5. Filtrar solo datos PRE-COVID ---
print(f"Datos de modelo (limpios) originales: {df_model.shape[0]} filas")
df_pre_covid = df_model[df_model['Date'] < '2020-01-01'].copy()
print(f"Usando solo datos Pre-COVID (2015-2019): {df_pre_covid.shape[0]} filas")

# --- 6. Dividir los Datos (Prueba de Hipótesis: Excluyendo COVID) ---
# Entrenamos con 2015-2018, probamos con 2019
cutoff_date = '2019-01-01' 

# Dividir X e y desde el DataFrame filtrado
train_mask = df_pre_covid['Date'] < cutoff_date
test_mask = df_pre_covid['Date'] >= cutoff_date

X_train, X_test = df_pre_covid[train_mask][feature_cols], df_pre_covid[test_mask][feature_cols]
y_train, y_test = df_pre_covid[train_mask][target_cols], df_pre_covid[test_mask][target_cols]

print(f"Datos de entrenamiento (Pre-2019): {X_train.shape[0]} filas")
print(f"Datos de prueba (Solo 2019): {X_test.shape[0]} filas")

# --- 7. Entrenar el Modelo (RandomForestRegressor) ---
print("\nEntrenando el modelo (RandomForestRegressor, Features Enriquecidas)...")

base_model = RandomForestRegressor(n_estimators=100, 
                                   random_state=42, 
                                   n_jobs=-1, 
                                   min_samples_leaf=3)

multi_model = MultiOutputRegressor(base_model)
multi_model.fit(X_train, y_train)
print("¡Modelo entrenado!")

# --- 8. Evaluar el Modelo ---
predictions = multi_model.predict(X_test)

print("\n--- Evaluación del Modelo (MAE con RandomForest, Features Enriquecidas) ---")

# Calcular el Error Absoluto Medio (MAE) para las 6 variables
for i, col in enumerate(target_cols):
    mae = mean_absolute_error(y_test[col], predictions[:, i])
    print(f"Error en '{col}': {mae:.2f} puntos porcentuales")


# --- 9. Visualizar Resultados (CON ESCALA FIJA 0-30% Y 6 GRÁFICOS) ---
print("\nGenerando visualización de resultados...")
plot_data = df_pre_covid.loc[test_mask].copy() # Usar df_pre_covid

# Crear columnas de predicción para graficar
pred_df = pd.DataFrame(predictions, columns=[f'{c}_pred' for c in target_cols], index=y_test.index)
plot_data = pd.concat([plot_data, pred_df], axis=1)

route_to_plot = 'PARIS LYON - MARSEILLE ST CHARLES'
route_data = plot_data[plot_data['Route'] == route_to_plot]

if not route_data.empty:
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 15))
    
    ax_list = axes.flatten() 
    
    for i, col_name in enumerate(target_cols):
        ax = ax_list[i]
        
        ax.plot(route_data['Date'], route_data[col_name], label='Real', marker='o', linestyle='--')
        ax.plot(route_data['Date'], route_data[f'{col_name}_pred'], label='Predicción (RandomForest)', marker='x') 
        
        title = col_name.replace('Delay due to ', '').replace(' management', ' mngmt')
        ax.set_title(f'Pronóstico vs Real: {title.upper()}', fontsize=12)
        
        ax.set_ylabel('% Retraso')
        ax.legend()
        ax.set_ylim(0, 30)
        ax.set_yticks(np.arange(0, 31, 5))
        
        if i >= 4:
            ax.set_xlabel('Fecha (2019)')
            
    plt.suptitle(f'Pronóstico vs Real para la Ruta: {route_to_plot}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

else:
    print(f"\nNo se encontraron datos de prueba para la ruta '{route_to_plot}' para graficar.")