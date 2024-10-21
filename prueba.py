import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, max_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
# Ruta del archivo Excel
ruta_archivo = "C://Users//Alienware X15//Desktop//tesis//BDlimpio.xlsx"
#df = pd.read_excel(ruta_archivo,sheet_name="Valores")

# Leer el archivo Excel y seleccionar solo las columnas "A", "C", y "E"
df = pd.read_excel(ruta_archivo,usecols=["pH_CAMPO", "TEMP_AGUA", "TEMP_AMB", "OD_%", "SDT","DQO_TOT","P_TOT","DBO_TOT","COLI_FEC","E_COLI",])

# Mostrar las primeras filas del DataFrame
print(df.head())
# Inicializar el escalador Min-Max
scaler = MinMaxScaler()

# Crear una copia del DataFrame para aplicar la normalización
df_normalizado = df.copy()

# Normalizar las columnas específicas del DataFrame
df_normalizado['OD_%'] = scaler.fit_transform(df_normalizado[['OD_%']])
df_normalizado['P_TOT'] = scaler.fit_transform(df_normalizado[['P_TOT']])
df_normalizado['SDT'] = scaler.fit_transform(df_normalizado[['SDT']])
df_normalizado['TEMP_AGUA'] = scaler.fit_transform(df_normalizado[['TEMP_AGUA']])
df_normalizado['TEMP_AMB'] = scaler.fit_transform(df_normalizado[['TEMP_AMB']])
df_normalizado['pH_CAMPO'] = scaler.fit_transform(df_normalizado[['pH_CAMPO']])
#df_normalizado['ORTO_PO4'] = scaler.fit_transform(df_normalizado[['ORTO_PO4']])
df_normalizado['DQO_TOT'] = scaler.fit_transform(df_normalizado[['DQO_TOT']])
df_normalizado['DBO_TOT'] = scaler.fit_transform(df_normalizado[['DBO_TOT']])
df_normalizado['COLI_FEC'] = scaler.fit_transform(df_normalizado[['COLI_FEC']])
df_normalizado['E_COLI'] = scaler.fit_transform(df_normalizado[['E_COLI']])



# Guardar el DataFrame normalizado en un archivo Excel
#df_normalizado.to_excel('df_normalizado.xlsx', index=False)

# Mostrar las primeras filas del DataFrame normalizado
print(df_normalizado.head())
# Asignar las variables de entrada (X) y la variable de salida (y)
X = df_normalizado.drop(columns=['P_TOT'])  # Todas las columnas excepto 'P_TOT'
y = df_normalizado['P_TOT']  # La columna 'P_TOT' como variable de salida
# Paso 1: Dividir los datos en 70% entrenamiento y 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Paso 2: Dividir el conjunto de entrenamiento en 70% para entrenamiento final y 30% para validación
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)

# Asegurarse de que las etiquetas sean arrays unidimensionales
y_train_final = y_train_final.values
y_test = y_test.values
y_val = y_val.values

# Verificar las formas
print("X_train_final shape:", X_train_final.shape)
print("X_test shape:", X_test.shape)
print("X_val shape:", X_val.shape)
print("y_train_final shape:", y_train_final.shape)
print("y_test shape:", y_test.shape)
print("y_val shape:", y_val.shape)

# Definir el modelo Random Forest
rf = RandomForestRegressor(n_estimators=339, 
                           max_depth=19, 
                           min_samples_split=2, 
                           min_samples_leaf=1, 
                           max_features='sqrt', 
                           random_state=42)

# Entrenar el modelo con X_train_final y y_train_final
rf.fit(X_train_final, y_train_final)

# Realizar predicciones en el conjunto de validación
y_pred_val = rf.predict(X_val)

# Evaluar el modelo con el conjunto de validación
mse_val = mean_squared_error(y_val, y_pred_val)
print(f'Error Cuadrático Medio en Validación: {mse_val}')

# Calcular el error absoluto máximo en el conjunto de validación
max_error_val = max_error(y_val, y_pred_val)
print(f'Error Absoluto Máximo en Validación: {max_error_val}')

# Una vez ajustado, evaluamos el modelo con el conjunto de test
y_pred_test = rf.predict(X_test)

# Evaluar el modelo con el conjunto de test
mse_test = mean_squared_error(y_test, y_pred_test)
print(f'Error Cuadrático Medio en Prueba: {mse_test}')

# Calcular el error absoluto máximo en el conjunto de prueba
max_error_test = max_error(y_test, y_pred_test)
print(f'Error Absoluto Máximo en Prueba: {max_error_test}')

# Visualización de las predicciones versus los valores verdaderos en el conjunto de prueba
plt.figure(figsize=(10, 6))
plt.plot(y_test[0:100], label='Valor Real', color='blue')
plt.plot(y_pred_test[0:100], label='Predicción', color='red', linestyle='--')
plt.legend()
plt.xlabel('Índice de Muestras')
plt.ylabel('P_TOT')
plt.title('Comparación de Valores Reales y Predicciones (Random Forest)')
plt.show()