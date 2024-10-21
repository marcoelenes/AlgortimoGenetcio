import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
import matplotlib.pyplot as plt

# Lista de las columnas que quieres seleccionar
columnas_a_cargar = ['SDT', 'pH_CAMPO', 'OD_%', 'TEMP_AMB', 'TEMP_AGUA', 'N_TOT']  # Cambia los nombres según tus columnas

# Lee el archivo CSV y lo convierte en un DataFrame
df = pd.read_csv('C:\\Users\\Alienware X15\\Desktop\\tesis\\BDreconstruccion\\BDWeka\\BDentrenamientoWeka.csv', usecols=columnas_a_cargar)

# Muestra las primeras filas del DataFrame para verificar que se ha cargado correctamente
print(df.head())

# Asignar las variables de entrada (X) y la variable de salida (y)
X = df.drop(columns=['N_TOT'])  # Todas las columnas excepto 'N_TOT'
y = df['N_TOT']  # La columna 'N_TOT' como variable de salida

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos de entrada
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Asegurarse de que las etiquetas sean arrays unidimensionales
y_train = y_train.values
y_test = y_test.values

# Verificar las formas
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Definir el modelo Random Forest para regresión
rf = RandomForestRegressor(n_estimators=100, random_state=42)  # Puedes ajustar el número de árboles (n_estimators)

# Entrenar el modelo
rf.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred_train = rf.predict(X_train_scaled)
y_pred_test = rf.predict(X_test_scaled)

# Evaluar el modelo (usando el error cuadrático medio)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f'Error Cuadrático Medio en Entrenamiento: {mse_train}')
print(f'Error Cuadrático Medio en Prueba: {mse_test}')

# Calcular el error absoluto máximo en el conjunto de entrenamiento y prueba
max_error_train = max_error(y_train, y_pred_train)
max_error_test = max_error(y_test, y_pred_test)

print(f'Error Absoluto Máximo en Entrenamiento: {max_error_train}')
print(f'Error Absoluto Máximo en Prueba: {max_error_test}')

# Visualización de las predicciones versus los valores verdaderos
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Valor Real', color='blue')
plt.plot(y_pred_test, label='Predicción', color='red', linestyle='--')
plt.legend()
plt.xlabel('Índice de Muestras')
plt.ylabel('P_TOT')
plt.title('Comparación de Valores Reales y Predicciones (Random Forest)')
plt.show()