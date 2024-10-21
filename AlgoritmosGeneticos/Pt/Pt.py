import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, max_error
from sklearn.preprocessing import MinMaxScaler


# Ruta del archivo Excel
ruta_archivo = "BDlimpio.xlsx"
#df = pd.read_excel(ruta_archivo,sheet_name="Valores")

# Leer el archivo Excel y seleccionar solo las columnas "A", "C", y "E"
df = pd.read_excel(ruta_archivo,usecols=["pH_CAMPO", "TEMP_AGUA", "TEMP_AMB", "OD_%", "SDT","P_TOT","DQO_TOT","DBO_TOT","COLI_FEC","E_COLI"],)

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
df_normalizado['DQO_TOT'] = scaler.fit_transform(df_normalizado[['DQO_TOT']])
df_normalizado['DBO_TOT'] = scaler.fit_transform(df_normalizado[['DBO_TOT']])
df_normalizado['COLI_FEC'] = scaler.fit_transform(df_normalizado[['COLI_FEC']])
df_normalizado['E_COLI'] = scaler.fit_transform(df_normalizado[['E_COLI']])

# Asignar las variables de entrada (X) y la variable de salida (y)
X = df_normalizado.drop(columns=['P_TOT'])  # Todas las columnas excepto 'P_TOT'
y = df_normalizado['P_TOT']  # La columna 'P_TOT' como variable de salida

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Guardar el DataFrame normalizado en un archivo Excel
#df_normalizado.to_excel('df_normalizado.xlsx', index=False)

# Mostrar las primeras filas del DataFrame normalizado
print(df_normalizado.head())


# Lista de los nombres de las columnas (parámetros de entrada)
column_names = X_train.columns.tolist()

# Definir la función de fitness
def fitness_function(individual, X_train, y_train, X_test, y_test):
    # Extraer el vector de hiperparámetros y selección de características
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features_bin = individual[:5]
    selected_columns = individual[5]  # Lista de columnas seleccionadas
    
    # Seleccionar solo las columnas indicadas por el individuo
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]
    
    # Definir el modelo Random Forest con los hiperparámetros del individuo
    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features='sqrt' if max_features_bin > 0.5 else None,
        random_state=42
    )
    
    # Entrenar el modelo
    rf.fit(X_train_selected, y_train)
    
    # Realizar predicciones
    y_pred_test = rf.predict(X_test_selected)
    
    # Evaluar el modelo usando el error cuadrático medio (MSE)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    return -mse_test,mse_test  # Usamos el negativo porque el algoritmo genético maximiza la fitness

# Definir los parámetros del algoritmo genético
population_size = 20
generations = 5
mutation_rate = 0.1
crossover_rate = 0.7

# Generar un individuo aleatorio (hiperparámetros + selección de columnas)
def generate_individual():
    n_estimators = random.randint(50, 1000)
    max_depth = random.randint(5, 30)
    min_samples_split = random.randint(2, 10)
    min_samples_leaf = random.randint(1, 4)
    max_features_bin = random.random()  # Usamos un valor binario para seleccionar sqrt o None
    
    # Seleccionar un subconjunto aleatorio de columnas
    selected_columns = random.sample(column_names, random.randint(1, len(column_names)))
    
    return [n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features_bin, selected_columns]

# Generar la población inicial
population = [generate_individual() for _ in range(population_size)]

# Proceso del algoritmo genético
for generation in range(generations):
    print(f"Generación {generation + 1}")
    # Evaluar la fitness de cada individuo
    fitness_scores = [fitness_function(ind, X_train, y_train, X_test, y_test) for ind in population]
    
    # Selección de los mejores individuos
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    population = sorted_population[:population_size]  # Mantener solo los mejores
    
    # Operadores genéticos: cruce y mutación
    new_population = []
    
    for i in range(0, population_size, 2):
        parent1 = population[i]
        parent2 = population[i + 1] if i + 1 < population_size else population[0]
        
        # Cruzamiento
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 2)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1, parent2
        
        # Mutación en hiperparámetros
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, 4)  # Solo mutamos en los primeros 5 elementos (hiperparámetros)
            child1[mutation_point] = generate_individual()[mutation_point]
        
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, 4)
            child2[mutation_point] = generate_individual()[mutation_point]
        
        # Mutación en selección de columnas
        if random.random() < mutation_rate:
            child1[5] = random.sample(column_names, random.randint(1, len(column_names)))
        
        if random.random() < mutation_rate:
            child2[5] = random.sample(column_names, random.randint(1, len(column_names)))
        
        new_population.append(child1)
        new_population.append(child2)
    
    population = new_population

# Al final, el mejor individuo
best_individual = population[0]
best_fitness,best_mse = fitness_function(best_individual, X_train, y_train, X_test, y_test)

print("Mejor individuo:", best_individual)
print("Mejor fitness:", best_fitness)
print(f"porcentaje del acuracy del modelo: {best_mse}")
resultado = ((best_mse**0.5)*2)
print(f"Cuanto se equivoco % se equivoco nuestro modelo {resultado*100}%")
# Guardar el mejor modelo en un archivo .pkl usando joblib
def save_model(model, file_name):
    joblib.dump(model, file_name)
    file_path = os.path.join(os.getcwd(), file_name)
    print(f'Modelo guardado en: {file_path}')

# Guardar el mejor modelo encontrado
best_rf_model = RandomForestRegressor(
    n_estimators=int(best_individual[0]),
    max_depth=int(best_individual[1]),
    min_samples_split=int(best_individual[2]),
    min_samples_leaf=int(best_individual[3]),
    max_features='sqrt' if best_individual[4] > 0.5 else None,
    random_state=42
)

# Volver a entrenar el modelo con el mejor conjunto de características seleccionadas
X_train_selected = X_train[best_individual[5]]
X_test_selected = X_test[best_individual[5]]

best_rf_model.fit(X_train_selected, y_train)


# Guardar el modelo entrenado
save_model(best_rf_model, "mejor_modelo_rf.pkl")

# Cargar el modelo guardado en el futuro
# loaded_model = joblib.load("mejor_modelo_rf.pkl")

print("Modelo y pesos guardados exitosamente.")