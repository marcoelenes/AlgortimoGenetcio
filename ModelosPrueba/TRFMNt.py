import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Lista de las columnas que quieres seleccionar
columnas_a_cargar = ['SDT', 'pH_CAMPO', 'OD_%', 'TEMP_AMB', 'TEMP_AGUA', 'N_TOT']  # Cambia los nombres según tus columnas

# Lee el archivo CSV y lo convierte en un DataFrame
df = pd.read_csv('C:\\Users\\Alienware X15\\Desktop\\tesis\\BDreconstruccion\\BDWeka\\BDentrenamientoWeka.csv', usecols=columnas_a_cargar)

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

# Definir una clase para el Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Definir una clase para el modelo Transformer
def build_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Capa de Embedding inicial para convertir la entrada en el tamaño correcto
    x = tf.keras.layers.Dense(64)(inputs)
    
    # Añadir el bloque Transformer
    transformer_block = TransformerBlock(embed_dim=64, num_heads=2, ff_dim=64)
    x = transformer_block(x)
    
    # Flatten antes de la salida
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Capa densa final con una sola salida (regresión)
    outputs = tf.keras.layers.Dense(1)(x)
    
    # Compilar el modelo
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    
    return model

# Construir el modelo Transformer
input_shape = (X_train_scaled.shape[1], 1)  # Asumiendo que las features son tratadas como una secuencia
X_train_scaled = np.expand_dims(X_train_scaled, axis=-1)  # Expandir para que tenga la forma (batch, timesteps, feature)
X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)

model = build_transformer_model(input_shape)
model.summary()

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluar el modelo
y_pred_test = model.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f'Error Cuadrático Medio en Prueba: {mse_test}')

# Visualización de las predicciones versus los valores verdaderos
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.xlabel('Valores Verdaderos')
plt.ylabel('Predicciones')
plt.title('Valores Verdaderos vs Predicciones (Transformer)')
plt.show()
