# Utiliza una imagen base de TensorFlow con soporte para GPU
FROM tensorflow/tensorflow:2.11.0-gpu

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /code

# Copia los archivos de la aplicación dentro del contenedor
COPY . .

# Instala las dependencias desde requirements.txt
COPY requirements.txt . 
RUN pip install -r requirements.txt

# Comando para ejecutar el script Python
CMD ["python", "Nt.py"]
