# Importamos todo lo necesario
print("Cargando librerias")
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from numpy import genfromtxt
import tensorflow as tf
import numpy as np
import cv2 as cv
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# instancia del objeto Flask
app = Flask(__name__)
print("Ejecuto instancia de flask")
# Carpeta de subida
app.config['UPLOAD_FOLDER'] = "C:\\Users\\Z4RG4\\OneDrive\\Documents\\Universidad\\Machine Learning\\pythonProject\\Archivos"

#Tratamiento de la imagen: Convertirla a 28x28, blanco y negro
def pixel(img):
	img_Ga = cv.GaussianBlur(img,(7,7),0) #Filtro gaussiano
	img_g = cv.cvtColor(img_Ga, cv.COLOR_BGR2GRAY) #Convertir a escala de grises
	img_r = cv.resize(img_g,(28,28),interpolation = cv.INTER_NEAREST) #Convertir a 28x28
	img_i = cv.bitwise_not(img_r) #Invertir los colores
	return img_i

print("Cargando rutas")

@app.route("/")
def upload_file():
	return render_template('index.html')

@app.route("/upload", methods=['POST'])
def uploader():
 if request.method == 'POST':
  # obtenemos el archivo del input "archivo"
  f = request.files['archivo']
  filename = secure_filename(f.filename)
  # Guardamos el archivo en el directorio "Archivos PDF"
  f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
  # Retornamos una respuesta satisfactoria

  # Test del modelo
  imgDir = "C:\\Users\\Z4RG4\\OneDrive\\Documents\\Universidad\\Machine Learning\\pythonProject\\Archivos\\"+filename
  

  img = cv.imread(imgDir) #Convertir img a un objeto de open cv
  img1 = pixel(img) #Imagen con tratamiento
  img2 = (np.expand_dims(img1,0))
  result = model.predict_classes(img2) #Predecir categoria

  print(class_names[result[0]])
  print(result[0])
  datos = class_names[result[0]]
  return redirect(url_for("clasificar",datos=datos))



@app.route("/clasificar/<datos>")
def clasificar(datos):
	return render_template("clasificar.html", data=datos)


# Iniciamos la aplicación
print("Cargando main")

if __name__ == '__main__':
	print("Iniciando app")
	fashion_mnist = tf.keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	class_names = ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat',
	           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	train_images = train_images / 255.0
	test_images = test_images / 255.0

	# Construccion del modelo
	print("Construyendo el modelo")
	model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(10, activation='sigmoid'),
	tf.keras.layers.Dense(10, activation='softmax')
	])

	# Compilar el modelo
	print("Compilando el modelo")
	model.compile(optimizer='adam',
	          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])
	# Entrenando el modelo
	print("Entrenando el modelo ")
	model.fit(train_images, train_labels, epochs=30)
	print("Modelo entrenado")
	
	app.run(debug=False)
