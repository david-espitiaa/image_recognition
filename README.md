# 🧠 Webcam Smart Predictor con TensorFlow y Ollama (LLaVA)

Este proyecto combina **visión por computadora** con **inteligencia artificial conversacional** para analizar imágenes capturadas por la webcam:

- 📸 Usa un modelo Keras para clasificar imágenes,  la cual se creo usando [Teachable Machine](https://teachablemachine.withgoogle.com/).
- 🤖 Si el modelo no está seguro, pregunta a [LlaVa](hhttps://llava-vl.github.io/) qué hay en la imagen.
- 🧠 Se consulta a [Ollama](https://ollama.com/), una herramienta que permite correr LLM´s localment

---

## 🚀 Requisitos

- 📕 Librerías:
- Tensorflow **2.12.1 (Solo funciona con esta versión)**
- numpy
- pillow
- matplot
- opencv
- ollama 
- LlaVa:7b

---

# 🛠️ Cambios En La Red

Si se desea cambiar las clases que detecta la red, se debe crear una nueva en [Teachable Machine](https://teachablemachine.withgoogle.com/) y exportar el modelo,solamente se deben de modificar los archivos **keras_model.h5 y labels.txt** por los nuevos obtenidos.
