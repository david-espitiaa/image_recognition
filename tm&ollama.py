#=======================================================
# Utilizando LlaVa:7b en Ollama para reconocer las 
# imagenes que no coinciden con la clases de la red de TM
#
# Para probar la red, usar cualquier de las imagens de  
# comida, y para probar Ollama usar la de "pc.jpeg"
#=======================================================

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from ollama import chat
import matplotlib.pyplot as plt


# CARGAR MODELO Y ETIQUETAS
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# PROCESAR IMAGEN
image_path = "pc.jpeg" #inserte el path de la imagen

image = Image.open(image_path).convert("RGB")
image_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image_resized)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# HACER PREDICCIÓN
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]


# DETERMINAR TEXTO A MOSTRAR
if confidence_score >= 0.8:
    display_text = f"Predicción: {class_name} ({confidence_score*100:.2f}%)"
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
else:
    response = chat(model='llava:7b', messages=[
        {
            'role': 'user',
            'content': 'very shortly, what is in this image: ',
            'images': [image_path]
        },
    ])
    display_text = f"Respuesta de Ollama en la terminal..."
    print(response['message']['content'])

# MOSTRAR IMAGEN CON RESULTADO
plt.imshow(image)
plt.axis('off')
plt.title(display_text, fontsize=12)
plt.show()
