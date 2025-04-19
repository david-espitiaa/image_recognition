#=======================================================
# Utilizando LlaVa:7b en Ollama para reconocer las 
# imagenes que no coinciden con la clases de la red 
# que son capturadas por medio de OpenCV
#=======================================================


from keras.models import load_model
import cv2
import numpy as np
from ollama import chat
import os
import time

# Configuración
CONFIDENCE_THRESHOLD = 0.7
TEMP_IMG_PATH = "temp_ollama_img.jpg"
PREDICTION_INTERVAL = 5  # segundos

# Cargar modelo y etiquetas
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Iniciar cámara
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
last_prediction_time = 0
last_low_confidence_frame = None
last_ollama_response_time = 0
display_text = "Esperando próxima predicción..."

while True:
    ret, image = cap.read()
    if not ret:
        print("No se pudo capturar imagen")
        continue

    current_time = time.time()

    if current_time - last_prediction_time >= PREDICTION_INTERVAL:
        img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_np = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        image_np = (image_np / 127.5) - 1

        # Predicción
        prediction = model.predict(image_np)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        if confidence_score >= CONFIDENCE_THRESHOLD:
            display_text = f"{class_name}: {confidence_score * 100:.2f}%"
        else:
            # Evitar múltiples consultas a Ollama si la imagen no ha cambiado mucho
            if last_low_confidence_frame is None or not np.array_equal(image, last_low_confidence_frame):
                # Guardar imagen y preguntar a Ollama
                cv2.imwrite(TEMP_IMG_PATH, image)
                display_text = "Consultando a Ollama..."
                cv2.putText(image, display_text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Webcam Image", image)
                cv2.waitKey(1)

                response = chat(model='llava:7b', messages=[
                    {
                        'role': 'user',
                        'content': 'shortly, what is in this image: ',
                        'images': [TEMP_IMG_PATH]
                    },
                ])
                print(response['message']['content'])
                display_text = f"Ollama: {response['message']['content']}"
                os.remove(TEMP_IMG_PATH)

                last_low_confidence_frame = image.copy()
                last_ollama_response_time = current_time
            else:
                display_text = "Esperando nueva imagen para consultar a Ollama..."

        last_prediction_time = current_time

    # Mostrar texto en pantalla
    cv2.putText(
        image, display_text, (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
    )

    cv2.imshow("Webcam Image", image)

    # Salir con ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
