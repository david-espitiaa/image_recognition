#=====================================================
# Probando la neurona de reconocimiento hecha con 
# Techable Machine, pero utilizando la camara y OpenCV
#=====================================================

from keras.models import load_model
import cv2
import numpy as np

# Desactiva la notación científica
np.set_printoptions(suppress=True)

# Carga el modelo
model = load_model("keras_Model.h5", compile=False)

# Carga las etiquetas
class_names = open("labels.txt", "r").readlines()

# Selección de cámara (puede ser 0 o 1 dependiendo del dispositivo)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, image = cap.read()
    if not ret:
        print("No se pudo capturar imagen")
        continue

    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Preprocesamiento de imagen
    image_np = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    image_np = (image_np / 127.5) - 1

    # Predicción
    prediction = model.predict(image_np)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Mostrar la predicción y la confianza en consola
    print("Class:", class_name)
    print("Confidence Score:", str(np.round(confidence_score * 100)) + "%")

    # Dibujar texto en la imagen original (no redimensionada)
    display_text = f"{class_name}: {confidence_score * 100:.2f}%"
    cv2.putText(
        image,                   # imagen base
        display_text,           # texto a mostrar
        (30, 50),               # coordenadas (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        1,                      # tamaño de fuente
        (0, 255, 0),            # color (B, G, R)
        2,                      # grosor de línea
        cv2.LINE_AA
    )

    # Mostrar la imagen con el texto
    cv2.imshow("Webcam Image", image)

    # Salir con ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
