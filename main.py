import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado (puedes usar 'yolov8n.pt' o cambiar a 'yolov8s.pt', 'yolov8m.pt', etc.)
model = YOLO('yolov8n.pt')

# Iniciar la captura de video desde la webcam
cap = cv2.VideoCapture(0)  # '0' es el índice de la cámara web, ajusta si es necesario

while cap.isOpened():
    ret, frame = cap.read()  # Leer el frame de la cámara

    if not ret:
        print("No se pudo capturar la imagen de la cámara.")
        break

    # Realizar la predicción usando el modelo YOLOv8
    results = model(frame)

    # Dibujar las predicciones sobre la imagen
    annotated_frame = results[0].plot()

    # Mostrar la imagen con las predicciones
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
