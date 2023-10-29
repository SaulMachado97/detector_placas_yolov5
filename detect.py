#importamos librerias
import torch
import cv2
import numpy as np
import pandas

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/saulmachado/Projects/Vision Por Computadora/Detector de Placas YOLOv5/model/best.pt')

# Nombres de las clases
class_names = model.names[0]

#Realizamos videocaptura
cap = cv2.VideoCapture(0)

#Empezamos
while True:
    #Realizamos la lectura de la videocaptura
    ret, frame = cap.read()

    # Verificar si se ha capturado un frame válido
    if frame is None:
        continue  # Salta este bucle y vuelve a intentar la siguiente captura

    #Realizamos detecciones
    detect = model(frame)

    info = detect.pandas().xyxy[0]
    print(info)
    info = info[info['confidence'] >= 0.70]

    #Mostramos los FPS
    #cv2.imshow('Detector de placas', np.squeeze(detect.render()))

    # Crear una copia del marco original para mostrar solo las detecciones con confianza
    frame_with_detections = frame.copy()

    for _, row in info.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        class_id = int(row['class'])
        class_name = class_names
        label = f"{class_name}: {confidence:.2f}"

        frame_with_detections = cv2.rectangle(frame_with_detections, (x_min, y_min), (x_max, y_max), (30, 224, 46), 3)
        frame_with_detections = cv2.putText(frame_with_detections, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 224, 46), 2)

    # Mostrar el marco con detecciones solo si hay detecciones válidas
    if not info.empty:
        cv2.imshow('Detector de placas', frame_with_detections)
    else:
        cv2.imshow('Detector de placas', frame)

    #Leemos el teclado
    t = cv2.waitKey(5)
    if t == 26:
        break

cap.release()
cv2.destroyAllWindows()
