import torch
import cv2
import pandas as pd
import easyocr

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/saulmachado/Projects/Vision Por Computadora/Detector de Placas YOLOv5/model/best.pt')

# Nombres de las clases
class_names = model.names[0]

# Inicializar EasyOCR
reader = easyocr.Reader(['en'], gpu=False)  # Puedes especificar otros idiomas según tus necesidades

# Realizamos videocaptura
cap = cv2.VideoCapture(0)

# Empezamos
while True:
    # Realizamos la lectura de la videocaptura
    ret, frame = cap.read()

    # Verificar si se ha capturado un frame válido
    if frame is None:
        continue  # Salta este bucle y vuelve a intentar la siguiente captura

    # Realizamos detecciones
    detect = model(frame)

    info = detect.pandas().xyxy[0]
    #print(info)
    info = info[info['confidence'] >= 0.70]

    # Crear una copia del marco original para mostrar solo las detecciones con confianza
    frame_with_detections = frame.copy()

    for _, row in info.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        class_id = int(row['class'])
        class_name = class_names
        label = f"{class_name}: {confidence:.2f}"

        frame_with_detections = cv2.rectangle(frame_with_detections, (x_min, y_min), (x_max, y_max), (30, 224, 46), 3)
        #frame_with_detections = cv2.putText(frame_with_detections, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 224, 46), 2)

        # Detectar texto usando EasyOCR en el área enmarcada
        text_detection = frame[y_min:y_max, x_min:x_max]
        results = reader.readtext(text_detection)

        # Mostrar el texto detectado
        #for (text, _, prob) in results:
        #    if prob >= 0.7:  # Puedes ajustar el umbral de confianza según tus necesidades
        #        cv2.putText(frame_with_detections, text, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 224, 46), 2)

        for (coordinates, text, prob) in results:
        #for result in results:
            print("coordenadas: ", coordinates)
            print("texto: ", text)
            print("probabilidad: ", prob)

            if prob >= 0.7:  # Ajusta el umbral de confianza según tus necesidades
                text = str(text)  # Asegúrate de que "text" sea una cadena
                cv2.putText(frame_with_detections, text, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 224, 46), 2)


    # Mostrar el marco con detecciones solo si hay detecciones válidas
    if not info.empty:
        cv2.imshow('Detector de placas', frame_with_detections)
    else:
        cv2.imshow('Detector de placas', frame)

    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 26:
        break

cap.release()
cv2.destroyAllWindows()
