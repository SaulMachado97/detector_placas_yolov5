import torch
import cv2
import pandas as pd
import easyocr

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/saulmachado/Projects/Vision Por Computadora/Detector de Placas YOLOv5/model/best_v5.pt')

# Nombres de las clases
class_names = model.names[0]
class_letras = model.names[1]
class_numeros = model.names[2]

# Diccionario que asocia cada clase con un color
class_colors = {
    'placa': (30, 224, 46),  # Rojo para la clase 'placa'
    'letras': (149, 28, 229),  # Verde para la clase 'letras'
    'numeros': (229, 130, 14)  # Azul para la clase 'numeros'
}

# Inicializar EasyOCR
reader = easyocr.Reader(['en'], gpu=False)  # Puedes especificar otros idiomas según tus necesidades

#definimos variables para almacenar mejores resultados de placas
best_letras = ""
best_confidence_letras = 0.0
best_numeros = ""
best_confidence_numeros = 0.0

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
        #print("Valores de Row: ", row)
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        class_id = int(row['class'])
        class_name = row['name']
        label = f"{class_name}: {confidence:.2f}"

        frame_with_detections = cv2.rectangle(frame_with_detections, (x_min, y_min), (x_max, y_max), class_colors[class_name], 3)
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
                text = class_name + ": " + str(text)  # Asegúrate de que "text" sea una cadena
                cv2.putText(frame_with_detections, text, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[class_name], 2)

            # Verifica si la confianza del texto actual es mayor que la confianza del mejor texto anterior
            #if prob > best_confidence:
            #    best_text = text
            #    best_confidence = prob


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
