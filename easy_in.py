import torch
import cv2
import easyocr
import requests
import time
import base64
import json
import re


def load_yolo_model(model_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

def quitar_caracteres_especiales(cadena):
    # Define una expresión regular que coincide con caracteres especiales
    patron = r'[^a-zA-Z0-9\s]'
    
    # Usa la función sub() para reemplazar los caracteres especiales con una cadena vacía
    cadena_limpia = re.sub(patron, '', cadena)
    
    return cadena_limpia

def initialize_classes_and_colors(model):
    class_names = model.names[0]
    class_letras = model.names[1]
    class_numeros = model.names[2]

    class_colors = {
        'placa': (30, 224, 46),  # Rojo para la clase 'placa'
        'letras': (149, 28, 229),  # Verde para la clase 'letras'
        'numeros': (229, 130, 14)  # Azul para la clase 'numeros'
    }

    return class_names, class_letras, class_numeros, class_colors

def initialize_easyocr():
    return easyocr.Reader(['en'], gpu=False)

def initialize_video_capture():
    return cv2.VideoCapture(0)

def detect_objects(model, frame):
    detect = model(frame)
    info = detect.pandas().xyxy[0]
    info = info[info['confidence'] >= 0.80]
    return info

def draw_detections(frame, detections, class_colors, reader, save_path, confidence_threshold=0.8):
    frame_with_detections = frame.copy()

    placaDetect = {}

    for _, row in detections.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        class_name = row['name']

        frame_with_detections = cv2.rectangle(frame_with_detections, (x_min, y_min), (x_max, y_max), class_colors[class_name], 3)
        text_detection = frame[y_min:y_max, x_min:x_max]
        results = reader.readtext(text_detection)

        for (coordinates, text, prob) in results:

            #print("clase: ", class_name)
            #print("coordenadas: ", coordinates)
            #print("texto: ", text)
            #print("probabilidad: ", prob)

            placaDetect[class_name] = {}

            if prob >= 0.7:
                
                placaDetect[class_name]['texto'] = str(text)
                text = class_name + ": " + str(text)
                cv2.putText(frame_with_detections, text, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[class_name], 2)

            print(placaDetect)

            if "letras" in placaDetect and "numeros" in placaDetect:
                if "texto" in placaDetect['letras'] and "texto" in placaDetect['numeros']:
                    
                    licensePlateValue = quitar_caracteres_especiales(placaDetect['letras']['texto'] + placaDetect['numeros']['texto'])
                    
                    time.sleep(0.5)

                    #Guardamos la placa en el servidor local
                    if class_name in ["placa"] and confidence >= confidence_threshold:
                        plate_image = frame[y_min:y_max, x_min:x_max]

                        # Guardar la imagen de la placa en el directorio especificado
                        img_path = save_path + "plate_" + str(licensePlateValue) + ".png"
                        cv2.imwrite(img_path, plate_image)

                        #Despues de guardar la imagen, lo que hacemos es pasarla a base64
                        with open(img_path, 'rb') as image_file:
                            base64_bytes = base64.b64encode(image_file.read())
                            base64_string = base64_bytes.decode()
                            #print(base64_bytes)
                            #print(base64_string)

                        print("ENVIAMOS A LA API la placa: " , licensePlateValue)
                        data = {
                            "plate": licensePlateValue,
                            "image": base64_string
                        }

                        json_path = "/Users/saulmachado/Projects/Vision Por Computadora/Detector de Placas YOLOv5/asset/json/" + licensePlateValue + ".json"
                        with open(json_path, 'w') as archivo_json:
                            json.dump(data, archivo_json)

                        # Realiza una única llamada a la API para enviar los valores detectados
                        try:
                            time.sleep(2)
                            api_url = "https://0177-2800-484-ee0d-9800-19a4-f91b-282f-4309.ngrok.io" + "/api/services"  # Reemplaza con la URL de tu API
                            response = requests.post(api_url, json=data)

                            if response.status_code // 100 == 2:
                                print("Solicitud exitosa")
                                print(response.json())
                                time.sleep(0.5)


                        except requests.exceptions.RequestException as e:
                            print("Error en la solicitud:", e)
                            #time.sleep(0.5)


                        except Exception as e:
                            print("Error inesperado:", e)
                            #time.sleep(0.5)


                else:
                    print("No hay letras/numeros")
                    #time.sleep(0.5)

            else:
                print("No enviamos anda a la API")
                #time.sleep(0.5)


    return frame_with_detections

def main():
    model_path = '/Users/saulmachado/Projects/Vision Por Computadora/Detector de Placas YOLOv5/model/best_v5.pt'
    save_path = '/Users/saulmachado/Projects/Vision Por Computadora/Detector de Placas YOLOv5/asset/images/'

    model = load_yolo_model(model_path)
    class_names, class_letras, class_numeros, class_colors = initialize_classes_and_colors(model)
    reader = initialize_easyocr()
    cap = initialize_video_capture()

    while True:
        ret, frame = cap.read()
        if frame is None:
            continue
        detections = detect_objects(model, frame)
        frame_with_detections = draw_detections(frame, detections, class_colors, reader, save_path)
        if not detections.empty:
            cv2.imshow('Detector de placas', frame_with_detections)
        else:
            cv2.imshow('Detector de placas', frame)
        t = cv2.waitKey(5)
        if t == 26:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
