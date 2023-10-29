#Link para descargar el modelo entrenado
https://drive.google.com/file/d/1I1xj1Ss09in01eM7Vh8WSJDUaVfovyJD/view?usp=sharing

#La estructuta del proyecto debe quedar como la imagen
./estructura.png

#Crear entorno virtual de python
> python -m venv detector_placa_v8

#activar el entorno virtual
> source detector_placa_v8/bin/activate

#desactivar entorno virtual
> deactivate

#################
#Instalar requerimientos
> pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# ojo deben corregir la linea 8
## model = torch.hub.load('ultralytics/yolov5', 'custom', path='RUTA_ABSOLUTA')
## Por el momento esta la ruta de mi PC Saul

#Ejecutar detector
> python detect.py

#NOTA: Si les da error en la linea 14, poner
#Windows => cap = cv2.VideoCapture(1)
#Linux - Mac => cap = cv2.VideoCapture(0)