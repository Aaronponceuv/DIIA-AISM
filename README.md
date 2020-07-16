## YOlOv2 🚀

A continuación se exponen los pasos para el entrenamiento, detección y validación de Yolov2 en pytorch.
Para ello realizar los siguientes instrucciones


### Creación de Custom Dataset 📋
La conformación de Dataset debe tener la siguiente estructura
```
Dataset(Directorio Raiz)
|-->Images
|   |--->imagenes
|        |--img0.jpg
|        |--img1.jpg
|        |--etc 
|   |--->labels
|        |--label_img0.txt
|        |--label_img1.txt
|        |--etc
|---Train.txt
|---Val.txt
```
### Creación de listado de Clases del Dataset
Crear un archivo de configuración en data/nombre_del_archivo_de_clases.names
Escribir el nombre de las clases de objetos linea por linea

**Atencion:** Al colocar cada clase preocuarar que se encuentra en el orden correcto, de acuerdo al formato YOLO

```
camion
grua
persona
```
### Creación de archivo de Arquitectura 🔧
Crear archivo de arquitectura de red e hiperparametros en cfg/ 
Se recomienda copiar y cambiar de nombre el archivo cfg/yolo-camion-darknet.cfg colocando un nombre referente al custom dataset 

### Entrenamiento de Yolo con Custom Dataset
Crear archivo de configuracion en cfg/nombre_del_archivo.data
en cada item colocar ruta absoluta de cada archivo, en names colocar el archivo creado en el paso anterior.
```
train  = /home/multimind/Dataset/train.txt
valid  =  /home/multimind/Dataset/Val.txt
names = data/voc.names
backup = backup
```
### Archivo de Configuración 🛠️
Una vez realizado los pasos anteriores solo queda configurar el archivo config.ini, con las rutas de los datos creados.

#### Entrenamiento 

Para entrenar modificar las siguientes rutas con los archivos de configuración YOLO
```
[Train_Parameters]
Configuracion utilizada en train.py
cfgdata = ./cfg/camion.data
cfgfile = ./cfg/yolo-camion-darknet.cfg
weights_conv = darknet19_448.conv.23
```
Para entrenar modificar valor de count a 4 en la siguiente sección
```
[Load_Weight]
count = X 
```
Una vez realizado lo anterior escribir el siguiente comando para entrenar
```
python3 train.py
```

#### Detección :movie_camera:
Para utilizar la detección con la red entrenada o con otros pesos es necesario modificar los siguientes parametros segun el caso
esta sección considera rutas de almacenamiento de datos y lectura de archivos de configuración de YOLO
```
[Detect_Parameters]
;Carpeta de almacenamiento de Frames con box detectados
folder_save = ./video-predict
;Archivo de clases
namesfile = data/camion.names
;Carpeta que almacena registros de deteccion en un frame 
registro_frame = ./registro_frame
;Carpeta de almacenamiento de videos
registro_video = ./video-box
;Arquitectura de la red utilizada
cfgfile = ./cfg/yolo-camion-darknet.cfg
;Pesos de la red
weightfile = ./backup/yolo-camion_14000.weights
;Umbral de confianza para deteccion
conf_thresh   = 0.5
;Supresión no máxima para selecciona del mejor box
nms_thresh    = 0.4
```
### Predicción de Objetos
Antes de ejecutar el detector es necesario indicarle que tipo de datos se va a procesar, para ello modificar la siguiente sección.

En option colocar 
+   video: Si se desea procesar **un** video
+   img:  Si se desea procesar **una** imagen
+   dimg: Si se desea procesar directorio con imagenes
+   dvideo: Si se desea procesar directorio de videos

Ejemplo de configuración para detectar **un** video en la ruta ./video-cam/camion-minuto.mp4
```
[Option_predicts]
option = video
imgfile = ./video-cam/camion-minuto.mp4
```
Para predecir escribir en consola
```
python3 detect.py
```
### Validación
Para realizar las pruebas de rendimiento ejeuctar el siguiente comando

```
python3 valid.py
```
El comando anterior generara un archivo en la carpeta result con las predicciones de cada imagen presente en el conjunto de validación, en este archivo se encuentran las cordenadas que arrojaron la mayor probabilidad en cada imagen.

Una vez ejecutado el comando anterior, ejecutar el siguiente
```
python3 testing.py
```
Este comando arrojara el mAP del rendimiento de la red, ademas genera un archivo Resultado_Test.xlsx con la informacion de cada imagen

imagen |	prob | class | iou	| TP/FP |	Precision |	Recall | IP
---    |---      |---    |---   |---    |---          |-----   | ---
0-847  |	0.97 |	0	 | 0.89	| TP    |	1         |0.01    |1
2-1320 |	0.97 |	0	 | 0.91	| TP  	|   1	      |0.01	   |1
0-1064 |	0.95 |	0	 | 0.69	| TP	|   1	      |0.02	   |1
2-1959 |    0.96 |  0	 | 0.78	| TP	|   1	      |0.02	   |1
0-846  |	0.97 |	0	 | 0.87	| TP	|   1	      |0.03	   |1
2-1338 |	0.97 |	0	 | 0.89	| TP	|   1 	      |0.03	   |1
2-2199 |	0.95 |	0	 | 0.86	| TP	|   1   	  |0.04	   |1
0-3	   |    0.89 |	0	 | 0.79	| TP	|   1	      |0.05	   |1
2-2316 |	0.94 |	0	 | 0.79	| TP	|	1         |0.05	   |1

### Referencia
Este proyecto tiene como base el proyecto de Marvis: https://github.com/marvis
#### pytorch-yolo2
Convert https://pjreddie.com/darknet/yolo/ into pytorch. This repository is trying to achieve the following goals.
- [x] implement RegionLoss, MaxPoolStride1, Reorg, GolbalAvgPool2d
- [x] implement route layer
- [x] detect, partial, valid functions
- [x] load darknet cfg
- [x] load darknet saved weights
- [x] save as darknet weights
- [x] fast evaluation
- [x] pascal voc validation
- [x] train pascal voc
- [x] LMDB data set
- [x] Data augmentation
- [x] load/save caffe prototxt and weights
- [x] **reproduce darknet's training results**
- [x] [convert weight/cfg between pytorch caffe and darknet](https://github.com/marvis/pytorch-caffe-darknet-convert)
- [x] add focal loss


