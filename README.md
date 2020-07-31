## YOlOv2 🚀

### Predicción de Objetos
Antes de ejecutar el detector es necesario indicarle que tipo de datos se va a procesar, para ello modificar la siguiente sección en Config.ini.

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


