import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import os
import cv2
import filetype
from shutil import rmtree

import cv2
import numpy as np
import glob

#from Utilhora import *
import pandas as pd

import time
from pathlib import Path
import shutil
def ordenerar(path):
    imgs = os.listdir(path)
    name_video = []
    ruta = []
    for i in range(len(imgs)):
        name_video.append(int(imgs[i].split(".")[0]))
    name_video = sorted(name_video)
    return name_video
 
def convert_img_video(carpeta_save,fps):
    img_array = []
    path = carpeta_save
    imgs = ordenerar(path)
    for filename in range(len(imgs)):
        print('---'+path+"/"+str(imgs[filename]))
        img = cv2.imread(path+"/"+str(imgs[filename])+".jpg")
        height, width, layer = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter("./video-box/"+path.split("/")[2]+'-prediccion.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def formato_hora(hora):
    hrs = hora[0:2]
    if(hrs[0] == '0'):
        hrs = hrs[1]
    min = hora[2:4]
    seg = hora[4:6]
    return str(hrs+':'+min+':'+seg)

def input_video(model,path,namesfile,carpeta_save,use_cuda):
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print('fps',fps)

    fecha = path.split("_")[1].split(".")[0][0:8]
    fecha_formato_reporte = str(fecha[6:8]+'.'+fecha[4:6]+'.'+fecha[0:4])
    hora_ruta = path.split("_")[1].split(".")[0][8:14]
    hora_inicio = formato_hora(hora_ruta)
    print('fecha: ',fecha,' fecha formato', fecha_formato_reporte, 'hora_ruta',hora_ruta, 'hora inicio',hora_inicio)

    frameCount = 0
    tren = []
    hora = []
    frames = []
    data = pd.DataFrame(columns=('frame','Deteccion','Fecha-Hora'))
    tiempo_de_prediccion = []

    while(1):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(img)
        sized = frame.resize((model.width, model.height))

        for i in range(2):
            start = time.time()
            boxes = do_detect(model, np.array(sized), 0.5, 0.4, use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (path, (finish - start)))

        class_names = load_class_names(namesfile)
        plot_boxes(frame.resize((640,360)), boxes,carpeta_save+"/"+str(frameCount)+'.jpg', class_names, frameCount)
        

        if(len(boxes) == 1):
            tren.append(1)
            data.loc[frameCount] = [frameCount,"SI",str(time.ctime())] 
        if(len(boxes) == 0):
            tren.append(0)
            data.loc[frameCount] = [frameCount,"NO",str(time.ctime())] 
        frameCount +=1
        
        """
        if(sum(tren[len(tren)-40:len(tren)])>=1):
            hora_detect = milisegundos_a_hora(cap.get(cv2.CAP_PROP_POS_MSEC))
            
          #  print("Deteccion de camion, frame ",frameCount,' hora detect',hora_detect[0:6],'hora grab',sumar_hora(hora_detect,hora_inicio) )
            hora.append(sumar_hora(hora_inicio,hora_detect[0:6]))
            frames.append(frameCount)

        if (sum(tren[len(tren)-40: len(tren)]) == 0 and sum(tren[len(tren)-80: len(tren)-40]) >= 1 ):
        #    print("se fue el camion",frameCount)
            hora_detect2 = milisegundos_a_hora(cap.get(cv2.CAP_PROP_POS_MSEC))
            data.loc[len(data)]=[hora[0],frames[0],sumar_hora(hora_inicio,hora_detect2[0:6]),frameCount] 
        """    

            #del tren[:], hora[:], frames[:]
    print(path.split("/")[7].split(".")[0])
    name_xlsx = path.split("/")[7].split(".")[0]
    data.to_excel("./registro_frame/"+str(name_xlsx)+'.xlsx', sheet_name='detectados' , index=False)
    print(fps)
    print('Fotogramas Procesados',frameCount)
    return fps

def input_dir_image(model, path, namesfile,use_cuda):
    imagenes = os.listdir(path)
    tiempo_de_prediccion = []
    folder_predictions = str(path)+'-predictions'
    if(os.path.isdir(folder_predictions) == False):
        os.mkdir(folder_predictions)
    for z in range(len(imagenes)):
        imgfile = path+'/'+str(imagenes[z])
        img = Image.open(imgfile).convert('RGB')
        sized = img.resize((model.width, model.height))
        for i in range(2):
            start = time.time()
            boxes = do_detect(model, sized, 0.5, 0.4, use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

        class_names = load_class_names(namesfile)
        plot_boxes(img, boxes, folder_predictions+'/'+str(imagenes[z]).split(".")[0]+'-predictions.jpg', class_names)
        tiempo_de_prediccion.append((finish - start))
    print('Tiempo de Procesamiento:',sum(tiempo_de_prediccion))

def input_image(model,path,namesfile,use_cuda):
    img = Image.open(path).convert('RGB')
    sized = img.resize((model.width, model.height))
    for i in range(2):
        start = time.time()
        boxes = do_detect(model, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (path, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)


def detect(cfgfile, weightfile, path, formato,carpeta_save=None):
    inicio = time.time()
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    print('Tiempo de Carga de Pesos', time.time()-inicio)
    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/camion.names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    inicio = time.time()
    if(formato == 0):
        input_image(m,path,namesfile,use_cuda)
    if(formato == 1):
        input_dir_image(m,path,namesfile,use_cuda)
    if(formato == 2):
        fps = input_video(m,path,namesfile,carpeta_save,use_cuda)
        convert_img_video(carpeta_save,fps)
    print('T',time.time()-inicio)

if __name__ == '__main__':
    cfgfile = "cfg/yolo-camion-darknet.cfg"
    weightfile = "backup/yolo-camion_14000.weights"
    carpeta_videos = "./video-predict"

    path = "./video-cam"

    estado = []

    def flujo(ruta):
        while(1):
            inicio = time.time()
            t_end = inicio + 10 * 1
            estado.append(Path(os.path.abspath(path+"/"+ruta)).stat().st_size)
            while(time.time() < t_end):
                print(Path(os.path.abspath(path+"/"+ruta)).stat().st_size)
            estado.append(Path(os.path.abspath(path+"/"+ruta)).stat().st_size)
            print(str(estado[len(estado)-2])+"  "+str(estado[len(estado)-1]))
            if(estado[len(estado)-2] == estado[len(estado)-1] ):
                    print("no hubo modificacion")
                    break
        return True

    #print(os.listdir(path)[0])
    #flujo(os.listdir(path)[0])

    while(1):
        if(len(os.listdir(path))>0):
            fl = flujo(os.listdir(path)[0])
            if fl:
                print("llamar detector")
                imgfile = os.path.abspath(path+"/"+os.listdir(path)[0])
                print(imgfile.split("/")[7].split(".")[0])
                carpeta_save = carpeta_videos+"/"+str(imgfile).split(".")[0].split("/")[7]
                if(os.path.isdir(carpeta_videos) == False):
                    os.mkdir(carpeta_videos)
                if(os.path.isdir(carpeta_save) == False):
                    os.mkdir(carpeta_save)
                detect(cfgfile, weightfile, imgfile, 2,carpeta_save)
                shutil.move(imgfile,os.path.abspath("videos-ok"+"/"+os.listdir(path)[0]))
                #break




    """    
    if len(sys.argv) == 2:
        star = time.time()
        #cfgfile = sys.argv[1]
        #weightfile = sys.argv[2]
        #imgfile = sys.argv[3]
        print(sys.argv[1])
        cfgfile = "cfg/yolo-camion-darknet.cfg"
        weightfile = "backup/yolo-camion_14000.weights"

        #imgfile = "./video-desarrollo/ch06_20200106101038.mp4" # Directorio de videos
        imgfile = sys.argv[1]
        #imgfile = "./946.jpg" # una imagen
        #imgfile = "./img-desarrollo" # directorio de imagenes

        carpeta_videos = "./video-predict"
        
        
        if(os.path.isfile(imgfile)): # Una imagen
            "Una Imagen"
            if(imgfile.split(".")[2] == "jpg"):
                detect(cfgfile, weightfile, imgfile, 0)
            "Un Video"
            if(imgfile.split(".")[2] == "mp4"):
                carpeta_save = carpeta_videos+"/"+str(imgfile).split(".")[1].split("/")[2]
                if(os.path.isdir(carpeta_videos) == False):
                    os.mkdir(carpeta_videos)
                if(os.path.isdir(carpeta_save) == False):
                    os.mkdir(carpeta_save)
                detect(cfgfile, weightfile, imgfile, 2,carpeta_save)
        "Directorios"
        if(os.path.isdir(imgfile)):
            "Directorio de Imagenes"
            if(os.listdir(imgfile)[0].split(".")[1] == "jpg"):
                detect(cfgfile, weightfile, imgfile, 1)

            "Directorio de Videos"
            if(os.listdir(imgfile)[0].split(".")[1] == "mp4"):
                videos = os.listdir(imgfile)
                for i in range(len(videos)):
                    carpeta_save = carpeta_videos+"/"+str(videos[i]).split(".")[0]
                    path = imgfile+"/"+str(videos[i])
                    if(os.path.isdir(carpeta_videos) == False):
                        os.mkdir(carpeta_videos)
                    if(os.path.isdir(carpeta_save) == False):
                        os.mkdir(carpeta_videos+"/"+str(videos[i]).split(".")[0])
                    detect(cfgfile, weightfile, path, 2, carpeta_save)
                    convert_img_video(carpeta_save)
        #
        
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        # detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
    """