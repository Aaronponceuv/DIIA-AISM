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

def input_video(model,path,namesfile,carpeta_save,use_cuda,mode=None):
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print('fps',fps)


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
    if(mode == "dvideo"):
        print("Modo dvideo")   
        print(path.split(".")[0].split("/")[1])
        name_xlsx = path.split(".")[0].split("/")[1]
        data.to_excel("./registro_frame/"+str(name_xlsx)+'.xlsx', sheet_name='detectados' , index=False)
    if(mode == "carpeta/video"):
        print(path.split(".")[0].split("/")[1])
        name_xlsx = path.split(".")[0].split("/")[1]
        data.to_excel("./registro_frame/"+str(name_xlsx)+'.xlsx', sheet_name='detectados' , index=False)
    else:
        print(path.split(".")[0])
        name_xlsx = path.split(".")[0]
       # data.to_excel("./registro_frame/"+str(name_xlsx)+'.xlsx', sheet_name='detectados' , index=False)

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
    print('Tiempo de Procesamiento:',sum(tiempo_de_prediccion),"seg")

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


def detect(cfgfile, weightfile, path, formato,carpeta_save=None,mode=None):
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
        fps = input_video(m,path,namesfile,carpeta_save,use_cuda,mode)
        convert_img_video(carpeta_save,fps)
    print('T',time.time()-inicio)

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

if __name__ == '__main__':
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        option = sys.argv[3]
        imgfile = sys.argv[4]
        carpeta_videos = "./video-predict"


        if(option == "img"):
            print("option imagen")
            detect(cfgfile, weightfile, imgfile, 0)
        if(option == "dimg"):
            print("directorio de imagen")
            detect(cfgfile, weightfile, imgfile, 1)
        if(option == "dvideo"):
            print("directorio de videos")
            videos = os.listdir(imgfile)
            print(videos)
            for vd in videos:
                print(vd.split(".")[0])
                carpeta_vd = vd.split(".")[0]
                carpeta_save = carpeta_videos+"/"+vd.split(".")[0]
                path = imgfile+"/"+str(vd)
                print(path)
                if(os.path.isdir(carpeta_videos+"/"+str(carpeta_vd)) == False):
                    os.mkdir(carpeta_videos+"/"+str(carpeta_vd))
                detect(cfgfile, weightfile, path, 2, carpeta_save,"dvideo")
        if(option == "video"):
            print("option video")
            if(len(str(imgfile).split("/")) > 1):
                print("video en un dir")
                print(str(imgfile).split("/")[1].split(".")[0])
                carpeta_save = carpeta_videos+"/"+str(imgfile).split("/")[1].split(".")[0]
                print(carpeta_save)
                detect(cfgfile, weightfile, imgfile, 2,carpeta_save,"carpeta/video")
            else:
                carpeta_save = carpeta_videos+"/"+str(imgfile).split(".")[0]
                print(carpeta_save)
                if(os.path.isdir(carpeta_save) == False):
                    os.mkdir(carpeta_save)
                detect(cfgfile, weightfile, imgfile, 2,carpeta_save)