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

def ordenerar(path):
    imgs = os.listdir(path)
    name_video = []
    ruta = []
    for i in range(len(imgs)):
        print(imgs[i].split(".")[0])
        name_video.append(int(imgs[i].split(".")[0]))
    name_video = sorted(name_video)
   # print(name_video)
    return name_video
 
def convert_img_video():
    img_array = []
    path = "./video-predict/camion-minuto"
    imgs = ordenerar(path)
    print(imgs)
    for filename in range(len(imgs)):
        print('---'+path+"/"+str(imgs[filename]))
        img = cv2.imread(path+"/"+str(imgs[filename])+".jpg")
        height, width, layer = img.shape
        size = (width,height)
        img_array.append(img)
    print(size)
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def input_video(model,path,namesfile,carpeta_save,use_cuda):
    cap = cv2.VideoCapture(path)

    frameCount = 0
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
        plot_boxes(frame, boxes,carpeta_save+"/"+str(frameCount)+'.jpg', class_names)
        frameCount +=1

def input_dir_image(model, path, namesfile,use_cuda):
    imagenes = os.listdir(path)
    folder_predictions = './'+str(path)+'-predictions'
    if(os.path.isdir(folder_predictions) == False):
        os.mkdir(folder_predictions)
    for z in range(len(imagenes)):
        imgfile = path+'/'+str(imagenes[z])
        print(imgfile)
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
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/camion.names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    if(formato == 0):
        print("0")
        input_image(m,path,namesfile,use_cuda)
    if(formato == 1):
        input_dir_image(m,path,namesfile,use_cuda)
    if(formato == 2):
        input_video(m,path,namesfile,carpeta_save,use_cuda)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        #cfgfile = sys.argv[1]
        #weightfile = sys.argv[2]
        #imgfile = sys.argv[3]

        cfgfile = "cfg/yolo-camion-darknet.cfg"
        weightfile = "backup/yolo-camion_14000.weights"

        imgfile = "./video-desarrollo" # Directorio de videos
        #imgfile = "./predictions.jpg" # una imagen
        #imgfile = "./img-desarrollo" # directorio de imagenes

        carpeta_videos = "./video-predict"
        
        if(os.path.isfile(imgfile)): # Una imagen
            print(imgfile.split(".")[2])
            if(imgfile.split(".")[2] == "jpg"):
                detect(cfgfile, weightfile, imgfile, 0)
            if(imgfile.split(".")[1] == "mp4"):
                detect(cfgfile, weightfile, imgfile, 2)
        if(os.path.isdir(imgfile)): # Directorio de imagenes
            if(os.listdir(imgfile)[0].split(".")[1] == "jpg"):
                detect(cfgfile, weightfile, imgfile, 1)
            if(os.listdir(imgfile)[0].split(".")[1] == "mp4"):
                print(os.listdir(imgfile))
                videos = os.listdir(imgfile)
               # if(os.path.isdir(carpeta_videos) == True):
               #     rmtree(carpeta_videos)
                for i in range(len(videos)):
                    carpeta_save = carpeta_videos+"/"+str(videos[i]).split(".")[0]
                    path = imgfile+"/"+str(videos[i])
                    if(os.path.isdir(carpeta_videos) == False):
                        os.mkdir(carpeta_videos)
                    if(os.path.isdir(carpeta_save) == False):
                        os.mkdir(carpeta_videos+"/"+str(videos[i]).split(".")[0])
                    print(path)
                    detect(cfgfile, weightfile, path, 2, carpeta_save)
        
        #convert_img_video()
        #ordenerar()
        
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        # detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
