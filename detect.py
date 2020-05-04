import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import os

def detect(cfgfile, weightfile, path):
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
    
    
    isDirectory = os.path.isdir(path)

    if(isDirectory == True):
        imagenes = os.listdir(path)
        folder_predictions = './'+str(path)+'-predictions'
        if(os.path.isdir(folder_predictions) == False):
            os.mkdir(folder_predictions)
        print(imagenes)
        for z in range(len(imagenes)):
            imgfile = path+'/'+str(imagenes[z])
            img = Image.open(imgfile).convert('RGB')
            sized = img.resize((m.width, m.height))
            for i in range(2):
                start = time.time()
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                finish = time.time()
                if i == 1:
                    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

            class_names = load_class_names(namesfile)
            plot_boxes(img, boxes, folder_predictions+'/'+str(imagenes[z]).split(".")[0]+'-predictions.jpg', class_names)
    else:
        img = Image.open(path).convert('RGB')
        sized = img.resize((m.width, m.height))
        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (path, (finish - start)))

        class_names = load_class_names(namesfile)
        plot_boxes(img, boxes, 'predictions.jpg', class_names)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        #cfgfile = sys.argv[1]
        cfgfile = "cfg/yolo-camion-darknet.cfg"
        #weightfile = sys.argv[2]
        weightfile = "backup/yolo-camion_14000.weights"
        #imgfile = sys.argv[3]
       # imgfile = "img/2-2345.jpg"
        imgfile = "img"
        print(imgfile)
        detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        # detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
