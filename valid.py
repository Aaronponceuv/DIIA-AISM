from darknet import Darknet
import dataset
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
import os
import configparser

def valid(datacfg, cfgfile, weightfile, outfile):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    name_list = options['names']
    prefix = 'results'
    names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()

    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    fps = [0]*m.num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(m.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    
    count = 0
    for batch_idx, (data, target) in enumerate(valid_loader):
        
        data = data.cuda()
        data = Variable(data, volatile = True)
        output = m(data).data
        batch_boxes = get_region_boxes(output, conf_thresh, m.num_classes, m.anchors, m.num_anchors, 0, 1)
        for i in range(output.size(0)):
            lineId = lineId + 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            width, height = get_image_size(valid_files[lineId])
            print(valid_files[lineId])
            boxes = batch_boxes[i]
            boxes = nms(boxes, nms_thresh)
            aux = []
            for box in boxes:
                """
                Descomentar en caso de re-escalar cordenadas
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height
                """
                x1 = (box[0]) 
                y1 = (box[1]) 
                x2 = (box[3]) 
                y2 = (box[3]) 
                det_conf = box[4]
                for j in range((len(box)-5)//2):
                    count +=1
                    cls_conf = box[5+2*j]
                    cls_id = box[6+2*j]
                    prob =det_conf * cls_conf
                    aux.append([prob.item(), x1.item(), y1.item(), x2.item(), y2.item()])
                    """
                    Para almacenar todos los boxes es necesario descomentar la linea de abajo y comentar linea 93
                    """
                    #fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))
                if(len(boxes) == len(aux)):
                    """
                    De todos los boxes arrojados se almacena solo el box con mayor probabilidad
                    almancenando la prediccion para una posterior evaluacion de mAP
                    """
                    i,j = np.where(np.array(aux) == max(np.array(aux)[:,0]))
                    fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, np.float64(np.array(aux)[i,j]), np.float64(np.array(aux)[i,1]), np.float64(np.array(aux)[i,2]), np.float64(np.array(aux)[i,3]), np.float64(np.array(aux)[i,4])))
    for i in range(m.num_classes):
        fps[i].close()
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        config = configparser.ConfigParser()
        config.read('./config.ini', encoding="utf-8")
        datacfg  = config['Train_Parameters']['cfgdata']
        cfgfile = config['Train_Parameters']['cfgfile']
        weightfile = config['Detect_Parameters']['weightfile']

        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')