import os
import numpy as np
import pandas as pd
"""
for class_archivo in archivos:
    f = open(os.path.abspath(os.path.join(path,class_archivo)),'r')
    lineas = f.read().splitlines()
    #print(lineas,"\n")
    f.close()
"""
def leer_predicciones(archivos):
    lista_archivos = []
    for file_archivos in archivos:
        data = pd.read_csv(os.path.abspath(os.path.join(path,file_archivos)), sep=" ", header=None)
        data.columns = ["imagen", "prob", "xmin_pred", "ymin_pred", "xmax_pred", "ymax_pred"]
        lista_archivos.append(data)
    return lista_archivos

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def leer_target_cord(lineas_archivo_valid):
    target = []
    for linea in lineas_archivo_valid:
        linea = linea.replace("imagenes","labels").replace(".jpg",".txt")
        nombre = os.path.basename(linea).split('.')[0]
        data_nombre = pd.DataFrame(columns=['imagen'])
        data_nombre.loc[0] = [nombre]
        data = pd.read_csv(linea, sep=" ", header=None)
        data = pd.concat([data_nombre,data],axis=1, ignore_index=True)
        data.columns = ["imagen","class", "xmin", "ymin", "xmax", "ymax"]
        target.append(data)
    return pd.concat(target)


def IOU(df):
    copy = df.copy()
    df_iou = pd.DataFrame(columns=['imagen','iou'])
    idx = 0
    for index, row in df.iterrows():
        xmin_inter = max(row["xmin"], row["xmin_pred"])
        ymin_inter = max(row["ymin"], row["ymin_pred"])
        xmax_inter = min(row["xmax"], row["xmax_pred"])
        ymax_inter = min(row["ymax"], row["ymax_pred"])

        # Calculo de area de intersecion de rectangulos
        inter_area = max(0, xmax_inter - xmin_inter + 1) * max(0, ymax_inter - ymin_inter + 1)
 
        # Calculo de area objetivo y area de prediccion
        actual_area = (row["xmax"] - row["xmin"] + 1) * (row["ymax"]- row["ymin"] + 1)
        pred_area = (row["xmax_pred"] - row["xmin_pred"] + 1) * (row["ymax_pred"] - row["ymin_pred"] + 1)

        # Calculo interseccion sobre union
        iou = inter_area / float(actual_area + pred_area - inter_area)
        df_iou.loc[idx] = [row["imagen"], iou]
        idx+=1
    merge = pd.merge(df, df_iou, on='imagen')
    return merge
    

def precision_recall(iou):
    Precision = []
    Recall = []
    TP = FP = 0
    FN = len(iou[iou['TP/FP']== 'TP'])
    for index , row in iou.iterrows():     
        if row['iou'] > 0.5:
            TP =TP+1
        else:
            FP =FP+1    
        try:
            AP = TP/(TP+FP)
            Rec = TP/(TP+FN)
        except ZeroDivisionError:
            AP = Recall = 0.0

        Precision.append(AP)
        Recall.append(Rec)
    iou['Precision'] = Precision
    iou['Recall'] = Recall
    return iou

def Map(iou):
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            x = iou[iou['Recall'] >= recall_level]['Precision']
            prec = max(x)
        except:
            prec = 0.0
        print("AP para Recall:",recall_level," ",prec)
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
   #print('11 point precision is ', prec_at_rec)
    print('mAP is ', avg_prec)

if __name__ == "__main__":
    path = "./results"
    datacfg = "./cfg/camion.data"

    archivos_pred = os.listdir(path)
    lista_archivos = leer_predicciones(archivos_pred)
    predicciones = pd.concat(lista_archivos)

    options = read_data_cfg(datacfg)
    path_archivo_valid = options['valid']
    
    f = open(path_archivo_valid, "r")
    lineas_archivo_valid = f.read().splitlines()

    cord_target = leer_target_cord(lineas_archivo_valid)
    union = pd.merge(predicciones, cord_target, on='imagen')

    iou=IOU(union)
    iou = iou.drop(["xmin", "ymin", "xmax", "ymax","xmin_pred", "ymin_pred", "xmax_pred", "ymax_pred"], axis=1)
    eval_table = pd.DataFrame()
    iou['TP/FP'] = iou['iou'].apply(lambda x: 'TP' if x>=0.5 else 'FP')
    iou = precision_recall(iou)
    iou['IP'] = iou.groupby('Recall')['Precision'].transform('max')
    Map(iou)
    iou.to_excel("testin.xlsx")
