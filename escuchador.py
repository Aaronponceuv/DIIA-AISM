import os 
#from moviepy.editor import VideoFileClip
import time
from pathlib import Path
path = "./filezilla"

estado = []

def flujo(ruta):
    while(1):
        inicio = time.time()
        t_end = inicio + 10 * 1
        estado.append(Path(os.path.abspath("video-cam/"+ruta)).stat().st_size)
        while(time.time() < t_end):
            print(Path(os.path.abspath("video-cam/"+ruta)).stat().st_size)
        estado.append(Path(os.path.abspath("video-cam/"+ruta)).stat().st_size)
        print(str(estado[len(estado)-2])+"  "+str(estado[len(estado)-1]))
        if(estado[len(estado)-2] == estado[len(estado)-1] ):
                print("no hubo modificacion")
                break
    return True

#print(os.listdir(path)[0])
#flujo(os.listdir(path)[0])

while(1):
    if(len(os.listdir(path))>0):
        estado = flujo(os.listdir(path)[0])
        if estado:
            print("llamar detector")
            break