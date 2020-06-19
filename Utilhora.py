import datetime

def formato_hora(hora):
    hrs = hora[0:2]
    if(hrs[0] == '0'):
        hrs = hrs[1]
    min = hora[2:4]
    seg = hora[4:6]
    return str(hrs+':'+min+':'+seg)

def segundos_a_hora(segundos):
    horas = int(segundos / 3600)
    segundos -= horas * 3600
    minutos = int(segundos / 60)
    segundos -= int(minutos * 60)
 #   print("%s:%s:%s" % (horas, minutos, segundos))
    #print(int(segundos))
    #if(str(segundos[1]) == "."):
    #    segundos = (0+segundos[0])
    return "%s:%s:%s" % (horas, minutos, int(segundos))

def sumar_hora(hora1,hora2):
    formato = "%H:%M:%S"
    lista = hora2.split(":")
    hora=int(lista[0])
    minuto=int(lista[1])
    segundo=int(lista[2])
    h1 = datetime.datetime.strptime(hora1, formato)
    dh = datetime.timedelta(hours=hora)
    dm = datetime.timedelta(minutes=minuto)
    ds = datetime.timedelta(seconds=segundo)
    resultado1 =h1 + ds
    resultado2 = resultado1 + dm
    resultado = resultado2 + dh
    resultado=resultado.strftime(formato)
    if(resultado[0]=='0'):
        resultado = resultado[1:len(resultado)]
    return str(resultado)

def hora_a_segundos(hora):
    hora = str(hora)
    hrs = int(hora.split(":")[0]) * 3600
    min = int(hora.split(":")[1] )*60
    seg =int(hora.split(":")[2] )
    total = hrs+min+seg
    return total

def milisegunos_a_segundos(milisisegundos):
    return milisisegundos/1000

def segundos_a_milisegudos(segundos):
    return segundos*1000

def hora_a_milisiegundos(hora):
    segundos = hora_a_segundos(hora)
    milisegundos = segundos_a_milisegudos(segundos)
    return milisegundos

def milisegundos_a_hora(milis):
    seg = milisegunos_a_segundos(milis)
    hora = segundos_a_hora(seg)
    return hora