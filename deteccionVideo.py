
#Librerias

from tkinter import *
import numpy as np
import cv2
from tkinter import ttk

import requests
import json

#credenciales

serverToken = 'Tu Server Token'
deviceToken = 'Tu Device Token'

window = Tk()

window.title("BE SAFE! - SOFTWARE - SECURITY CAM")

tab_control = ttk.Notebook(window)

#Paginas - Frames
tab1 = Frame(tab_control)
tab2 = Frame(tab_control)
tab3 = Frame(tab_control)

#Paginas Titulos
tab_control.add(tab1, text='Login')
tab_control.add(tab2, text='Mi perfil')
tab_control.add(tab3, text='Seguridad Cam')

##
#Pagina 1
LT1 = Label(tab1, text="BESAFE", justify='center')
LT1.grid(column=3, row=1)
LT2 = Label(tab1, text="Reconocimiento a través de cámaras web\npara prevenir violencia intrafamiliar\n")
LT2.grid(column=3, row=2)
#

lbl = Label(tab1, text="Iniciar sesión")
lbl.grid(column=0, row=4)

lbl1 = Label(tab1, text="")
lbl1.grid(column=1, row=4)

lblUser = Label(tab1, text="Usuario:")
lblUser.grid(column=0, row=5)

lblPass = Label(tab1, text="Contraseña:")
lblPass.grid(column=0, row=6)

txt = Entry(tab1,width=10)
txt.grid(column=1, row=5)

txt1 = Entry(tab1,width=10)
txt1.grid(column=1, row=6)
##

def clicked():

    res = "Bienvenida " 
    res1 = txt.get()

    lbl.configure(text= res)
    lbl1.configure(text= res1)

    
def newtab():
    tab_control.tab(0, state="disabled")
    tab_control.select(tab2) 

btn = Button(tab1, text="Click", command=newtab)

btn.grid(column=1, row=7)


##
#Pagina2
#Pantalla perfil

def logout() :
    tab_control.tab(0, state="normal")
    tab_control.tab(1, state="disabled")
    tab_control.select(tab1)

def changePage() :
    tab_control.select(tab3)


btn1 = Button(tab2, text="Seguridad rastrear", command=changePage)
btn1.grid(column=1, row=7)

btn2 = Button(tab2, text="Logout", command=logout)
btn2.grid(column=1, row=8)

##

##
#Pagina3

###---Notificacion Push---####
def sendPushNotification():
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=' + serverToken,
    }

    body = {
        'notification': {
            'body' : "Algo esta ocurriendole a tu familia!!",
            'title' : "ALERTA!!!"
        },
        'to': deviceToken,
        'priority': 'high',
        'data':{
            'click_action': "FLUTTER_NOTIFICATION_CLICK",
            'solicitud' : "1010",            
            'mensaje'   : "EMERGENCIA! ESTA OCURRIENDO VIOLENCIA!!"
        },
    }
    response = requests.post("https://fcm.googleapis.com/fcm/send",headers = headers, data=json.dumps(body))
    print(response.status_code)

    print(response.json())


################################

###--Reconocimiento Accion---####
def ReconocimientoAccion():
    UmbralDeAceptacion = 0.45
    nmsUmbralAceptacion = 0.2

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture('PruebaGuille3_Trim.mp4')
    cap.set(3, 640)
    cap.set(4, 480)

    classNames = []
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    #print(classNames)

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)

    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        ret, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=UmbralDeAceptacion)
        # print(classIds)
        # print(classIds, bbox)
        # print(bbox)
        bbox2 = list(bbox)  # Convierto el bbox en una lista de objetos capturados en ese momento
        # Veamos, esto es simple, bbox me da los array de todos los objetos detectados
        # print( bbox2[0] )
        # En aqui, por ejemplo, selecciono el objeto 1 (En el array seria 0), y en ahi estarian las coordenadas del objeto
        # en mi panel de vista 
        confs2 = list(np.array(confs).reshape(1, -1)[0])  # En aqui me da los niveles de confianza en una array
        confs2 = list(map(float, confs2))
        # print(type(confs2[0]))                                  # Hmmmmm

        indices = cv2.dnn.NMSBoxes(bbox2, confs2, UmbralDeAceptacion, nmsUmbralAceptacion)
        # print (indices)
        # CajaPersona=[1,2,3,4]
        # CajaCuchillo=[1,2,3,4]
        Señal = [False, False]
        for i in indices:
            i = i[0]
            box = bbox2[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            # print(i)
            if classNames[classIds[i][0] - 1].upper() == 'PERSONA':
                CajaPersona = box
                Señal[0] = True
            if classNames[classIds[i][0] - 1].upper() == 'CUCHILLO':
                CajaCuchillo = box
                Señal[1] = True
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            # classIds[i][0] es el id del indice en coco.name, luego se coloca el -1 porque el array inicia en 0
            cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        if Señal[0] == True and Señal[1] == True:
            if (CajaPersona[1] <= CajaCuchillo[1]) and (
                    (CajaPersona[1] + CajaPersona[3]) >= (CajaCuchillo[1] + CajaCuchillo[3])):
                cv2.putText(img, 'ADVERTENCIA:POSIBLE AMENAZA!!', (20, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                #Enviar notificación de Auxilio!
                sendPushNotification()                           
               
        # if len( classIds ) !=0 :
        #    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        #        cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
        #        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
        #                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


btn3 = Button(tab3, text="Ver Cámara", command=ReconocimientoAccion)
btn3.grid(column=5, row=2)

##

tab_control.pack(expand=1, fill='both')

window.geometry("640x480")
window.mainloop()