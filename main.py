from tkinter import *
import numpy as np
import cv2

raiz=Tk()
raiz.title("Reconocimiento de amenaza")

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
        # print('-')
        # print( bbox2[0] )
        # En aqui, por ejemplo, selecciono el objeto 1 (En el array seria 0), y en ahi estarian las coordenadas del objeto
        # en mi panel de vista xd
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

button=Button(raiz,text="Acceder a la funcion", command=ReconocimientoAccion)
button.pack()

raiz.geometry("640x480")

raiz.mainloop()