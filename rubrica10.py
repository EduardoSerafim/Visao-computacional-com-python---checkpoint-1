import math
import cv2
import os,sys, os.path
import numpy as np



def image_da_webcam(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    

    #aplicando a mascara nas cores
    mask_hsv1 = cv2.inRange(img_hsv, (150, 140, 100 ), (200, 255, 255)) # 337°, 78%, 61%
    mask_hsv2 = cv2.inRange(img_hsv, (20, 50, 70 ), (50, 180, 250)) # 58°, 44%, 55%

    mask_soma = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    
    #achando os contornos
    contornos, _ = cv2.findContours(mask_soma, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
    

    #Descobrindo o centro de massa dos circulos
    if len(contornos) != 0 and len(contornos) != 1 :
        cv2.drawContours(img_rgb, contornos_ordenados[0], -1, [0, 0, 255], 5);
        cv2.drawContours(img_rgb, contornos_ordenados[1], -1, [0, 0, 255], 5);
        contornoMaior = contornos_ordenados[0]
        contornoMaior2 = contornos_ordenados[1]
        cnt1 = contornoMaior
        M1 = cv2.moments(cnt1)
        

        cx1 = int(M1['m10']/M1['m00'])
        cy1 = int(M1['m01']/M1['m00'])

        cnt2 = contornoMaior2
        M2 = cv2.moments(cnt2)

        cx2 = int(M2['m10']/M2['m00'])
        cy2 = int(M2['m01']/M2['m00'])


        #desenhando cruz no centro de massa dos circulos 
        size = 20
        color = (0,255,0)

        cv2.line(img_rgb,(cx1 - size,cy1),(cx1 + size,cy1),color,3)
        cv2.line(img_rgb,(cx1,cy1 - size),(cx1, cy1 + size),color,3)

        cv2.line(img_rgb,(cx2 - size,cy2),(cx2 + size,cy2),color,3)
        cv2.line(img_rgb,(cx2,cy2 - size),(cx2, cy2 + size),color,3)


        #calculando area dos circulos
        area1 = cv2.contourArea(cnt1)
        area2 = cv2.contourArea(cnt2)

        #Traçando reta entre os dois centros de massa
        cv2.line(img_rgb,(cx1,cy1),(cx2, cy2 ),(0,255,0),3)

        #Calculando o angulo da reta
        coeficiente_angular = (cy1 - cy2) / (cx1 -  cx2)

        angulo = math.degrees(math.atan(coeficiente_angular))
        angulo_ajustado = round(angulo, 2)

        #Escrevendo o angulo na imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_rgb, f"({angulo_ajustado} graus)", (200, 100), font,1,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(img_rgb, str(f'({cx1}, {cy1})'), (cx1 - 90,cy1 + 50), font,1,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(img_rgb, str(f'({cx2}, {cy2})'), (cx2 - 80,cy2 + 65), font,1,(255,0,0),2,cv2.LINE_AA)
    
    return img_rgb

cv2.namedWindow("preview")
vc = cv2.VideoCapture("video.mp4")


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    
    img = image_da_webcam(frame)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("preview", img_rgb)

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()