import cv2
import os,sys, os.path
import numpy as np
import math

#importes para emular precionamento de teclas
from pynput.keyboard import Key, Controller
import pynput
import time
import random

keys = [
    pynput.keyboard.KeyCode.from_char('w'),  # A
    pynput.keyboard.KeyCode.from_char('a'),  # B
    pynput.keyboard.KeyCode.from_char('s'),  # X
    pynput.keyboard.KeyCode.from_char('d'),  # X
]



#Inicializa o controle 
keyboard = Controller()


image_lower_hsv2 = np.array([10,80,140])
image_upper_hsv2 = np.array([50,200,250])



def filtro_de_cor(img_bgr, low_hsv, high_hsv):
    """ retorna a imagem filtrada"""
    img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, low_hsv, high_hsv)
    return mask 

def mascara_or(mask1, mask2):

    """ retorna a mascara or"""
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def mascara_and(mask1, mask2):
     """ retorna a mascara and"""
     mask = cv2.bitwise_and(mask1, mask2)
     
     return mask

def desenha_cruz(img, cX,cY, size, color):
     """ faz a cruz no ponto cx cy"""
     cv2.line(img,(cX - size,cY),(cX + size,cY),color,5)
     cv2.line(img,(cX,cY - size),(cX, cY + size),color,5)    

def escreve_texto(img, text, origem, color):
     """ faz a cruz no ponto cx cy"""
 
     font = cv2.FONT_HERSHEY_SIMPLEX
     
     cv2.putText(img, str(text), origem, font,1,color,2,cv2.LINE_AA)



def image_da_webcam(img):
    """
    ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-
        deve receber a imagem da camera e retornar uma imagems filtrada.
    """  
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    mask_hsv = filtro_de_cor(img, image_lower_hsv2, image_upper_hsv2)
    
    contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)

    #Descobrindo o centro de massa dos circulos
    if len(contornos) != 0 and len(contornos) != 1 :
        contornoMaior = contornos_ordenados[0]
        contornoMaior2 = contornos_ordenados[1]
        
        cv2.drawContours(img_rgb, contornos_ordenados[0], -1, [0, 0, 255], 5);
        cv2.drawContours(img_rgb, contornos_ordenados[1], -1, [0, 0, 255], 5);

        cnt1 = contornoMaior
        cnt2 = contornoMaior2
        M1 = cv2.moments(cnt1)
        M2 = cv2.moments(cnt2)
        if M1["m00"] != 0 and M2["m00"] != 0:
            
            cx1 = int(M1['m10']/M1['m00'])
            cy1 = int(M1['m01']/M1['m00'])


            cx2 = int(M2['m10']/M2['m00'])
            cy2 = int(M2['m01']/M2['m00'])


            #desenhando cruz no centro de massa dos circulos 
            size = 20
            color = (0,255,0)

            cv2.line(img_rgb,(cx1 - size,cy1),(cx1 + size,cy1),color,3)
            cv2.line(img_rgb,(cx1,cy1 - size),(cx1, cy1 + size),color,3)

            cv2.line(img_rgb,(cx2 - size,cy2),(cx2 + size,cy2),color,3)
            cv2.line(img_rgb,(cx2,cy2 - size),(cx2, cy2 + size),color,3)


            #calculando area 
            area1 = cv2.contourArea(cnt1)
            area2 = cv2.contourArea(cnt2)

            #Traçando reta entre os dois centros de massa
            cv2.line(img_rgb,(cx1,cy1),(cx2, cy2 ),(0,255,0),3)

            #Calculando o angulo da reta
            if cx1 - cx2 != 0:
                coeficiente_angular = (cy1 - cy2) / (cx1 -  cx2)

                angulo = math.degrees(math.atan(coeficiente_angular))
                angulo_ajustado = round(angulo, 2)

                #Escrevendo o angulo na imagem
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_rgb, f"({angulo_ajustado} graus)", (10, 60), font,0.5,(255,0,0),1,cv2.LINE_AA)
                cv2.putText(img_rgb, f"(Area:{area1})", (10, 20), font,0.5,(255,0,0),1,cv2.LINE_AA)
                cv2.putText(img_rgb, f"(Centro de massa 1 - {cx1}, {cy2})", (10, 100), font,0.5,(255,0,0),1,cv2.LINE_AA)
                cv2.putText(img_rgb, f"(Centro de massa 2 - {cx2}, {cy2})", (10, 140), font,0.5,(255,0,0),1,cv2.LINE_AA)


                #presionando teclas
                if area1 >= 10000:
                    keyboard.press(keys[0])
                    time.sleep(0.1)
                    keyboard.release(keys[0])
                elif area1 < 6000:
                    keyboard.press(keys[2])
                    time.sleep(1)
                    keyboard.release(keys[2])

                if angulo_ajustado > 15 :
                    keyboard.press(keys[1])
                    time.sleep(0.1)
                    keyboard.release(keys[1])
                elif angulo_ajustado < -15:
                    keyboard.press(keys[3])
                    time.sleep(0.1)
                    keyboard.release(keys[3])


    return img_rgb

cv2.namedWindow("preview")
# define a entrada de video para webcam
vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)


#configura o tamanho da janela 
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    
    img = image_da_webcam(frame) # passa o frame para a função imagem_da_webcam e recebe em img imagem tratada

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow("preview", img_rgb)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()