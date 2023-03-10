import cv2 
import numpy as np
import math
from matplotlib import pyplot as plt


# importando imagem
img = cv2.imread("circulo.png") 
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


#aplicando a mascara nas cores
mask_hsv1 = cv2.inRange(img_hsv, (80, 160, 220 ), (90, 170, 230)) # 173°, 65%, 89%
mask_hsv2 = cv2.inRange(img_hsv, (0, 200, 165 ), (0, 240, 180)) # 0°, 94%, 69%

mask_soma = cv2.bitwise_or(mask_hsv1, mask_hsv2)

#achando os contornos
contornos, _ = cv2.findContours(mask_soma, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

img_rgb = cv2.bitwise_and(img, img, mask=mask_soma)
img_final = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
cv2.drawContours(img_final, contornos, -1, [0, 0, 255], 5);

#Descobrindo o centro de massa dos circulos
cnt1 = contornos[0]
M1 = cv2.moments(cnt1)

cx1 = int(M1['m10']/M1['m00'])
cy1 = int(M1['m01']/M1['m00'])

cnt2 = contornos[1]
M2 = cv2.moments(cnt2)

cx2 = int(M2['m10']/M2['m00'])
cy2 = int(M2['m01']/M2['m00'])


#desenhando cruz no centro de massa dos circulos 
size = 20
color = (128,128,0)

cv2.line(img_final,(cx1 - size,cy1),(cx1 + size,cy1),color,3)
cv2.line(img_final,(cx1,cy1 - size),(cx1, cy1 + size),color,3)

cv2.line(img_final,(cx2 - size,cy2),(cx2 + size,cy2),color,3)
cv2.line(img_final,(cx2,cy2 - size),(cx2, cy2 + size),color,3)


#calculando area dos circulos
area1 = cv2.contourArea(cnt1)
area2 = cv2.contourArea(cnt2)


#Escrevendo a massa e centro de massa na imagem
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img_final, str(area1), (cx1 - 50,cy1 - 40), font,1,(0,255,0),2,cv2.LINE_AA)
cv2.putText(img_final, str(f'({cx1}, {cy1})'), (cx1 - 90,cy1 + 50), font,1,(0,255,0),2,cv2.LINE_AA)


cv2.putText(img_final, str(area2), (cx2 - 65,cy2 - 30), font,1,(255,0,0),2,cv2.LINE_AA)
cv2.putText(img_final, str(f'({cx2}, {cy2})'), (cx2 - 80,cy2 + 65), font,1,(255,0,0),2,cv2.LINE_AA)


#Traçando reta entre os dois centros de massa
cv2.line(img_final,(cx1,cy1),(cx2, cy2 ),(0,255,0),3)

#Calculando o angulo da reta
coeficiente_angular = (cy1 - cy2) / (cx1 -  cx2)

angulo = math.degrees(math.atan(coeficiente_angular))
angulo_ajustado = round(angulo, 2)

#Escrevendo o angulo na imagem
cv2.putText(img_final, f"({angulo_ajustado} graus)", (int(img.shape[1] / 2), int(img.shape[0] / 2 )), font,1,(255,0,0),2,cv2.LINE_AA)


#Exibindo a imagem
plt.imshow(img_final, cmap="Greys_r")
plt.show()