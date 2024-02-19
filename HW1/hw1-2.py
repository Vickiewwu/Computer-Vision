import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy import linalg
import math

#讀圖片且轉成灰階 要跟code放在同一個資料夾內
img = cv2.imread('hw1-2.jpg',cv2.IMREAD_GRAYSCALE)
rows,cols = img.shape[0:2]

#取得kernel內部參數
kernel_size = 3
sigma = 3

#利用一維參數取得二維參數 T表示轉置
kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
kernel_2d = kernel_1d * kernel_1d.T

#進行gaussian filter(blur)
#-1表示產出的圖跟原圖會有相同depth
img_gb = cv2.filter2D(img, -1, kernel_2d)
cv2.imwrite('hw1-2i.jpg', img_gb)
cv2.imshow("image", img_gb)
cv2.waitKey(0)

#apply Sobel operator Hx
Hx = np.array(
[[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]]
)
gx = convolve2d(img_gb, Hx, mode='same')

#把原本有正有負的結果 轉成0-255印出
gmx = np.zeros((rows,cols), dtype = int)
for i in range(rows):
    for j in range(cols):
        gmx[i,j] = gx[i,j]
xmin = gx.min()
xmax = gx.max()
for i in range(rows):
    for j in range(cols):
        gmx[i,j] = ((gmx[i,j]-xmin)/(xmax-xmin))*255
gmx = gmx.astype(np.uint8)
cv2.imwrite('hw1-2ii01.jpg', gmx)
cv2.imshow("image", gmx)
cv2.waitKey(0)

#apply Sobel operator Hy
Hy = np.array(
[[1, 2, 1],
[0, 0, 0],
[-1, -2, -1]]
)
gy = convolve2d(img_gb, Hy, mode='same')

#把原本有正有負的結果 轉成0-255印出
gmy = np.zeros((rows,cols), dtype = int)
for i in range(rows):
    for j in range(cols):
        gmy[i,j] = gy[i,j]
ymin = gy.min()
ymax = gy.max()
for i in range(rows):
    for j in range(cols):
        gmy[i,j] = ((gmy[i,j]-ymin)/(ymax-ymin))*255
gmy = gmy.astype(np.uint8)
cv2.imwrite('hw1-2ii02.jpg', gmy)
cv2.imshow("image", gmy)
cv2.waitKey(0)

gmx = gmx.astype(np.int32)
gmy = gmy.astype(np.int32)
#compute Ix^2 Iy^2 Ix*Iy of each pixel
#注意要用原本的gx gy算
Ix2 = np.zeros((rows,cols), dtype = np.int32)
Iy2 = np.zeros((rows,cols), dtype = np.int32)
Ixy = np.zeros((rows,cols), dtype = np.int32)
for i in range(rows):
    for j in range(cols):
        Ix2[i,j] = gx[i,j]**2
        Iy2[i,j] = gy[i,j]**2
        Ixy[i,j] = gx[i,j] * gy[i,j]

#compute sum value according to window size
Sx = np.zeros((rows,cols), dtype = np.int32)
Sy = np.zeros((rows,cols), dtype = np.int32)
Sxy = np.zeros((rows,cols), dtype = np.int32)

#zero padding
for i in range(1,rows-1):
    for j in range(1,cols-1):
        #window size = 3
        for x in range(-1,2):
            for y in range(-1,2):
                Sx[i,j] += Ix2[i+x,j+y]
                Sy[i,j] += Iy2[i+x,j+y]  #=sum_H10
                Sxy[i,j] += Ixy[i+x,j+y]

#compute matrix H according to window size
img_H = np.zeros((rows,cols,2,2), dtype = int)

#zero padding
for i in range(1,rows-1):
    for j in range(1,cols-1):
        img_H[i,j,0,0] = Sx[i,j]
        img_H[i,j,0,1] = Sxy[i,j]
        img_H[i,j,1,0] = Sxy[i,j]
        img_H[i,j,1,1] = Sy[i,j]

#compute R of each pixel and stored in img_R
arr_R = np.zeros((rows,cols), dtype = np.float64)

k = 0.04
for i in range(rows):
    for j in range(cols):
        det = linalg.det(img_H[i,j])
        trace = img_H[i,j].trace()
        arr_R[i,j] = round(det-k*(trace**2))

rmax = arr_R.max()
#output only R>threshold
#list for collecting possible corners
#(value,列編號,行編號,0表示沒檢查過)
list = []
threshold = 0.0005*rmax
img_R = np.zeros((rows,cols), dtype = np.uint8)
for i in range(rows):
    for j in range(cols):
        if arr_R[i,j] > threshold:
            list.append([arr_R[i,j], i, j, 0])
            img_R[i,j] = 255
img_R = img_R.astype(np.uint8)
cv2.imwrite('hw1-2iii.jpg', img_R)
cv2.imshow("image",img_R)
cv2.waitKey(0)

#non-maximum suppression
#sort the list in decreasing order
list.sort(reverse=True)
winsize = 5

#從r最大的開始檢查還沒看過的點
for c in list:
    if c[3] == 0:
        for k in list:
            if k[3] == 0:
                #check the value of other possible corner in the window
                dist = math.sqrt((k[1] - c[1])**2 + (k[2] - c[2])**2)
                #這些k都是c的鄰居且比c小 最後要濾掉 1用來表示看過且要suppress的點
                if (dist <= winsize and dist > 0):
                    k[3] = 1
        if c[0] < threshold:
            c[3] = 1

#最後留下的點放入lmlist
lmlist = filter(lambda x: x[3] == 0, list)
img_NMS = np.zeros((rows,cols), dtype = np.uint8)
for c in lmlist:
    i = c[1]
    j = c[2]
    img_NMS[i,j] = 255;
img_NMS = img_NMS.astype(np.uint8)
cv2.imwrite('hw1-2iv.jpg', img_NMS)
cv2.imshow("image",img_NMS)
cv2.waitKey(0)

#把corner疊在灰階圖上
img_gray = cv2.imread('hw1-2.jpg', cv2.IMREAD_GRAYSCALE)
img_final = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
for i in range(rows):
    for j in range(cols):
        if img_NMS[i,j] == 255:
            x = j
            y = i
            cv2.circle(img_final, (x,y), radius=1, color=(0, 0, 255), thickness=-1)

cv2.imwrite('hw1-2b.jpg', img_final)
cv2.imshow("image",img_final)
cv2.waitKey(0)

#hw1-2(c)(i)different windowsize
#compute sum value according to window size 5
Sx2 = np.zeros((rows,cols), dtype = np.int32)
Sy2 = np.zeros((rows,cols), dtype = np.int32)
Sxy2 = np.zeros((rows,cols), dtype = np.int32)

#zero padding
for i in range(2,rows-2):
    for j in range(2,cols-2):
        #window size = 5
        for x in range(-2,3):
            for y in range(-2,3):
                Sx2[i,j] += Ix2[i+x,j+y]
                Sy2[i,j] += Iy2[i+x,j+y]
                Sxy2[i,j] += Ixy[i+x,j+y]

#compute matrix H according to window size
img_H2 = np.zeros((rows,cols,2,2), dtype = int)

#zero padding
for i in range(2,rows-2):
    for j in range(2,cols-2):
        img_H2[i,j,0,0] = Sx2[i,j]
        img_H2[i,j,0,1] = Sxy2[i,j]
        img_H2[i,j,1,0] = Sxy2[i,j]
        img_H2[i,j,1,1] = Sy2[i,j]
#print(img_H[1,1])

#compute R of each pixel and stored in img_R
arr_R2 = np.zeros((rows,cols), dtype = np.float64)
k = 0.04
for i in range(rows):
    for j in range(cols):
        det = linalg.det(img_H2[i,j])
        trace = img_H2[i,j].trace()
        arr_R2[i,j] = round(det-k*(trace**2))

rmax = arr_R2.max()
#output only R>threshold
#list for collecting possible corners
#(value,列編號,行編號,0表示沒檢查過)
list2 = []
threshold2 = threshold
img_R2 = np.zeros((rows,cols), dtype = np.uint8)
for i in range(rows):
    for j in range(cols):
        if arr_R2[i,j] > threshold2:
            list2.append([arr_R2[i,j], i, j, 0])
            img_R2[i,j] = 255
img_R2 = img_R2.astype(np.uint8)
cv2.imwrite('hw1-2ci.jpg', img_R2)
cv2.imshow("image",img_R2)
cv2.waitKey(0)

#hw1-2(c)(ii)different threshold
#compute sum value according to window size
Sx3 = np.zeros((rows,cols), dtype = np.int32)
Sy3 = np.zeros((rows,cols), dtype = np.int32)
Sxy3 = np.zeros((rows,cols), dtype = np.int32)

#zero padding
for i in range(1,rows-1):
    for j in range(1,cols-1):
        #window size = 3
        for x in range(-1,2):
            for y in range(-1,2):
                Sx3[i,j] += Ix2[i+x,j+y]
                Sy3[i,j] += Iy2[i+x,j+y]  #=sum_H10
                Sxy3[i,j] += Ixy[i+x,j+y]

#compute matrix H according to window size
img_H3 = np.zeros((rows,cols,2,2), dtype = int)

#zero padding
for i in range(1,rows-1):
    for j in range(1,cols-1):
        img_H3[i,j,0,0] = Sx3[i,j]
        img_H3[i,j,0,1] = Sxy3[i,j]
        img_H3[i,j,1,0] = Sxy3[i,j]
        img_H3[i,j,1,1] = Sy3[i,j]
#print(img_H[1,1])

#compute R of each pixel and stored in img_R
arr_R3 = np.zeros((rows,cols), dtype = np.float64)
k = 0.04
for i in range(rows):
    for j in range(cols):
        det = linalg.det(img_H3[i,j])
        trace = img_H3[i,j].trace()
        arr_R3[i,j] = round(det-k*(trace**2))

rmax = arr_R3.max()
#output only R>threshold
#list for collecting possible corners
#(value,列編號,行編號,0表示沒檢查過)
list3 = []
threshold3 = 0.005*rmax
img_R3 = np.zeros((rows,cols), dtype = np.uint8)
for i in range(rows):
    for j in range(cols):
        if arr_R3[i,j] > threshold3:
            list.append([arr_R3[i,j], i, j, 0])
            img_R3[i,j] = 255
img_R3 = img_R3.astype(np.uint8)
cv2.imwrite('hw1-2cii.jpg', img_R3)
cv2.imshow("image",img_R3)
cv2.waitKey(0)
