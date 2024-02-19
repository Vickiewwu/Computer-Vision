import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#讀圖片且轉成灰階 要跟code放在同一個資料夾內
img = cv2.imread('hw1-1.jpg',cv2.IMREAD_GRAYSCALE)
#print(img)
arr_i = img.flatten()
#print(arr_i)

#建立全0陣列 要用來統計每個灰階值出現幾次
arr = np.zeros(256,dtype=int)
#存放cumulative的灰階值
arr_c = np.zeros(256,dtype=int)
#存放equalize後的灰階值
arr_t = np.zeros(256,dtype=int)
h,w = img.shape[0:2]

#把灰階值出現次數存入arr
for i in range(h):
    for j in range(w):
        v = img[i,j]
        arr[v] += 1
#np.set_printoptions(formatter={'int': '{:d}'.format})
#print(arr)

#檢查arr內是否包含每一個Pixel的值(加起來等於h*w)
#sum_arr = 0
#for i in range(256):
        #sum_arr +=arr[i]
#print(sum_arr)

#計算cumulative灰階值
for i in range(256):
    sum_i = 0
    for j in range(i+1):
        sum_i += arr[j]
    arr_c[i] = sum_i
#print(arr_c)

#進行equalization
for i in range(256):
    arr_t[i] = round((255/(h*w))*arr_c[i])
#print(arr_t)

#建立陣列儲存新圖片的灰階值
arr_e = np.zeros((h,w), dtype = int)

#更新灰階值 k是新的灰階值
for i in range(h):
    for j in range(w):
        k = img[i,j]
        arr_e[i][j] = arr_t[k]
#print(arr_e)
img_e = arr_e.astype(np.uint8)

#把兩張圖依水平方向放在一起
imgs = np.hstack((img,img_e))
cv2.imwrite('hw1-101.jpg', imgs)
cv2.imshow("image", imgs)
cv2.waitKey(0)
arr_e = arr_e.flatten()

#繪製原本灰階圖的histogram
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].hist(arr_i, bins=256, range=None, density=None, cumulative=False, histtype='bar', align='mid', orientation='vertical', rwidth=None, color=None,  label=None, stacked=False)

#繪製更新後灰階圖的histogram
ax[1].hist(arr_e, bins=256, range=None, density=None, cumulative=False, histtype='bar', align='mid', orientation='vertical', rwidth=None, color=None, label=None, stacked=False)

plt.tight_layout()
plt.savefig('hw1-102.jpg')
plt.show()
