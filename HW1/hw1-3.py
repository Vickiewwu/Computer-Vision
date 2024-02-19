import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import math

#讀圖片且轉成灰階 要跟code放在同一個資料夾內
img01 = cv2.imread('hw1-3-1.jpg',cv2.IMREAD_GRAYSCALE)
img02 = cv2.imread('hw1-3-2.jpg',cv2.IMREAD_GRAYSCALE)

#建立sift
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

#取得特徵點及descriptor
keypoints01, descriptors01 = sift.detectAndCompute(img01, None)
keypoints02, descriptors02 = sift.detectAndCompute(img02, None)
n_keypoints01 = len(keypoints01)
n_keypoints02 = len(keypoints02)

#在圖2中找兩個跟descriptor最相近的keypoint
min_1 = {}
min_2 = {}
match = {}
for k1, A in zip(range(n_keypoints01), descriptors01):
    min_dist1 = float('inf')
    min_b1 = None

    for k2, B in zip(range(n_keypoints02), descriptors02):
        dist = np.linalg.norm(A - B)
        if dist < min_dist1:
            min_dist1 = dist
            min_b1 = k2
    min_1[k1] = min_b1

    min_dist2 = float('inf')
    min_b2 = None
    for k2, B in zip(range(n_keypoints02), descriptors02):
        if k2 != min_b1:
            dist = np.linalg.norm(A - B)
            if dist < min_dist2:
                min_dist2 = dist
                min_b2 = k2
    min_2[k1] = min_b2
    #只接受d(v,b1)/d(v,d2)<1/2的點
    if (min_dist1/min_dist2)<0.5:
        match[k1] = [min_b1, min_dist1]
#print(len(match))

#sort the match in ascending order
match_s = {k: v for k, v in sorted(match.items(), key=lambda item: [item[1][1]])}

#new dictionary for at most 20 matches of each objects
match_20 = {}
front = 0.33*img02.shape[0]
medium = 0.66*img02.shape[0]
bottom = img02.shape[0]

A = B = C = 0
for k1, (k2, d) in match_s.items():
    y = keypoints02[k2].pt[1]
    if y <= front and A<20:
        match_20[k1] = (k2,y)
        A+=1
    elif front<y<=medium  and B<20:
        match_20[k1] = (k2,y)
        B+=1
    elif medium<y<=bottom and C<20:
        match_20[k1] = (k2,y)
        C+=1

# turn into 3 channels
img01_bgr= cv2.cvtColor(img01, cv2.COLOR_GRAY2BGR)
img02_bgr = cv2.cvtColor(img02, cv2.COLOR_GRAY2BGR)

# Draw the matches
matches = [[cv2.DMatch(k, v[0], 0) for k, v in match_20.items()]]
img_match = cv2.drawMatchesKnn(img01_bgr, keypoints01, img02_bgr, keypoints02, matches,None, matchColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert BGR image to RGB
img_match_rgb = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
plt.imshow(img_match_rgb)
plt.savefig('hw1-3c.jpg')
plt.show()

# Resize the images
img01_2x = cv2.resize(img01, ((img01.shape[1])*2, (img01.shape[0])*2))

#建立sift
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

#取得特徵點及descriptor
keypoints01, descriptors01 = sift.detectAndCompute(img01_2x, None)
keypoints02, descriptors02 = sift.detectAndCompute(img02, None)
n_keypoints01 = len(keypoints01)
n_keypoints02 = len(keypoints02)

#在圖2中找兩個跟descriptor最相近的keypoints
min_1 = {}
min_2 = {}
match = {}
for k1, A in zip(range(n_keypoints01), descriptors01):
    min_dist1 = float('inf')
    min_b1 = None

    for k2, B in zip(range(n_keypoints02), descriptors02):
        dist = np.linalg.norm(A - B)
        if dist < min_dist1:
            min_dist1 = dist
            min_b1 = k2
    min_1[k1] = min_b1

    min_dist2 = float('inf')
    min_b2 = None
    for k2, B in zip(range(n_keypoints02), descriptors02):
        if k2 != min_b1:
            dist = np.linalg.norm(A - B)
            if dist < min_dist2:
                min_dist2 = dist
                min_b2 = k2
    min_2[k1] = min_b2
    #只接受d(v,b1)/d(v,d2)<1/2的點
    if (min_dist1/min_dist2)<0.5:
        match[k1] = [min_b1, min_dist1]
#print(len(match))

#sort the match in ascending order
match_s = {k: v for k, v in sorted(match.items(), key=lambda item: [item[1][1]])}

#new dictionary for at most 20 matches of each objects
match_20 = {}
front = 0.33*img02.shape[0]
medium = 0.66*img02.shape[0]
bottom = img02.shape[0]

A = B = C = 0
for k1, (k2, d) in match_s.items():
    y = keypoints02[k2].pt[1]
    if y <= front and A<20:
        match_20[k1] = (k2,y)
        A+=1
    elif front<y<=medium  and B<20:
        match_20[k1] = (k2,y)
        B+=1
    elif medium<y<=bottom and C<20:
        match_20[k1] = (k2,y)
        C+=1

# turn into 3 channels
img01_bgr= cv2.cvtColor(img01_2x, cv2.COLOR_GRAY2BGR)
img02_bgr = cv2.cvtColor(img02, cv2.COLOR_GRAY2BGR)

# Draw the matches
#matches is a list of matches
matches = [[cv2.DMatch(k, v[0], 0) for k, v in match_20.items()]]
img_match = cv2.drawMatchesKnn(img01_bgr, keypoints01, img02_bgr, keypoints02, matches,None, matchColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert BGR image to RGB
img_match_rgb = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
plt.imshow(img_match_rgb)
plt.savefig('hw1-3d.jpg')
plt.show()
