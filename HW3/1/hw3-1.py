import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import math
import random

#讀圖片且轉成灰階 要跟code放在同一個資料夾內
img01 = cv2.imread('1-book1.jpg')
img02 = cv2.imread('1-image.jpg')
img03 = cv2.imread('1-book2.jpg')
img04 = cv2.imread('1-book3.jpg')

img01g = cv2.imread('1-book1.jpg',cv2.IMREAD_GRAYSCALE)
img02g = cv2.imread('1-image.jpg',cv2.IMREAD_GRAYSCALE)
img03g = cv2.imread('1-book2.jpg',cv2.IMREAD_GRAYSCALE)
img04g = cv2.imread('1-book3.jpg',cv2.IMREAD_GRAYSCALE)

#建立sift
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

#取得特徵點及descriptor
keypoints01, descriptors01 = sift.detectAndCompute(img01g, None)
keypoints02, descriptors02 = sift.detectAndCompute(img02g, None)
keypoints03, descriptors03 = sift.detectAndCompute(img03g, None)
keypoints04, descriptors04 = sift.detectAndCompute(img04g, None)

n_keypoints01 = len(keypoints01)
n_keypoints02 = len(keypoints02)
n_keypoints03 = len(keypoints03)
n_keypoints04 = len(keypoints04)
print(n_keypoints01)
print(n_keypoints02)
print(n_keypoints03)
print(n_keypoints04)

#在圖2中找跟descriptor最相近的keypoint
def find_match(n_keypoints01,descriptors01,n_keypoints02,descriptors02,t):
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
        if (min_dist1/min_dist2)<t:
            match[k1] = [min_b1, min_dist1]
    return match
  
#找出圖1及圖2中的對應特徵點
match1 = find_match(n_keypoints01,descriptors01,n_keypoints02,descriptors02,0.7)
print(len(match1))

#Draw the matches
matches01 = [[cv2.DMatch(k, v[0], 0) for k, v in match1.items()]]
img_match = cv2.drawMatchesKnn(img01, keypoints01, img02, keypoints02, matches01,None, matchColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert BGR image to RGB
img_match_rgb = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
plt.imshow(img_match_rgb)
plt.savefig('./output/hw3-1-A1.jpg')
plt.show()

#找出圖3及圖2中的對應特徵點
match2 = find_match(n_keypoints03,descriptors03,n_keypoints02,descriptors02,0.7)
print(len(match2))

#Draw the matches
matches02 = [[cv2.DMatch(k, v[0], 0) for k, v in match2.items()]]
img_match2 = cv2.drawMatchesKnn(img03, keypoints03, img02, keypoints02, matches02,None, matchColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert BGR image to RGB
img_match2_rgb = cv2.cvtColor(img_match2, cv2.COLOR_BGR2RGB)
plt.imshow(img_match2_rgb)
plt.savefig('./output/hw3-1-A2.jpg')
plt.show()

#找出圖4及圖2中的對應特徵點
match3 = find_match(n_keypoints04,descriptors04,n_keypoints02,descriptors02,0.5)
print(len(match3))

#Draw the matches
matches03 = [[cv2.DMatch(k, v[0], 0) for k, v in match3.items()]]
img_match3 = cv2.drawMatchesKnn(img04, keypoints04, img02, keypoints02, matches03,None, matchColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert BGR image to RGB
img_match3_rgb = cv2.cvtColor(img_match3, cv2.COLOR_BGR2RGB)
plt.imshow(img_match3_rgb)
plt.savefig('./output/hw3-1-A3.jpg')
plt.show()

#透過對應點找出Homography矩陣
def Find_Homography(src,tar):
    A = []
    for i in range(len(src)):
        x, y = src[i]
        u, v = tar[i]
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u, -u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v, -v])

    A = np.array(A)
    _, _, VT = np.linalg.svd(A)
    H = VT[-1].reshape(3, 3)

    return H / H[2,2]

#把輸入的點座標透過Homography矩陣轉換成新座標
def transform(points,H):
    num = len(points)
    transp = np.zeros_like(points[:, :2])
    for i in range(num):
        hpoints = np.hstack((points[i, :2],1))
        thpoints = np.dot(H, hpoints)
        transp[i] = thpoints[:2]/thpoints[2]
    return transp

#計算圖2中的特徵點跟轉換後的圖1特徵點之間的距離
def get_dist(matches, H,keypoints01,keypoints02):
    num = len(matches)
    p1 = np.array([[keypoints01[m.queryIdx].pt[0], keypoints01[m.queryIdx].pt[1]] for m in matches])
    p2 = np.array([[keypoints02[m.trainIdx].pt[0], keypoints02[m.trainIdx].pt[1]] for m in matches])

    #對所有圖1中的特徵點做轉換
    t_p1 = transform(p1, H)

    dist = np.linalg.norm(p2 - t_p1, axis=1) ** 2

    return dist

#透過RANSAC找出最好的特徵點
def RANSAC(matches,keypoints01,keypoints02,iters,threshold):
    best_num = 0
    n = 4

    for i in range(iters):
        #隨機取出4組對應的特徵點
        indices = random.sample(range(len(matches)), n)
        sample = [matches[idx] for idx in indices]
        src = np.float32([keypoints01[m.queryIdx].pt for m in sample]).reshape(-1, 2)
        dst = np.float32([keypoints02[m.trainIdx].pt for m in sample]).reshape(-1, 2)
        
        #透過這些點算出Homography矩陣
        H = Find_Homography(src,dst)

        #計算圖2中的特徵點跟轉換後的圖1特徵點之間的距離
        dist = get_dist(matches,H,keypoints01,keypoints02)

        #計算認同這個Homography矩陣的特徵點有多少，用來評估這個Homography矩陣好不好
        inlier = np.array(matches)[np.where(dist < threshold)[0]]
        num = len(inlier)

        #保留有最多inlier的Homography矩陣及對應的inlier
        if num > best_num:
            best_num = num
            best_H = H.copy()
            best_inlier = inlier.copy()
            best_inlier = best_inlier.tolist()

    print(best_num)
    return best_H, best_inlier

#轉成DMatch物件所構成的list
matches1 = [cv2.DMatch(k, v[0], 0) for k, v in match1.items()]
matches2 = [cv2.DMatch(k, v[0], 0) for k, v in match2.items()]
matches3 = [cv2.DMatch(k, v[0], 0) for k, v in match3.items()]

#把輸入的點座標透過Homography矩陣轉換成新座標
def transform_p(points, H):
    num = len(points)
    transp = np.zeros_like(points[:, :2])
    for i in range(num):
        hpoints = np.hstack((points[i, :2], 1))
        thpoints = np.dot(H, hpoints)
        transp[i] = thpoints[:2] / thpoints[2]
    return transp

#把長方形(書本邊界)畫出來
def draw_rectangle(img, corners):
    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color=(255, 0, 0), thickness=5)
    return img

#找出圖1到圖2的homography矩陣及inliner
H1, inlier1 = RANSAC(matches1,keypoints01,keypoints02,500,0.5)
inlier_matches1 = [cv2.DMatch(m.queryIdx, m.trainIdx, m.distance) for m in inlier1]
#圖1中的書本四個角落座標
corners1 = np.array([[124,124], [1260,98], [1256,1002], [148,990]], dtype=np.int32)
#將圖1中的書本四個角落座標透過homography矩陣轉換到圖2中的座標
hcorners1 = transform(corners1,H1)
img02_1 = cv2.imread('1-image.jpg')
#在圖1中框出書本周圍(透過四個座標)
img01 = draw_rectangle(img01, corners1)
#在圖2中框出書本周圍(透過轉換後四個座標)
img02_1 = draw_rectangle(img02_1, hcorners1)
#畫出圖1到圖2特徵點對應線條
img_matches = cv2.drawMatchesKnn(img01, keypoints01, img02_1, keypoints02, [inlier_matches1],None, matchColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_match_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
plt.imshow(img_match_rgb)
plt.savefig('./output/hw3-1-B1.jpg')
plt.show()

#找出圖3到圖2的homography矩陣及inliner
H2, inlier2 = RANSAC(matches2,keypoints03,keypoints02,500,0.5)
inlier_matches2 = [cv2.DMatch(m.queryIdx, m.trainIdx, m.distance) for m in inlier2]
#圖3中的書本四個角落座標
corners2 = np.array([[130,80], [1356,68], [1368,1048], [132,1044]], dtype=np.int32)
#將圖3中的書本四個角落座標透過homography矩陣轉換到圖2中的座標
hcorners2 = transform(corners2,H2)
print(hcorners2)
img02_2 = cv2.imread('1-image.jpg')
#在圖3中框出書本周圍(透過四個座標)
img03 = draw_rectangle(img03, corners2)
#在圖2中框出書本周圍(透過轉換後四個座標)
img02_2 = draw_rectangle(img02_2, hcorners2)
#畫出圖3到圖2特徵點對應線條
img_matches2 = cv2.drawMatchesKnn(img03, keypoints03, img02_2, keypoints02, [inlier2],None, matchColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_match_rgb2 = cv2.cvtColor(img_matches2, cv2.COLOR_BGR2RGB)
plt.imshow(img_match_rgb2)
plt.savefig('./output/hw3-1-B2.jpg')
plt.show()

#找出圖4到圖2的homography矩陣及inliner
H3, inlier3 = RANSAC(matches3,keypoints04,keypoints02,500,0.5)
inlier_matches2 = [cv2.DMatch(m.queryIdx, m.trainIdx, m.distance) for m in inlier3]
#圖4中的書本四個角落座標
corners3 = np.array([[82,132], [1290,126], [1298,988], [86,978]], dtype=np.int32)
#將圖4中的書本四個角落座標透過homography矩陣轉換到圖2中的座標
hcorners3 = transform(corners3,H3)

img02_3 = cv2.imread('1-image.jpg')
#在圖4中框出書本周圍(透過四個座標)
img04 = draw_rectangle(img04, corners3)
#在圖2中框出書本周圍(透過轉換後四個座標)
img02_3 = draw_rectangle(img02_3, hcorners3)
#畫出圖4到圖2特徵點對應線條
img_matches3 = cv2.drawMatchesKnn(img04, keypoints04, img02_3, keypoints02, [inlier3],None, matchColor=(0, 0, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_match_rgb3 = cv2.cvtColor(img_matches3, cv2.COLOR_BGR2RGB)
plt.imshow(img_match_rgb3)
plt.savefig('./output/hw3-1b-B3.jpg')
plt.show()

#畫出圖2中的原特徵點及圖1、3、4轉換後的對應特徵點差距(經由SIFT找出的Homography矩陣轉換)
img02_p = cv2.imread('1-image.jpg')
points1 =  np.float32([keypoints01[m.queryIdx].pt for m in matches1]).reshape(-1, 2)
t_points1 = transform_p(points1, H1)
o_points1 = np.float32([keypoints02[m.trainIdx].pt for m in matches1]).reshape(-1, 2)

points2 =  np.float32([keypoints03[m.queryIdx].pt for m in matches2]).reshape(-1, 2)
t_points2 = transform_p(points2, H2)
o_points2 = np.float32([keypoints02[m.trainIdx].pt for m in matches2]).reshape(-1, 2)

points3 =  np.float32([keypoints04[m.queryIdx].pt for m in matches3]).reshape(-1, 2)
t_points3 = transform_p(points3, H3)
o_points3 = np.float32([keypoints02[m.trainIdx].pt for m in matches3]).reshape(-1, 2)

for i in range(len(o_points1)):
    pt1 = (int(o_points1[i, 0]), int(o_points1[i, 1]))
    pt2 = (int(t_points1[i, 0]), int(t_points1[i, 1]))

    cv2.arrowedLine(img02_p, pt1,pt2,(255,0, 0), 2)

for i in range(len(o_points2)):
    pt1 = (int(o_points2[i, 0]), int(o_points2[i, 1]))
    pt2 = (int(t_points2[i, 0]), int(t_points2[i, 1]))

    cv2.arrowedLine(img02_p, pt1,pt2,(0,255, 0), 2)

for i in range(len(o_points3)):
    pt1 = (int(o_points3[i, 0]), int(o_points3[i, 1]))
    pt2 = (int(t_points3[i, 0]), int(t_points3[i, 1]))

    cv2.arrowedLine(img02_p, pt1,pt2,(0,0, 255), 2)

filename3 = f'./output/hw3-1-B4.jpg'
cv2.imwrite(filename3, img02_p)
cv2.imshow('arrows', img02_p)
cv2.waitKey(0)

#畫出圖2中的原特徵點及圖1、3、4轉換後的對應特徵點差距(經由RANSAC找出的Homography矩陣轉換)
img02_p = cv2.imread('1-image.jpg')

points1 =  np.float32([keypoints01[m.queryIdx].pt for m in inlier1]).reshape(-1, 2)
t_points1 = transform_p(points1, H1)
o_points1 = np.float32([keypoints02[m.trainIdx].pt for m in inlier1]).reshape(-1, 2)

points2 =  np.float32([keypoints03[m.queryIdx].pt for m in inlier2]).reshape(-1, 2)
t_points2 = transform_p(points2, H2)
o_points2 = np.float32([keypoints02[m.trainIdx].pt for m in inlier2]).reshape(-1, 2)

points3 =  np.float32([keypoints04[m.queryIdx].pt for m in inlier3]).reshape(-1, 2)
t_points3 = transform_p(points3, H3)
o_points3 = np.float32([keypoints02[m.trainIdx].pt for m in inlier3]).reshape(-1, 2)

for i in range(len(o_points1)):
    pt1 = (int(o_points1[i, 0]), int(o_points1[i, 1]))
    pt2 = (int(t_points1[i, 0]), int(t_points1[i, 1]))

    cv2.arrowedLine(img02_p, pt1,pt2,(255,0, 0), 2)

for i in range(len(o_points2)):
    pt1 = (int(o_points2[i, 0]), int(o_points2[i, 1]))
    pt2 = (int(t_points2[i, 0]), int(t_points2[i, 1]))

    cv2.arrowedLine(img02_p, pt1,pt2,(0,255, 0), 2)

for i in range(len(o_points3)):
    pt1 = (int(o_points3[i, 0]), int(o_points3[i, 1]))
    pt2 = (int(t_points3[i, 0]), int(t_points3[i, 1]))

    cv2.arrowedLine(img02_p, pt1,pt2,(0,0, 255), 2)

filename3 = f'./output/hw3-1-B5.jpg'
cv2.imwrite(filename3, img02_p)
cv2.imshow('arrows', img02_p)
cv2.waitKey(0)
