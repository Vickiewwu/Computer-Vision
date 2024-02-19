import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

#read input files and record the points
def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        num = int(file.readline().strip())
        for i in range(num):
            x, y = map(float, file.readline().strip().split())
            points.append([x, y, 1])
    return np.array(points)

points1 = read_points("./assets/pt_2D_1.txt")  #shape(59,3)
points2 = read_points("./assets/pt_2D_2.txt")

def compute_f(points1, points2):  #p2->p1的F
    #construct matrix A
    x1, y1, _ = points1.T
    x2, y2, _ = points2.T
    A = np.column_stack((x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, np.ones(len(x1))))

    # Solve for the fundamental matrix using SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3) #最小sungular value所對向量

    # Enforce the rank-2 constraint by making the last singular value zero
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  #讓rank=2
    F = U @ np.diag(S) @ Vt

    return F

# Normalized version
def nor_f(points1, points2):
    n = points1.shape[0]
    points1_xy = points1[:, 0:2]
    points2_xy = points2[:, 0:2]

    #取得座標平均
    points1_mean = np.mean(points1_xy,axis=0)
    points2_mean = np.mean(points2_xy,axis=0)

    #計算距離
    points1_dist = points1_xy - points1_mean
    points2_dist = points2_xy - points2_mean

    # 計算scale
    scale1 = np.sqrt(2 / (np.sum(points1_dist**2)/n ))
    scale2 = np.sqrt(2 / (np.sum(points2_dist**2)/n ))

    T1 = np.array([
        [scale1, 0, -points1_mean[0] * scale1],
        [0, scale1, -points1_mean[1] * scale1],
        [0, 0, 1]
    ])
    
    T2 = np.array([
        [scale2, 0, -points2_mean[0] * scale2],
        [0, scale2, -points2_mean[1] * scale2],
        [0, 0, 1]
    ])

    #轉換到新座標
    q1 = (T1 @ points1.T).T   #59*3
    q2 = (T2 @ points2.T).T

    # 計算fundamental matrix
    F = compute_f(q1, q2)

    #denormalize
    Fn = T1.T @ F @ T2

    return Fn

F = compute_f(points1, points2)
print("Fundamental Matrix:\n", F)

Fn = nor_f(points1, points2)
print("Normalized Fundamental Matrix:\n", Fn)

def get_distance(points1, points2, F):
    lines = F @ points2.T
    n = points1.shape[0]

    A = lines[0]
    B = lines[1]
    C = lines[2]
    x = points1[:, 0]
    y = points1[:, 1]

    # Compute distances
    distances = np.abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)

    dis_sum = np.sum(distances)

    return dis_sum/n

F_image1 = get_distance(points1, points2, F)
print(F_image1,"\n" )
F_image2 = get_distance(points2, points1, F.T)
print(F_image2,"\n" )
Fn_image1 = get_distance(points1, points2, Fn)
print(Fn_image1,"\n" )
Fn_image2 = get_distance(points2, points1, Fn.T)
print(Fn_image2,"\n" )


# calculate the epipolar lines
lines1 = F @ points2.T
numl =lines1.shape[1]

file_path="./output"
image1 = cv2.imread('./assets/image1.jpg')
image2 = cv2.imread('./assets/image2.jpg')

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(numl)]

for i in range(numl):
    A = lines1[0, i]
    B = lines1[1, i]
    C = lines1[2, i]
    W = image1.shape[1]
    y1 = -C/B
    y2 = -(A * W + C) / B
    cv2.line(image1, (0, int(y1)), (W, int(y2)), colors[i], 1)
    cv2.circle(image1, (int(points1[i, 0]), int(points1[i, 1])), 3, colors[i], -1)

cv2.imwrite(os.path.join(file_path,"wo_normalized_img1.png"),image1)
cv2.imshow('wo_normalized version on image 1', image1)
cv2.waitKey(0)

lines2 = F.T @ points1.T
for i in range(numl):
    A = lines2[0, i]
    B = lines2[1, i]
    C = lines2[2, i]
    W = image2.shape[1]
    y1 = -C/B
    y2 = -(A * W + C) / B
    cv2.line(image2, (0, int(y1)), (W, int(y2)), colors[i], 1)
    cv2.circle(image2, (int(points2[i, 0]), int(points2[i, 1])), 3, colors[i], -1)
cv2.imwrite(os.path.join(file_path,"wo_normalized_img2.png"),image2)
cv2.imshow('wo_normalized version on image 2', image2)
cv2.waitKey(0)

# calculate the epipolar lines
lines1 = Fn @ points2.T
lines2 = Fn.T @ points1.T
numl =lines1.shape[1]
image1 = cv2.imread('./assets/image1.jpg')
image2 = cv2.imread('./assets/image2.jpg')

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(numl)]

for i in range(numl):
    A = lines1[0, i]
    B = lines1[1, i]
    C = lines1[2, i]
    W = image1.shape[1]
    y1 = -C/B
    y2 = -(A * W + C) / B
    cv2.line(image1, (0, int(y1)), (W, int(y2)), colors[i], 1)
    cv2.circle(image1, (int(points1[i, 0]), int(points1[i, 1])), 3, colors[i], -1)
cv2.imwrite(os.path.join(file_path,"normalized_img1.png"),image1)
cv2.imshow('normalized version on image 1', image1)
cv2.waitKey(0)

for i in range(numl):
    A = lines2[0, i]
    B = lines2[1, i]
    C = lines2[2, i]
    W = image2.shape[1]
    y1 = -C/B
    y2 = -(A * W + C) / B
    cv2.line(image2, (0, int(y1)), (W, int(y2)), colors[i], 1)
    cv2.circle(image2, (int(points2[i, 0]), int(points2[i, 1])), 3, colors[i], -1)
cv2.imwrite(os.path.join(file_path,"normalized_img2.png"),image2)
cv2.imshow('normalized version on image 2', image2)
cv2.waitKey(0)
