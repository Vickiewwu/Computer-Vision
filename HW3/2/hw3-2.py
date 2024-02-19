import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from mpl_toolkits.mplot3d import Axes3D

#read images
image = cv2.imread('2-image.jpg')
image2 = cv2.imread('2-masterpiece.jpg')

data = image.reshape((-1,3))
data = data.astype(np.float32)
data2 = image2.reshape((-1,3))
data2 = data2.astype(np.float32)

k_list = [4,6,8]

#randomly initialize centers
def init_center(data,k):
    index = np.random.choice(len(data),k,replace=False)
    return data[index]

#assign points to the nearest cluster
def assign_cluster(data, center):
    cluster = np.argmin(np.linalg.norm(data[:,np.newaxis]-center,ord=2,axis=2),axis=1)
    return cluster

#update the center of the cluster by the updated points in the cluster
def update_center(data, clusters, k):
    new_center = np.array([data[clusters == j].mean(axis=0) for j in range(k)])
    return new_center

#apply kmeans algorithm on the image
def kmeans(data,k,iters,initial_num):
    best_center = best_cluster = 0
    best_dist = float('inf')
    for i in range(initial_num):
        center = init_center(data,k)
        for i in range(iters):
            cluster = assign_cluster(data,center)
            new_center = update_center(data,cluster,k)
            if np.all(center == new_center):
                break
            center = new_center
        #compute the sum of distance to the closest center
        dist = np.sum(np.min(np.linalg.norm(data[:, np.newaxis]-center, axis=2),axis=1))
        if dist < best_dist:
            best_dist = dist
            best_center = new_center
            best_cluster = cluster
    return best_center, best_cluster

for k in k_list:
    max iteration number=100 , initial guess = 50
    center, cluster = kmeans(data,k,100,50)
    center = np.uint8(center)
    seg_data = center[cluster.flatten()]
    seg_img = seg_data.reshape(image.shape)

    filename = f'./output/hw3-2-A_k={k}.jpg'
    cv2.imwrite(filename, seg_img)
    cv2.imshow(f'hw3-2-A_k={k}.jpg', seg_img)
    cv2.waitKey(0)

    center2, cluster2 = kmeans(data2,k,100,50)
    center2 = np.uint8(center2)
    seg_data2 = center2[cluster2.flatten()]
    seg_img2 = seg_data2.reshape(image2.shape)

    filename2 = f'./output/hw3-2-Am_k={k}.jpg'
    cv2.imwrite(filename2, seg_img2)
    cv2.imshow(f'hw3-2-Am_k={k}.jpg', seg_img2)
    cv2.waitKey(0)

#kmeans ++
#collecting initial centers
def kmeans_pp_init(data,k):
    center = [data[np.random.choice(len(data))]]

    while len(center) < k:
        dist = np.min(np.square(np.linalg.norm(data - np.array(center)[:, np.newaxis], axis=2)), axis=0)
        next_center = data[np.random.choice(len(data), p=dist / np.sum(dist))]
        center.append(next_center)

    return np.array(center)

#apply kmeans++ algorithm on the image
def kmeans_pp(data,k,iters):

    center = kmeans_pp_init(data,k)

    for i in range(iters):
        cluster = assign_cluster(data,center)
        new_center = update_center(data,cluster,k)

        if np.all(center == new_center):
            break
        center = new_center

    return center, cluster

for k in k_list:
    max iteration number=100
    center, cluster = kmeans_pp(data,k,100)
    center = np.uint8(center)
    seg_data = center[cluster.flatten()]
    seg_img = seg_data.reshape(image.shape)

    filename = f'./output/hw3-2-B_k={k}.jpg'
    cv2.imwrite(filename, seg_img)
    cv2.imshow(f'hw3-2-B_k={k}.jpg', seg_img)
    cv2.waitKey(0)

    center2, cluster2 = kmeans_pp(data2,k,100)
    center2 = np.uint8(center2)
    seg_data2 = center2[cluster2.flatten()]
    seg_img2 = seg_data2.reshape(image2.shape)

    filename2 = f'./output/hw3-2-Bm_k={k}.jpg'
    cv2.imwrite(filename2, seg_img2)
    cv2.imshow(f'hw3-2-Bm_k={k}.jpg', seg_img2)
    cv2.waitKey(0)

#create the image of pixel distributions in the R*G*B feature space before applying mean-shift
image = cv2.imread('2-image.jpg')
#image = cv2.imread('2-masterpiece.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = image_rgb[:,:,0].ravel()
g = image_rgb[:,:,1].ravel()
b = image_rgb[:,:,2].ravel()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(r, g, b, c=np.array([r, g, b]).T / 255.0, marker='o', alpha=0.5)

ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')

plt.savefig('./output/hw3-2-C2m.jpg')
plt.show()

#adopt uniform kernel in meanshift
def uni_kernel(d,bandwidth):
    c = 1
    return c if d<=bandwidth else 0

#find the closest center for each pixel
def find_closest_center(pixel, centers):
    distances = np.linalg.norm(centers - pixel, axis=1)
    return np.argmin(distances)

#apply meanshift algorithm on the image
def meanshift(image, x_seg, y_seg, bandwidth):
    rows, cols, _ = image.shape
    result = np.zeros_like(image, dtype=np.uint8)
    idx = np.zeros(rows * cols, dtype=np.int32)
    idx = idx.reshape((rows, cols))
    x_size = cols // x_seg
    y_size = rows // y_seg

    # Collect RGB values of center points
    center_points = []
    for i in range(0, rows, y_size):
        for j in range(0, cols, x_size):
            block = image[i:i + y_size, j:j + x_size].reshape(-1, 3)
            center = np.mean(block, axis=0)
            center_points.append(center)

    center_points = np.array(center_points)
    print(center_points)
    print(len(center_points))

    #assgin center
    for i in range(rows):
        for j in range(cols):
            pixel = image[i, j]
            closest_center_idx = find_closest_center(pixel, center_points)
            idx[i,j] = closest_center_idx
    # Apply mean shift in RGB space
    for i in range(center_points.shape[0]):
        shifting = 1
        ncenter = center_points[i]
        center = center_points[i]
        while True :
            center = ncenter
            shift = np.zeros_like(center, dtype=np.float32)
            ncenter = shift
            weight_sum = 0

            for j in range(center_points.shape[0]):
                d = np.sqrt(np.sum((center - center_points[j]) ** 2))
                w = uni_kernel(d, bandwidth)
                shift += w * center_points[j]
                weight_sum += w

            shift /= weight_sum
            ncenter = shift

            if np.linalg.norm(ncenter - center) < 0.001:
                center_points[i] = ncenter
                for k in range(i-1):
                    if np.linalg.norm(center_points[k] - ncenter) < 0.1:
                        center_points[i] = center_points[k]

                break

    # Assign each pixel in the image to its representing center
    for i in range(rows):
        for j in range(cols):
            pixel = image[i, j]
            closest_center_idx = idx[i, j]
            closest_center = center_points[closest_center_idx]
            result[i, j] = closest_center

    return result

#meanshift algorithm considering spatial distance
def meanshift_spatial(image, x_seg, y_seg, hs , hr):
    rows, cols, _ = image.shape
    result = np.zeros_like(image, dtype=np.uint8)
    idx = np.zeros(rows * cols, dtype=np.int32)
    idx = idx.reshape((rows, cols))
    x_size = cols // x_seg
    y_size = rows // y_seg

    center_points2=[]
    for i in range(0, rows, y_size):
        for j in range(0, cols, x_size):
            block = image[i:i + y_size, j:j + x_size].reshape(-1, 3)
            center_rgb = np.mean(block, axis=0)
            center_coords = (i + y_size // 2, j + x_size // 2)
            center_points2.append((center_rgb, center_coords))
            center_points = np.array(center_points2, dtype=[('rgb', '3float32'), ('coords', '2float32')])

    for i in range(rows):
        for j in range(cols):
            pixel = image[i, j]
            closest_center_idx = find_closest_center(pixel, center_points['rgb'])
            idx[i,j] = closest_center_idx


    # Apply mean shift in RGB space
    for i in range(center_points.shape[0]):
        ncenter= center_points[i]
        center = center_points[i]
        while True :
            center = ncenter
            shift = np.zeros_like(center)
            ncenter = shift
            weight_sum = 0

            for j in range(center_points.shape[0]):
                dr = np.sqrt(np.sum((center['rgb'] - center_points[j]['rgb']) ** 2))
                if dr <= hr:
                    ds = np.linalg.norm(center['coords'] - center_points[j]['coords'])
                    w = uni_kernel(ds, hs)
                    if w == 1:
                        shift['rgb'] += center_points[j]['rgb']
                        shift['coords'] += center_points[j]['coords']
                        weight_sum += w

            shift['rgb']  /= weight_sum
            shift['coords']  /= weight_sum
            ncenter = shift

            if np.linalg.norm(ncenter['rgb'] - center['rgb']) < 0.001 and np.linalg.norm(ncenter['coords'] - center['coords']) < 0.001:
                center_points[i] = ncenter
                for k in range(i-1):
                    if np.linalg.norm(center_points[k]['rgb'] - ncenter['rgb']) <0.1 and np.linalg.norm(center_points[k]['coords'] - ncenter['coords']) < 0.1:
                        center_points[i] = center_points[k]

                break

    # Assign each pixel in the image its repesenting center
    for i in range(rows):
        for j in range(cols):
            pixel = image[i, j]
            closest_center_idx = idx[i, j]
            closest_center= center_points[closest_center_idx]
            result[i, j] = closest_center['rgb']

    return result

#Parameters setting
bandwidth = 30.0
x_seg = 20
y_seg = 20
hs = 200
hr = 20

seg_img = meanshift(image, x_seg, y_seg, bandwidth)
filename = f'./output/hw3-2-C_h={bandwidth}.jpg'
cv2.imwrite(filename, seg_img)
cv2.imshow(f'hw3-2-C_h={bandwidth}.jpg', seg_img)
cv2.waitKey(0)

seg_img_sr = meanshift_spatial(image, x_seg, y_seg, hs , hr)
filename = f'./output/hw3-2-D_h={hr}.jpg'
cv2.imwrite(filename, seg_img_sr)
cv2.imshow(f'hw3-2-D_h={hr}.jpg', seg_img_sr)
cv2.waitKey(0)

seg_image_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

#create the image of pixel distributions in the R*G*B feature space after applying mean-shift
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

R = seg_image_rgb[:,:,0].ravel()
G = seg_image_rgb[:,:,1].ravel()
B = seg_image_rgb[:,:,2].ravel()

ax.scatter(r, g, b, c=np.array([R, G, B]).T / 255.0, marker='o', alpha=0.5)

ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')

plt.savefig('./output/hw3-2-C3m_h=50.jpg')
plt.show()
