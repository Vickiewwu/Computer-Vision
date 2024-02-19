import cv2
import numpy as np
import copy
import os
# mouse callback function
def mouse_callback(event, x, y, flags, param):

    global corner_tar
    if event == cv2.EVENT_LBUTTONDOWN:
        if(len(corner_tar)<4):
            corner_tar.append((x,y))

def Find_Homography(src,tar):
    #given corresponding point and return the homagraphic matrix
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

if __name__=="__main__":
    img_src = cv2.imread("./assets/post.png")
    src_H,src_W,_=img_src.shape
    corner_src=[(0,0),(src_W-1,0),(src_W-1,src_H-1),(0,src_H-1)]
    print("src=",corner_src)
    file_path="./output"
    img_tar = cv2.imread("./assets/display.jpg")
    tar_H,tar_W,_=img_tar.shape
    re_h = int(img_tar.shape[0] * 0.5)
    re_w = int(img_tar.shape[1] * 0.5)
    img_tar = cv2.resize(img_tar, (re_w, re_h))
    cv2.namedWindow("Interative window")
    cv2.setMouseCallback("Interative window", mouse_callback)
    cv2.setMouseCallback("Interative window", mouse_callback)

    corner_tar=[]
    while True:
        fig=img_tar.copy()
        key = cv2.waitKey(1) & 0xFF


        if(len(corner_tar)==4):
            # implement the inverse homography mapping and bi-linear interpolation
            pass

        # quit
        if key == ord("q"):
            break

        # reset the corner_list
        if key == ord("r"):
            corner_tar=[]
          
        # show the corner list
        if key == ord("p"):
            print(corner_tar)
        cv2.imshow("Interative window", fig)

    cv2.destroyAllWindows()
    #x_points, y_points = zip(*corner_tar)
    #print("1x_points,y_points=",x_points,y_points)

    H = Find_Homography(corner_src,corner_tar)
    print(H)

    def warp(img_src, H, tar_h,tar_w):
        dst = np.zeros((tar_h, tar_w, img_src.shape[2]), dtype=img_src.dtype)

        H_inv = np.linalg.inv(H)

        for y in range(tar_h):
            for x in range(tar_w):
                # calculate the src coordinates
                src_co = H_inv @ np.array([x, y, 1])
                src_x, src_y, src_w = src_co / src_co[2]

                # check boundary
                if 0 <= src_x < img_src.shape[1] - 1 and 0 <= src_y < img_src.shape[0] - 1:
                    x0, y0 = int(src_x), int(src_y)
                    x1, y1 = x0 + 1, y0 + 1
                    dx, dy = src_x - x0, src_y - y0

                    # bilinear interpolation
                    pixel = (
                    (1 - dx) * (1 - dy) * img_src[y0, x0] +
                    dx * (1 - dy) * img_src[y0, x1] +
                    (1 - dx) * dy * img_src[y1, x0] +
                    dx * dy * img_src[y1, x1]
                    )

                    dst[y, x] = pixel

        return dst

    src_h,src_w,_=img_src.shape
    tar_h,tar_w,_=img_tar.shape

    result = warp(img_src, H, tar_h , tar_w)
    cv2.fillConvexPoly(img_tar, np.array([corner_tar], dtype=np.int32) ,(0,0,0))
    image = img_tar+result

    #draw 4 lines
    p1, p2, p3, p4 = corner_tar
    l1 = np.array([p1, p2], dtype=np.int32).reshape((-1, 1, 2))
    l2 = np.array([p2, p3], dtype=np.int32).reshape((-1, 1, 2))
    l3 = np.array([p3, p4], dtype=np.int32).reshape((-1, 1, 2))
    l4 = np.array([p4, p1], dtype=np.int32).reshape((-1, 1, 2))

    image = cv2.polylines(image, [l1, l2, l3, l4], isClosed=True, color=(0, 255, 0), thickness=2)

    #compute vanishing point
    line1 = np.cross([p1[0], p1[1],1], [p2[0], p2[1],1])
    line2 = np.cross([p4[0], p4[1],1], [p3[0], p3[1],1])
    vp = np.cross(line1,line2)
    vp = (vp/vp[2])[:2]

    vp = vp.astype(int)
    print(vp)

    cv2.circle(image, vp, 5, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(file_path,"homography.png"),image)
    cv2.imshow("image",image)
    cv2.waitKey(0)
