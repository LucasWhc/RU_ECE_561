# Primitive Panorama Stitching
# By Haocong Wang

import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt

top, bot, left, right = 100, 100, 0, 500
img1 = cv.imread('pic1.jpg')
# print(img1)
img2 = cv.imread('pic2.jpg')
img1_1 = cv.copyMakeBorder(img1, top, bot, left, right,
                           cv.BORDER_CONSTANT, value=(0, 0, 0))
img2_1 = cv.copyMakeBorder(img2, top, bot, left, right,
                           cv.BORDER_CONSTANT, value=(0, 0, 0))
img1_gray = cv.cvtColor(img1_1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2_1, cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d_SIFT().create()

# find key points and descriptors
k1, d1 = sift.detectAndCompute(img1_gray, None)
k2, d2 = sift.detectAndCompute(img2_gray, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(d1, d2, k=2)

# draw good matches
matches_mask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        good.append(m)
        pts2.append(k2[m.trainIdx].pt)
        pts1.append(k1[m.queryIdx].pt)
        matches_mask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matches_mask,
                   flags=0)
img3 = cv.drawMatchesKnn(img1_gray, k1, img2_gray, k2,
                         matches, None, **draw_params)
plt.imshow(img3, )
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('descriptors.jpg', dpi=600)
plt.show()

rows, cols = img1_1.shape[:2]
MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    # print(([k1[m.queryIdx].pt for m in good]).shape)
    img1_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    img2_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(img1_pts, img2_pts, cv.RANSAC, 5.0)
    warp_img = cv.warpPerspective(img2_1, np.array(
        M), (img2_1.shape[1], img2_1.shape[0]), flags=cv.WARP_INVERSE_MAP)

for col in range(0, cols):
    if img1_1[:, col].any() and warp_img[:, col].any():
        left = col
    break
for col in range(cols - 1, 0, -1):
    if img1_1[:, col].any() and warp_img[:, col].any():
        right = col
    break

# merge two pictures
res = np.zeros([rows, cols, 3], np.uint8)
for row in range(0, rows):
    for col in range(0, cols):
        if not img1_1[row, col].any():
            res[row, col] = warp_img[row, col]
        elif not warp_img[row, col].any():
            res[row, col] = img1_1[row, col]
        else:
            img1_len = float(abs(col - left))
            img2_len = float(abs(col - right))
            alpha = img1_len / (img1_len + img2_len)
            res[row, col] = np.clip(img1_1[row, col] * (1 - alpha) +
                                    warp_img[row, col] * alpha, 0, 255)

# convert color
res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
# show the result
plt.figure()
plt.imshow(res)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('result.jpg', dpi=600)
plt.show()
# else:
# print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
# matchesMask = None
