# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:42:47 2026

@author: reina
"""

import cv2
import numpy as np


def calc_homography(pts_src, pts_dst):
    A = np.zeros((8, 9), dtype=np.float64)  # for 4 known point
    # correspondences (2D-2D)

    for i in range(4):
        x_src = pts_src[i, 0]
        y_src = pts_src[i, 1]
        x_dst = pts_dst[i, 0]
        y_dst = pts_dst[i, 1]
        A[2 * i, 0:3] = np.array([-x_src, -y_src, -1])
        A[2 * i, 6:9] = np.array([x_src * x_dst, y_src * x_dst, x_dst])
        A[2 * i + 1, 3:6] = np.array([-x_src, -y_src, -1])
        A[2 * i + 1, 6:9] = np.array([x_src * y_dst, y_src * y_dst, y_dst])
    u, s, vh = np.linalg.svd(A)
    H = vh[-1, :].T
    H = H.reshape((3, 3))
    return H


if __name__ == "__main__":
    img_dst = cv2.imread("library1.jpg")
    img_src = cv2.imread("library2.jpg")
    img_dst_copy = img_dst.copy()
    img_src_copy = img_src.copy()
    ht_dst, wid_dst, _ = img_dst.shape
    ht_src, wid_src, _ = img_src.shape
    pts_dst = np.array([[300, 68], [296, 217], [454, 40], [449, 217]])
    pts_src = np.array([[69, 55], [64, 211], [226, 35], [219, 209]])
    # View key points
    for i in range(4):
        cv2.circle(img_dst_copy, (pts_dst[i, 0], pts_dst[i, 1]), 2, (0, 0, 255), 5)
        cv2.circle(img_src_copy, (pts_src[i, 0], pts_src[i, 1]), 2, (0, 0, 255), 5)

    cv2.imshow("library1", img_dst_copy)
    cv2.imwrite("library1_with_points.png", img_dst_copy)
    cv2.imshow("library2", img_src_copy)
    cv2.imwrite("library2_with_points.png", img_src_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Compute homography for direct projection
    # H = calc_homography(pts_src, pts_dst)
    # img_canvas = np.zeros((ht_dst, wid_dst * 2, 3), dtype=np.uint8)
    # h, w, c = img_canvas.shape
    # for j in range(ht_src):
    #     for i in range(wid_src):
    #         pt = np.array([[i, j, 1]])
    #         proj_pt = (H @ pt.T).T
    #         proj_pt_dehom = proj_pt[:, :2] / proj_pt[:, -1]
    #         x = int(proj_pt_dehom[0, 0]); y = int(proj_pt_dehom[0, 1])
    #         if y > 0 and y < h and x > 0 and x < w:
    #             img_canvas[y, x, :] = img_src[j, i, :]

    # Compute homography for inverse projection
    G = calc_homography(pts_dst, pts_src)
    img_canvas = np.zeros((ht_dst, wid_dst * 2, 3), np.uint8)
    h, w, c = img_canvas.shape
    for j in range(ht_dst):
        for i in range(wid_dst):
            pt = np.array([[i + wid_dst, j, 1]])
            proj_pt = (G @ pt.T).T
            proj_pt_dehom = proj_pt[:, :2] / proj_pt[:, -1]
            x = int(proj_pt_dehom[0, 0])
            y = int(proj_pt_dehom[0, 1])
            if y > 0 and y < ht_src and x > 0 and x < wid_src:
                img_canvas[j, i + wid_dst, :] = img_src[y, x, :]

    # print(np.max(img_warped))
    cv2.imshow("img_warped", img_canvas)
    cv2.imwrite("img_warped.png", img_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_canvas[:, :wid_src, :] = img_dst
    cv2.imshow("img_stitched", img_canvas)
    cv2.imwrite("img_stitched.png", img_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
