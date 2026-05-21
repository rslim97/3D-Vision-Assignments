# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:49:10 2026

@author: reina
"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


def homogenize(x):
    # x is of size (d, n)
    # d - dimension
    # n - number of points
    # Converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x, np.ones((1, x.shape[1]))))


def dehomogenize(x):
    # x is of size (d+1, n)
    # d - dimension
    # n - number of points
    # Converts points from homogeneous to inhomogeneous coordinates
    return x[:-1, :] / x[-1, :]


def test_homography(H, pts_src):
    # pts_src: (d, n): (2, n)
    # H: (d+1, d+1): (3, 3)
    pts_src_hom = homogenize(pts_src)  # (d+1, n)
    pts_dst_hom = H @ pts_src_hom
    pts_dst = dehomogenize(pts_dst_hom)
    return pts_dst


# TODO 1 (15%): Implement calc_homography()
def calc_homography(pts_src, pts_dst):
    # pts_src: (d, n)
    # pts_dst: (d, n)
    n = int(pts_src.shape[1])
    A = np.zeros((2 * n, 9), dtype=np.float64)  # for 4 known points, 2n x 9, n=4

    for i in range(n):
        x_src = pts_src[0, i]
        y_src = pts_src[1, i]
        x_dst = pts_dst[0, i]
        y_dst = pts_dst[1, i]
        A[2 * i, 0:3] = np.array([-x_src, -y_src, -1])
        A[2 * i, 6:9] = np.array([x_src * x_dst, y_src * x_dst, x_dst])
        A[2 * i + 1, 3:6] = np.array([-x_src, -y_src, -1])
        A[2 * i + 1, 6:9] = np.array([x_src * y_dst, y_src * y_dst, y_dst])
    u, s, vh = np.linalg.svd(A)
    H = vh[-1, :].T
    H = H.reshape((3, 3))
    return H


def project_img(H, img_src, img_dst):
    ht_src, wid_src, _ = img_src.shape
    ht_dst, wid_dst, _ = img_dst.shape
    print("img_src.shape", img_src.shape)
    print("img_dst.shape", img_dst.shape)
    for j in range(ht_src):
        for i in range(wid_src):
            pt = np.array([[i, j, 1]])  # shape (1, 3)
            proj_pt = (H @ pt.T).T
            proj_pt_ = proj_pt[:, :2] / proj_pt[:, -1]
            x = int(proj_pt_[0, 0])
            y = int(proj_pt_[0, 1])
            if y > 0 and y < ht_dst and x > 0 and x < wid_dst:
                img_dst[y, x, :] = img_src[j, i, :]

    # return img_dst


def inverse_project_img(G, img_src, img_dst, pts_dst):
    ht_src, wid_src, _ = img_src.shape
    ht_dst, wid_dst, _ = img_dst.shape
    poly = cv2.fillPoly(img_dst, [pts_dst], (0, 0, 0))
    # print(poly.shape)
    # cv2.namedWindow('poly', cv2.WINDOW_NORMAL)
    # ht_window, wid_window = int(ht_dst*0.3), int(wid_dst*0.3)
    # cv2.resizeWindow('poly', ht_window, wid_window)
    # cv2.imshow('poly', poly)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for j in range(ht_dst):
        for i in range(wid_dst):
            if (poly[j, i, :] == (0, 0, 0)).all():
                pt = np.array([[i, j, 1]])  # shape (1, 3)
                proj_pt = (G @ pt.T).T
                proj_pt_dehom = proj_pt[:, :2] / proj_pt[:, -1]
                x = int(proj_pt_dehom[0, 0])
                y = int(proj_pt_dehom[0, 1])
                if y > 0 and y < ht_src and x > 0 and x < wid_src:
                    img_dst[j, i, :] = img_src[y, x, :]


if __name__ == "__main__":
    img_src = cv2.imread("monet.jpg")
    img_dst = cv2.imread("painting.jpg")
    ht_src, wid_src, _ = img_src.shape
    ht_dst, wid_dst, _ = img_dst.shape
    pts_src = np.array([[0, 0], [wid_src, 0], [wid_src, ht_src], [0, ht_src]])  # (n, d)
    pts_dst = np.array([[299, 515], [1776, 360], [1680, 1834], [239, 1615]])  # (n, d)
    print("source_shape:", ht_src, wid_src)
    print("destination_shape:", ht_dst, wid_dst)

    cv2.namedWindow("monet", cv2.WINDOW_NORMAL)
    ht_window = int(ht_src * 0.3)
    wid_window = int(wid_src * 0.3)
    cv2.resizeWindow("monet", wid_window, ht_window)
    cv2.imshow("monet", img_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow("painting", cv2.WINDOW_NORMAL)
    ht_window = int(ht_dst * 0.3)
    wid_window = int(wid_dst * 0.3)
    cv2.resizeWindow("painting", wid_window, ht_window)
    cv2.imshow("painting", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    H = calc_homography(pts_src.T, pts_dst.T)
    H = H / H[2, 2]
    print(H)
    print(test_homography(H, pts_src.T))

    # TODO 2 (10%): Project the source image on the frame
    #         formed by v1, v2, v3, and v4 in the target image via direct mapping.
    # Display the projection result.
    project_img(H, img_src, img_dst)

    cv2.namedWindow("painting", cv2.WINDOW_NORMAL)
    ht_window = int(ht_dst * 0.3)
    wid_window = int(wid_dst * 0.3)
    cv2.resizeWindow("painting", wid_window, ht_window)
    cv2.imshow("painting", img_dst)
    cv2.imwrite("direct_projection.png", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    G = calc_homography(pts_dst.T, pts_src.T)
    G = G / G[2, 2]

    # TODO 3 (20%): Project the source image on the frame
    #         formed by v1, v2, v3, and v4 in the target image via inverse mapping.
    # Display the projection result.
    inverse_project_img(G, img_src, img_dst, pts_dst)

    cv2.namedWindow("painting", cv2.WINDOW_NORMAL)
    ht_window = int(ht_dst * 0.3)
    wid_window = int(wid_dst * 0.3)
    cv2.resizeWindow("painting", wid_window, ht_window)
    cv2.imshow("painting", img_dst)
    cv2.imwrite("inverse_projection.png", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
