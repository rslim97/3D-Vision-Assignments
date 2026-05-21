# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:28:09 2026

@author: reina
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def data_normalization(pts):
    """
    Data normalization of n dimensional pts

    Input:
        pts - in inhomogeneous coordinates
    Output:
        pts - data normalized points
        T - corresponing transformation matrix
    """
    # pts: (d, n)
    pts_hom = homogenize(pts)
    pts_mean = np.mean(pts_hom, axis=1)
    pts_var = np.var(pts_hom, axis=1)
    # d==2
    T = np.zeros((3, 3))
    var_x = pts_var[0]
    var_y = pts_var[1]
    mean_x = pts_mean[0]
    mean_y = pts_mean[1]
    s = np.sqrt(2 / (var_x + var_y))
    T[0, 0] = s
    T[0, 2] = -s * mean_x
    T[1, 1] = s
    T[1, 2] = -s * mean_y
    T[2, 2] = 1
    pts = T @ pts_hom
    return pts, T


def compute_fundamental_matrix(pts1_normal, pts2_normal, T1, T2):
    # pts in inhomogeneous coordinates
    # pts1_normal: (d, n), d = 2
    # pts2_normal: (d, n), d = 2
    n = pts1_normal.shape[1]
    A = np.zeros((n, 9), dtype=np.float64)
    for i in range(n):
        u, v = pts1_normal[:, i]
        u_, v_ = pts2_normal[:, i]
        A[i, :] = np.array([u_ * u, u_ * v, u_, v_ * u, v_ * v, v_, u, v, 1])
    u, s, vh = np.linalg.svd(A)
    F = vh[-1, :].T.reshape(3, 3)
    print(F.shape)
    u_, s_, vh_ = np.linalg.svd(F)
    s_[2] = 0.0
    F_normal = u_ @ np.diag(s_) @ vh_
    # Un-normalize F
    F_unnormal = T2.T @ F_normal @ T1
    return F_unnormal


def compute_epipolar_error(F, pts1, pts2):
    # pts1: (d, n), n=2
    # pts2: (d, n), n=2
    pts1_hom = homogenize(pts1)  # (d+1, n)
    pts2_hom = homogenize(pts2)  # (d+1, n)
    error = np.sum(pts2_hom.T @ F @ pts1_hom)
    return error


if __name__ == "__main__":
    img1 = plt.imread("box_left_2k.jpg")
    img2 = plt.imread("box_right_2k.jpg")
    print(img1.shape)

    pts1 = np.array(
        [
            [1148, 351],
            [1441, 481],
            [1427, 662],
            [1172, 525],
            [1524, 80],
            [1795, 134],
            [1744, 307],
            [1564, 320],
            [1579, 306],
            [1584, 288],
        ]
    )  # x: (n, d), left image points

    pts2 = np.array(
        [
            [747, 397],
            [944, 522],
            [1010, 704],
            [822, 558],
            [1212, 116],
            [1440, 170],
            [1433, 350],
            [1131, 360],
            [1152, 342],
            [1166, 326],
        ]
    )  # x': (n, d), right image points

    K1 = np.array([[1062.04, 0, 1098.52], [0, 1061.74, 599.665], [0, 0, 1]])
    K2 = np.array([[1060.59, 0, 1127.03], [0, 1060.06, 645.711], [0, 0, 1]])

    # TODO 1 (15 points): Normalize image points.
    #        There is no need to use for loops in this task. If you use for loops in this task, you can get at most 10 point.

    pts1_normal_hom, T1 = data_normalization(pts1.T)
    pts2_normal_hom, T2 = data_normalization(pts2.T)
    print(np.mean(np.linalg.norm(dehomogenize(pts1_normal_hom), axis=0)))

    # TODO 2 (25 points): Compute the fundamental matrix F.

    pts1_normal = dehomogenize(pts1_normal_hom)
    pts2_normal = dehomogenize(pts2_normal_hom)
    F = compute_fundamental_matrix(pts1_normal, pts2_normal, T1, T2)
    print("Fundamental matrix, F: ", F)
    error = compute_epipolar_error(F, pts1.T, pts2.T)
    print("epipolar error: ", error)

    # TODO 3 (15 points): Draw epipolar lines in image1.
    pts_left_hom = homogenize(pts1.T)
    pts_right_hom = homogenize(pts2.T)
    print(pts_left_hom.shape)  # (3, 10)
    print(pts_right_hom.shape)  # (3, 10)
    print(F.shape)  # (3, 3)

    l_left = F.T @ pts_right_hom
    l_right = F @ pts_left_hom

    print("l_left.shape", l_left.shape)  # (3, 10)
    print("l_right.shape", l_right.shape)  # (3, 10)
    marker = ["ro--", "bo--", "go--"]
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    # Plot first three epipolar lines in left image
    for i in range(3):
        lx, ly, lz = l_left[:, i]
        for j in range(img1.shape[1]):
            # from ax + by + c = lx * x + ly * y + lz * 1 = 0
            x = int(j)
            y = int(-(lx / ly) * x - (lz / ly))
            # print('x, y', x, y)
            ax1.plot(x, y, marker[i], linewidth=1, markersize=1)

    ax1.imshow(img1)

    ax2 = fig.add_subplot(122)
    ax2.plot(pts2[0, 0], pts2[0, 1], "ro")
    ax2.plot(pts2[1, 0], pts2[1, 1], "bo")
    ax2.plot(pts2[2, 0], pts2[2, 1], "go")
    ax2.imshow(img2)
    plt.tight_layout()
    plt.savefig("plot_epipolar_lines.png")
    plt.show()

    # TODO 4 (10 points): Draw the epipole and epipolar lines in image1.
    # Compute epipole (point)
    u, s, vh = np.linalg.svd(F)
    e_hom = vh[-1, :].T
    e = (e_hom[:2] / e_hom[-1]).astype(int)
    fig = plt.figure(figsize=(10, 5))
    ax3 = fig.add_subplot(111)
    ax3.plot(e[0], e[1], "kx")
    for i in range(3):
        lx, ly, lz = l_left[:, i]
        for j in range(e[0]):
            # from ax + by + c = lx * x + ly * y + lz * 1 = 0
            x = int(j)
            y = int(-(lx / ly) * x - (lz / ly))
            # print('x, y', x, y)
            ax3.plot(x, y, marker[i], linewidth=1, markersize=1)
    img1_copy = np.zeros((img1.shape[0], e[0] + 50, 3), dtype=np.uint8)
    img1_copy.fill(255)
    img1_copy[: img1.shape[0], : img1.shape[1], :] = (
        img1  # img[y, x], indexing in image
    )
    ax3.imshow(img1_copy)
    plt.tight_layout()
    plt.savefig("plot_epipole.png")
    plt.show()
    # TODO 5 (5 points): Compute the 4 possible camera matrices.
    # Compute essential matrix
    E = K2.T @ F @ K1
    # Perform SVD on the essential matrix
    U, S, Vh = np.linalg.svd(E)  # Vh is V transpose
    # Let u3 be the last column of U
    u3 = U[:, -1]
    # Define W=R_{90}, rotates u1 and u2 by 90^o
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    if np.linalg.det(U @ W @ Vh) < 0:
        W = -W
    # Suppose the first camera matrix is P=[I, 0].
    # Then there are four possible choices for the second camera matrix P'=[R, t]
    R1 = U @ W @ Vh
    R2 = U @ W.T @ Vh
    t1 = u3
    t2 = -u3
    print(R1.shape)
    print(t1.shape)
    Rt_list = [
        np.hstack((R1, t1[..., np.newaxis])),
        np.hstack((R1, t2[..., np.newaxis])),
        np.hstack((R2, t1[..., np.newaxis])),
        np.hstack((R2, t2[..., np.newaxis])),
    ]
    # TODO 6 (25 points): Triangulation.
    #        There is no need to use for loops when computing the reconstruction errors of pts1 (or pts2).
    #        If you do, you can get at most 20 point.
    X_list = []
    P1 = np.zeros((3, 4), dtype=np.float64)
    P1[:, :3] = K1
    for j in range(len(Rt_list)):
        P2 = K2 @ Rt_list[j]
        X = np.zeros((pts1.shape[0], 3), dtype=np.float64)
        for i in range(pts1.shape[0]):
            A = np.zeros((4, 3), dtype=np.float64)
            b = np.zeros((4, 1), dtype=np.float64)
            xi, yi = pts1[i, 0], pts1[i, 1]
            xi_, yi_ = pts2[i, 0], pts2[i, 1]
            p11 = P1[0, 0]
            p11_ = P2[0, 0]
            p12 = P1[0, 1]
            p12_ = P2[0, 1]
            p13 = P1[0, 2]
            p13_ = P2[0, 2]
            p14 = P1[0, 3]
            p14_ = P2[0, 3]
            p21 = P1[1, 0]
            p21_ = P2[1, 0]
            p22 = P1[1, 1]
            p22_ = P2[1, 1]
            p23 = P1[1, 2]
            p23_ = P2[1, 2]
            p24 = P1[1, 3]
            p24_ = P2[1, 3]
            p31 = P1[2, 0]
            p31_ = P2[2, 0]
            p32 = P1[2, 1]
            p32_ = P2[2, 1]
            p33 = P1[2, 2]
            p33_ = P2[2, 2]
            p34 = P1[2, 3]
            p34_ = P2[2, 3]
            # Construct A matrix
            A11 = p11 - p31 * xi
            A12 = p12 - p32 * xi
            A13 = p13 - p33 * xi
            A21 = p21 - p31 * yi
            A22 = p22 - p32 * yi
            A23 = p23 - p33 * yi
            A31 = p11_ - p31_ * xi_
            A32 = p12_ - p32_ * xi_
            A33 = p13_ - p33_ * xi_
            A41 = p21_ - p31_ * yi_
            A42 = p22_ - p32_ * yi_
            A43 = p23_ - p33_ * yi_
            A[0, 0] = A11
            A[0, 1] = A12
            A[0, 2] = A13
            A[1, 0] = A21
            A[1, 1] = A22
            A[1, 2] = A23
            A[2, 0] = A31
            A[2, 1] = A32
            A[2, 2] = A33
            A[3, 0] = A41
            A[3, 1] = A42
            A[3, 2] = A43
            b[0, 0] = p34 * xi - p14
            b[1, 0] = p34 * yi - p24
            b[2, 0] = p34_ * xi_ - p14_
            b[3, 0] = p34_ * yi_ - p24_
            # Solve for [Xi, Yi, Zi].T using Ax=b
            m = np.linalg.lstsq(A, b)[0]
            X[i, :] = m.squeeze()

        X_list.append(X)

    j_best = [j for j in range(len(X_list)) if np.all(X_list[j][:, 2] > 0)]

    print("j_best", j_best)
    M = X_list[j_best[0]]  # (n, d) = (n, 3), estimated 3D points
    m1 = P1 @ homogenize(M.T)  # reprojected 2D points in image1
    P2 = K2 @ Rt_list[j_best[0]]

    m2 = P2 @ homogenize(M.T)  # reprojected 2D points in image2
    print(m1)  # (d+1, n)
    print(m2)  # (d+1, n)
    m1_dehom = dehomogenize(m1)
    m2_dehom = dehomogenize(m2)
    reproj_error1 = np.sum(np.sqrt(np.mean((m1_dehom - pts1.T) ** 2, axis=1)))
    reproj_error2 = np.sum(np.sqrt(np.mean((m2_dehom - pts2.T) ** 2, axis=1)))

    print(reproj_error1)
    print(reproj_error2)

    # TODO 7 (5 points): Visualize the 3D reconstruction.
    x = M[:, 0]
    y = M[:, 1]
    z = M[:, 2]

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.view_init(elev=-90, azim=-120)
    ax1.set_aspect("equal")
    ax1.scatter(x, y, z, s=1.7)
    # Plot points using plt.plot(x's, y's, z's)
    ax1.plot(M[:2, 0], M[:2, 1], M[:2, 2], "ro-")  # (v0,v1)
    ax1.plot(M[1:3, 0], M[1:3, 1], M[1:3, 2], "ro-")  # (v1,v2)
    ax1.plot(M[2:4, 0], M[2:4, 1], M[2:4, 2], "ro-")  # (v2,v3)
    ax1.plot(
        [M[3, 0], M[0, 0]], [M[3, 1], M[0, 1]], [M[3, 2], M[0, 2]], "ro-"
    )  # (v3,v0)
    ax1.plot(M[4:6, 0], M[4:6, 1], M[4:6, 2], "ro-")  # (v4, v5)
    ax1.plot(M[5:7, 0], M[5:7, 1], M[5:7, 2], "ro-")  # (v5, v6)
    ax1.plot(
        [M[0, 0], M[4, 0]], [M[0, 1], M[4, 1]], [M[0, 2], M[4, 2]], "ro-"
    )  # (v0, v4)
    ax1.plot(
        [M[1, 0], M[5, 0]], [M[1, 1], M[5, 1]], [M[1, 2], M[5, 2]], "ro-"
    )  # (v1, v5)
    ax1.plot(
        [M[2, 0], M[6, 0]], [M[2, 1], M[6, 1]], [M[2, 2], M[6, 2]], "ro-"
    )  # (v2, v6)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=-139, azim=-113)
    ax2.set_aspect("equal")
    ax2.scatter(x, y, z, s=1.7)
    # Plot points using plt.plot(x's, y's, z's)
    ax2.plot(M[:2, 0], M[:2, 1], M[:2, 2], "ro-")  # (v0,v1)
    ax2.plot(M[1:3, 0], M[1:3, 1], M[1:3, 2], "ro-")  # (v1,v2)
    ax2.plot(M[2:4, 0], M[2:4, 1], M[2:4, 2], "ro-")  # (v2,v3)
    ax2.plot(
        [M[3, 0], M[0, 0]], [M[3, 1], M[0, 1]], [M[3, 2], M[0, 2]], "ro-"
    )  # (v3,v0)
    ax2.plot(M[4:6, 0], M[4:6, 1], M[4:6, 2], "ro-")  # (v4, v5)
    ax2.plot(M[5:7, 0], M[5:7, 1], M[5:7, 2], "ro-")  # (v5, v6)
    ax2.plot(
        [M[0, 0], M[4, 0]], [M[0, 1], M[4, 1]], [M[0, 2], M[4, 2]], "ro-"
    )  # (v0, v4)
    ax2.plot(
        [M[1, 0], M[5, 0]], [M[1, 1], M[5, 1]], [M[1, 2], M[5, 2]], "ro-"
    )  # (v1, v5)
    ax2.plot(
        [M[2, 0], M[6, 0]], [M[2, 1], M[6, 1]], [M[2, 2], M[6, 2]], "ro-"
    )  # (v2, v6)
    plt.savefig("triangulated_3D_points.png")

    plt.show()
