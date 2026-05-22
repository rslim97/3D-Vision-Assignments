# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:15:24 2026

@author: reina
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, atan2, sqrt
from copy import deepcopy


def rad2deg(x):
    return x / np.pi * 180


def homogenize(x):
    # x is of size (d, n)
    return np.vstack((x, np.ones((1, x.shape[1]))))


def dehomogenize(x):
    # x is of size (d+1, n)
    return x[:-1, :] / x[-1, :]


def pose_estimation(img, face2d, face3d, color):
    """
    Params:
    img : image
    face2d : known points in 2D,
    face3d : known corresponding points in 3D,
    color : color of the plotted points
    Given a known intrinsic matrix, image correspondences
    in 3D and 2D, compute the transformation from world to camera.
    The extrinsic and intrinsic matrices are then used to compute
    a projection matrix to project from 3D-2D. Reprojection error
    is also computed.
    """
    _, n = face2d.shape
    _, rotvec, transvec = cv2.solvePnP(face3d.T, face2d.T, K, None)
    R, _ = cv2.Rodrigues(rotvec)
    print(R)
    r11 = R[0, 0]
    r21 = R[1, 0]
    l = np.sqrt(r11**2 + r21**2)
    r32 = R[2, 1]
    r33 = R[2, 2]
    r31 = R[2, 0]
    r23 = R[1, 2]
    r22 = R[1, 1]
    if l >= 1e-6:
        pitch = atan2(r32, r33)
        yaw = atan2(-r31, l)
        roll = atan2(r21, r11)
    else:
        pitch = -atan2(r23, r22)
        yaw = atan2(-r31, l)
        roll = 0
    yaw = rad2deg(yaw)
    pitch = rad2deg(pitch)
    roll = rad2deg(roll)
    print(yaw, pitch, roll)
    # TODO 3: Compute the reprojected 2D landmarks, face2d_repr
    # inrinsic matrix
    K_i = K
    # extrinsic matrix
    K_o = np.zeros((4, 4))
    K_o[:3, :3] = R
    K_o[:3, 3] = transvec.squeeze()
    K_o[3, 3] = 1.0
    print(K_o)
    pi_0 = np.zeros((3, 4))  # Canonical projection matrix i.e. 3D-2D
    pi_0[:3, :3] = np.eye(3)
    face3d_hom = homogenize(face3d)  # (4, n)
    # Compute projection matrix P
    P = K_i @ pi_0 @ K_o  # (3, 3), (3, 4), (4, 4)
    face2d_repr = dehomogenize(P @ face3d_hom)  # (4, n)
    print(face2d_repr.shape)

    for i in range(face2d_repr.shape[1]):
        cv2.circle(img, (int(face2d_repr[0, i]), int(face2d_repr[1, i])), 1, color, 2)
    cv2.imshow("emma_watson", img)
    cv2.imwrite("emma_watson_" + str(n) + "_pts.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ### Calculate reprojection error ###
    # TODO 4: Print the reprojection error
    # repr_error = np.sum((face2d - face2d_repr)**2) / face2d.shape[1]
    repr_error = np.sum(np.mean((face2d - face2d_repr) ** 2, axis=1))
    print(face2d_repr.shape)
    print(repr_error)
    return P, repr_error


if __name__ == "__main__":
    """
    Pose estimation of world points where
    world_points = face_points.
    Given intrinsic camera matrix, 3D world points,
    and the corresponding 2D camera points.
    """

    img_name = "emma_watson"
    face2d = np.load(img_name + ".npy")
    face2d = face2d.astype("float32")
    face3d = np.load("basel_68_pts.npy")
    print(face2d.shape)  # (d, n)
    print(face3d.shape)  # (d, n)
    # (d, n) points format commonly used in 3D vision
    # Do not change the following two lines
    face3d[1, :] *= -1
    face3d[2, :] *= -1
    print("face3d.shape", face3d.shape)

    # TODO1: Display the 2D landmarks
    # enter your code here
    img = cv2.imread("emma_watson.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Only used when displaying using matplotlib imshow
    img1 = deepcopy(img)
    img2 = deepcopy(img)
    for i in range(face2d.shape[1]):
        cv2.circle(img1, (int(face2d[0, i]), int(face2d[1, i])), 1, (0, 255, 0), 2)
    cv2.imshow("emma_watson", img1)
    cv2.imwrite("emma_watson_keypoints.png", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    h, w = img.shape[:2]
    # Construct pseudo intrinsic matrix
    f = np.maximum(h, w)
    cx = w / 2
    cy = h / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    print(K)

    ### Pose Estimation using 68 points ###
    # TODO 2: Pose estimation
    # repr_error : 104.44326337181667
    P, repr_error = pose_estimation(img2, face2d, face3d, (255, 0, 0))

    ### Pose Estimation using 51 points ###
    # TODO 5: Repeat TODO 2 ~ TODO 4 using 51 landmarks
    # repr_error : 122.04128202553773
    _, _ = pose_estimation(img2, face2d[:, :51], face3d[:, :51], (0, 0, 255))

    # TODO 6: Augmented reality
    xmin, ymin, zmin = np.min(face3d, axis=1) - 1
    xmax, ymax, zmax = np.max(face3d, axis=1) + 1

    # Construc array of vertices
    vertices3d = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ]
    ).T  # (d, n)

    vertices3d_hom = homogenize(vertices3d)
    vertices2d_hom = P @ vertices3d_hom
    vertices2d = dehomogenize(vertices2d_hom)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(face3d[0, :], face3d[1, :], face3d[2, :])

    vertices_pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (1, 2),
        (0, 3),
        (5, 6),
        (4, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i in range(len(vertices_pairs)):
        v_i = vertices_pairs[i][0]
        v_j = vertices_pairs[i][1]
        plt.plot(
            [vertices3d[0, v_i], vertices3d[0, v_j]],
            [vertices3d[1, v_i], vertices3d[1, v_j]],
            [vertices3d[2, v_i], vertices3d[2, v_j]],
            "ro-",
        )
    ax.view_init(elev=-113, azim=-120)
    plt.savefig("face3d_and_vertices3d.png")
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    for i in range(len(vertices_pairs)):
        v_i = vertices_pairs[i][0]
        v_j = vertices_pairs[i][1]
        plt.plot(
            [vertices2d[0, v_i], vertices2d[0, v_j]],
            [vertices2d[1, v_i], vertices2d[1, v_j]],
            "ro-",
        )
    plt.savefig("augmented_reality.png")
    plt.show()
