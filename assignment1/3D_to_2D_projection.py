# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:21:00 2026

@author: reina
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from math import cos, sin


def deg2rad(x):
    return x * np.pi / 180.0


def yaw(a):
    """
    a: yaw angle in degrees
    """
    x = deg2rad(a)
    c = np.cos(x)
    s = np.sin(x)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pitch(b):
    """
    b: pitch angle in degrees
    """
    x = deg2rad(b)
    c = np.cos(x)
    s = np.sin(x)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def roll(c):
    """
    c: roll angle in degrees
    """
    x = deg2rad(c)
    c = np.cos(x)
    s = np.sin(x)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def homogenize(x):
    # x is of size (d, n)
    return np.vstack((x, np.ones((1, x.shape[1]))))


def dehomogenize(x):
    # x is of size (d+1, n)
    return x[:-1, :] / x[-1, :]


if __name__ == "__main__":
    face3d = np.load("f3d_68_pts.npy")
    print(face3d.shape)  # (d, n)

    # TODO 1: Draw the 3D facial landmarks.
    # enter your code here

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    # ax = mplot3d.Axes3D(fig)
    x = face3d[0, :]
    y = face3d[1, :]
    z = face3d[2, :]
    # Create scatter plot
    ax.scatter3D(x, y, z, color="blue", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("3D_facial_landmarks.png")
    plt.show()

    # TODO 2: Computer the Euclidean distance between inner eyes.
    # enter your code here
    d = np.sqrt(np.sum((face3d[:, 39] - face3d[:, 42]) ** 2))
    print(d)

    K_i = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]])
    R = np.eye(3)  # (3, 3)
    tvec = np.array([0, 0, 30])[:, np.newaxis]  # (3, 1)
    K_o = np.hstack((R, tvec))  # (3, 4)
    # Camera Projection Matrix
    M = K_i @ K_o  # (3, 3) @ (3, 4)
    face3d_hom = homogenize(face3d)
    face2d_hom = M @ face3d_hom
    face2d = dehomogenize(face2d_hom)
    print(face2d.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(face2d[0, :], face2d[1, :])
    plt.savefig("projected_face_in_2D-1.png")
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(face2d[0, :], face2d[1, :])
    white_image = np.full((480, 640, 3), 255)
    plt.imshow(white_image)
    plt.savefig("projected_face_in_2D-2.png")
    plt.show()

    alpha = 0
    beta = 20
    gamma = 0
    R1 = yaw(alpha) @ pitch(beta) @ roll(gamma)
    alpha = 0
    beta = 60
    gamma = 0
    R2 = yaw(alpha) @ pitch(beta) @ roll(gamma)

    K_o1 = np.hstack((R1, tvec))  # Extrinsic matrix
    M1 = K_i @ K_o1  # Camera projection matrix
    face2d_hom = M1 @ face3d_hom  # Project points
    face2d = dehomogenize(face2d_hom)  # Dehomogenize projected points

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(face2d[0, :], face2d[1, :])
    white_image = np.full((480, 640, 3), 255)
    plt.imshow(white_image)
    plt.savefig("projected_face_in_2D-3.png")
    plt.show()

    K_o2 = np.hstack((R2, tvec))
    M2 = K_i @ K_o2
    face2d_hom = M2 @ face3d_hom
    face2d = dehomogenize(face2d_hom)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(face2d[0, :], face2d[1, :])
    white_image = np.full((480, 640, 3), 255)
    plt.imshow(white_image)
    plt.savefig("projected_face_in_2D-4.png")
    plt.show()

    xmin, ymin, zmin = np.min(face3d, axis=1) - 1
    xmax, ymax, zmax = np.max(face3d, axis=1) + 1

    # Construct array of vertices
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
    ).T
    vertices3d_hom = homogenize(vertices3d)
    vertices2d_hom = M1 @ vertices3d_hom
    vertices2d = dehomogenize(vertices2d_hom)
    face2d_hom = M1 @ face3d_hom  # Project points
    face2d = dehomogenize(face2d_hom)
    print("vertices2d.shape", vertices2d.shape)  # (2, 8)
    xlim_min, ylim_min = np.min(vertices2d, axis=1)
    xlim_max, ylim_max = np.max(vertices2d, axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(face2d[0, :], face2d[1, :])
    plt.plot(vertices2d[0, :2], vertices2d[1, :2], "ro-")
    plt.plot(vertices2d[0, 2:4], vertices2d[1, 2:4], "ro-")
    plt.plot(vertices2d[0, 4:6], vertices2d[1, 4:6], "ro-")
    plt.plot(vertices2d[0, 6:8], vertices2d[1, 6:8], "ro-")

    plt.plot(
        [vertices2d[0, 1], vertices2d[0, 2]],
        [vertices2d[1, 1], vertices2d[1, 2]],
        "ro-",
    )
    plt.plot(
        [vertices2d[0, 0], vertices2d[0, 3]],
        [vertices2d[1, 0], vertices2d[1, 3]],
        "ro-",
    )
    plt.plot(
        [vertices2d[0, 5], vertices2d[0, 6]],
        [vertices2d[1, 5], vertices2d[1, 6]],
        "ro-",
    )
    plt.plot(
        [vertices2d[0, 4], vertices2d[0, 7]],
        [vertices2d[1, 4], vertices2d[1, 7]],
        "ro-",
    )

    plt.plot(
        [vertices2d[0, 0], vertices2d[0, 4]],
        [vertices2d[1, 0], vertices2d[1, 4]],
        "ro-",
    )
    plt.plot(
        [vertices2d[0, 1], vertices2d[0, 5]],
        [vertices2d[1, 1], vertices2d[1, 5]],
        "ro-",
    )
    plt.plot(
        [vertices2d[0, 2], vertices2d[0, 6]],
        [vertices2d[1, 2], vertices2d[1, 6]],
        "ro-",
    )
    plt.plot(
        [vertices2d[0, 3], vertices2d[0, 7]],
        [vertices2d[1, 3], vertices2d[1, 7]],
        "ro-",
    )

    white_image = np.full((540, 640, 3), 255)
    plt.imshow(white_image)
    plt.savefig("projected_3D_Cube_in_2D_space.png")
    plt.show()

    print(vertices2d.shape)
