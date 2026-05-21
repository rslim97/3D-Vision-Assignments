# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 11:47:40 2026

@author: reina
"""

import cv2

color_red = (0, 0, 255)

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        cv2.circle(img, (x, y), 1, color_red, 2)
        cv2.imshow('image', img)
        
        
if __name__ == '__main__':
    img = cv2.imread('library1.jpg')
    img_copy = img.copy()
    print(img.shape)
    ht, wid, _ = img.shape
    is_running = True
    while is_running:
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)
        keyPressed = cv2.waitKey(0)
        if keyPressed == ord('q'):
            is_running = False
        if keyPressed == ord('c'):
            print('clear image')
            img = img_copy.copy()
        cv2.destroyAllWindows()