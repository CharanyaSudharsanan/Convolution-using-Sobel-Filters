# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:33:44 2018

@author: dues1
"""

import numpy as np
import cv2
import time

 
#import image 
img = cv2.imread('lena_gray.jpg',0)

#display input grayscale image
cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.imshow('Input Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#initializing kernels - Gx,Gy
sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype = np.float)
#print(sobelx)
sobely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype = np.float) 
#print(sobely)

#create placeholders to hold the output
sobelx_output = np.zeros((512,512))
sobely_output = np.zeros((512,512))
sobelxy_output = np.zeros_like(img) 

sobelx_output1 = np.zeros((512,512))
sobely_output1 = np.zeros((512,512))
sobelxy_output1 = np.zeros_like(img)
#zerpadding 
def pad_with(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

img = np.pad(img, 1 , pad_with)
print(img.shape)
print(img)


row = img.shape[0]
col = img.shape[1]

st = time.clock()

for x in range(1,row-1):     
    for y in range(1,col-1):
        sobx =(sobelx*img[x-1:x-1+3,y-1:y-1+3]).sum()
        sobelx_output[x-1,y-1] = sobx
        soby =(sobely*img[x-1:x-1+3,y-1:y-1+3]).sum()
        sobely_output[x-1,y-1] = soby
        sobxy = np.sqrt(sobx * sobx + soby * soby)
        sobelxy_output[x-1,y-1] = sobxy
        

end = time.clock()
tt = end - st
print('Time taken for 2D Convolution :', tt)
        
#print('sobelx:',sobelx_output)
#print(sobelx_output.shape)
cv2.imshow('sobelx image',sobelx_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print('sobely:',sobely_output)
#print(sobely_output.shape)
cv2.imshow('sobely image',sobely_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(sobelxy_output)
#print(sobelxy_output.shape)
cv2.imshow('sobelxy image', sobelxy_output)
cv2.waitKey(0)
cv2.destroyAllWindows()


