# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:57:47 2018

@author: dues1
"""
import numpy as np
import cv2
import time

#Random matrices for 1D Convolution
sobelx1 = np.random.rand(101,1)
sobelx2 = np.random.rand(1,101)
sobely1 = np.random.rand(101,1)
sobely2 = np.random.rand(1,101)

#Random matrices for 2D Convolution
arrayx = np.outer(sobelx1,sobelx2) #2D Sobel X filter
arrayy = np.outer(sobely1,sobely2) #2D Sobel Y filter

#import image and convert it into an array
img = cv2.imread('lena_gray.jpg',0)
#print(img.shape)

#display input grayscale image
cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.imshow('Input Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

sobelx_output_1D = np.zeros_like(img) #Gx 
sobely_output_1D = np.zeros_like(img) #Gy
sobelxy_output_1D = np.zeros_like(img) #G

sobelx_output_2D = np.zeros_like(img) #Gx
sobely_output_2D = np.zeros_like(img) #Gy
sobelxy_output_2D = np.zeros_like(img) #G 
 
#reference : https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html
def pad_with(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

img = np.pad(img, 50 , pad_with)

#print(img.shape)
#print(img)

row = img.shape[0]
col = img.shape[1]


st = time.clock()

#2D Convolution
for x in range(51,row-51):     
    for y in range(51,col-51):        
        sobx = ((np.outer(sobelx2,sobelx1))*img[x-51:x-51+101,y-51:y-51+101]).sum()
        sobelx_output_2D[x-51,y-51] = sobx
        soby = ((np.outer(sobely2,sobely1))*img[x-51:x-51+101,y-51:y-51+101]).sum()
        sobely_output_2D[x-51,y-51] = soby
        sobxy = np.sqrt(sobx * sobx + soby * soby)
        sobelxy_output_2D[x-51,y-51] = sobxy
        
        
end = time.clock()
tt = end - st

print('Time taken for 2D Convolution :', tt)
        
#print('sobelx_2D:',sobelx_output_2D)
#print(sobelx_output_2D.shape) 
cv2.imshow('sobelx_2D image',sobelx_output_2D)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
#print('sobely_2D:',sobely_output_2D)
#print(sobely_output_2D.shape)
cv2.imshow('sobely_2D image',sobely_output_2D)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print('sobelxy_2D',sobelxy_output_2D)
#print(sobelxy_output_2D.shape)
cv2.imshow('sobelxy_2D image', sobelxy_output_2D)
cv2.waitKey(0)
cv2.destroyAllWindows()




#1D Convolution
 


st1 = time.clock()

interx = np.zeros((512,512))
intery = np.zeros((512,512))

#since we've padded 50 zeros all edges
for x in range(51,row-51):     
    for y in range(51,col-51):    
        sobx = np.sum(np.multiply(sobelx1[0:101,0],img[x-51:x-51+101,y])) 
        interx[x-51,y-51] = sobx        
        soby = np.sum(np.multiply(sobely1[0:101,0],img[x-51:x-51+101,y])) 
        intery[x-51,y-51] = soby


interx = np.pad(interx, 50 , pad_with)
intery = np.pad(intery, 50 , pad_with)


for x in range(51,interx.shape[0]-51):     
    for y in range(51,interx.shape[1]-51):
        sobx = np.sum(np.multiply(sobelx2[0,0:101],interx[x,y-51:y-51+101])) 
        sobelx_output_1D[x-51,y-51] = sobx        
        soby = np.sum(np.multiply(sobely2[0,0:101],intery[x,y-51:y-51+101])) 
        sobely_output_1D[x-51,y-51] = soby
        sobxy = np.sqrt(sobx * sobx + soby * soby)
        sobelxy_output_1D[x-51,y-51] = sobxy


end1 = time.clock()
tt1 = end1 - st1

print('Time taken for 1D Convolution :', tt1)
        
#print('sobelx_1D:',sobelx_output_1D)
#print(sobelx_output_1D.shape)
cv2.imshow('sobelx image',sobelx_output_1D)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print('sobely_1D:',sobely_output_1D)
#print(sobely_output_1D.shape)
cv2.imshow('sobely image',sobely_output_1D)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
#print('sobelxy_1D:',sobelxy_output_1D)
#print(sobelxy_output_1D.shape)
cv2.imshow('sobelxy image',sobelxy_output_1D)
cv2.waitKey(0)
cv2.destroyAllWindows()

        