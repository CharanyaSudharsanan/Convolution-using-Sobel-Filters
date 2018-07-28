# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:17:19 2018

@author: dues1
"""

import numpy as np
import cv2
import time
 
img = cv2.imread('lena_gray.jpg',0)
cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.imshow('Input Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

sobelx1 = np.array([1,2,1])[:,None]
sobelx2 = np.array([-1,0,1])[None,:]

sobely1 = np.array([-1,0,1])
sobely1 = sobely1[:,None] 

sobely2 = np.array([1,2,1])
sobely2 = sobely2[None,:]

sobelx_output = np.zeros((512,512))
sobely_output = np.zeros((512,512))
sobelxy_output = np.zeros_like(img)

row = img.shape[0]
col = img.shape[1]

interx = np.zeros((512,512))
intery = np.zeros((512,512))

st = time.clock()


for x in range(1,row-1):     
    for y in range(1,col-1):
        sobx = sobelx1[0][0]*img[x-1][y] + sobelx1[1][0]*img[x][y] + sobelx1[2][0]*img[x+1][y]
        interx[x-1,y-1] = sobx
        
        soby = sobely1[0][0]*img[x-1][y] + sobely1[1][0]*img[x][y] + sobely1[2][0]*img[x+1][y]
        intery[x-1,y-1] = soby


#print(interx)              
#print(intery) 

#reference : https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html       
def pad_with(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

interx = np.pad(interx, 1 , pad_with)
intery = np.pad(intery, 1 , pad_with)

#print(interx)
#print(intery)
        
for x in range(1,interx.shape[0]-1):     
    for y in range(1,interx.shape[1]-1):
        
        sobx = sobelx2[0][0]*interx[x][y-1] + sobelx2[0][1]*interx[x][y] + sobelx2[0][2]*interx[x][y+1]
        sobelx_output[x-1,y-1] = sobx
        
        soby = sobely2[0][0]*intery[x][y-1] + sobely2[0][1]*intery[x][y] + sobely2[0][2]*intery[x][y+1]
        sobely_output[x-1,y-1] = soby
        
        sobxy = np.sqrt(sobx * sobx + soby * soby)
        sobelxy_output[x-1,y-1] = sobxy
        

end = time.clock()
tt = end - st

print('Time taken for 1D Convolution :', tt)    
#print(sobelx_output)
#print(sobelx_output.shape)    
cv2.imshow('sobelx image - 1D', sobelx_output)
cv2.waitKey(0)
cv2.destroyAllWindows()            

#print(sobely_output)
#print(sobely_output.shape)    
cv2.imshow('sobely image - 1D', sobely_output)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#print(sobelxy_output)
#print(sobelxy_output.shape)    
cv2.imshow('sobelxy image - 1D', sobelxy_output)
cv2.waitKey(0)
cv2.destroyAllWindows() 

cv2.imwrite('1D_sobelGx.png',sobelx_output) 
cv2.imwrite('1D_sobelGy.png',sobely_output)
cv2.imwrite('1D_sobelG.png',sobelxy_output)