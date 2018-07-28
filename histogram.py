# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 12:26:00 2018

@author: dues1
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

inputimg = cv2.imread('Bradley-Cooper_sm.jpg')

img = cv2.imread('Bradley-Cooper_sm.jpg',0)

#print(img)
#print(img.shape)
 
  
row = img.shape[0]
col = img.shape[1]

cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.imshow('Input Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


H = np.zeros(256)
Hc = np.zeros(256)
T = np.zeros(256)
Final = np.zeros(256)

for l in range(0,row):
    for m in range(0,col):
        H[img[l][m]] += 1
        

Hc[0] = H[0]

for p in range(1,256):
    Hc[p] = Hc[p-1] + H[p]
   

inter = (255.0/(row*col))

for p in range(0,256):
    T[p] = round(inter * Hc[p])


enhanced = np.zeros_like(img)
print('enhanced before')
print(enhanced)

for i in range(0,row):
    for j in range(0,col):
        enhanced[i][j] = T[img[i][j]]

for i in range(0,row):
    for j in range(0,col):
        Final[enhanced[i][j]] += 1

print('enhanced after :')
print(enhanced)  

plt.imshow(cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.savefig('Input img.jpg')
plt.show()

plt.imshow(img, cmap=plt.cm.gray)
plt.title('Grayscale Image')
plt.savefig('img.jpg')
plt.show()

plt.plot(H)
plt.title('Histogram of Grayscale Image')
plt.savefig('H.jpg')
plt.show()

plt.plot(Hc)
plt.title('Cumulative Histogram')
plt.savefig('Hc.jpg')
plt.show()

plt.plot(T)
plt.title('T Lookup table')
plt.savefig('T.jpg')
plt.show()


plt.plot(Final)
plt.title('Histogram Equalization')
plt.savefig('Final.jpg')
plt.show()

plt.imshow(enhanced, cmap=plt.cm.gray)
plt.title('Enhanced Image')
plt.savefig('Enhanced.jpg')
plt.show()

plt.subplot(1, 3, 1).imshow(cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB))
plt.title('Input Image')

plt.subplot(1, 3, 2).imshow(img, cmap=plt.cm.gray)
plt.title('Grayscale Image')

plt.subplot(1, 3, 3).imshow(enhanced, cmap=plt.cm.gray)
plt.title('Enhanced Image')
plt.savefig('Comparison.jpg')


plt.show()


        

