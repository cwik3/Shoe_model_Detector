import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
methods = [cv2.INTER_LINEAR,cv2.INTER_NEAREST,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4,cv2.INTER_AREA]
metName = ['INTER_LINEAR','INTER_NEAREST','INTER_CUBIC','INTER_LANCZOS4','INTER_AREA']
img2 = cv2.imread('lena1.png',0)
cv2.imshow('image2',img2)
print(img2.shape)


resize2 = cv2.resize(img2,None,fx = 1.3 , fy = 1.3,interpolation=cv2.INTER_LINEAR)  # img , width x hight x chanels fx,fy scaling factors
cv2.imshow('resize2',resize2) # resampling , type of frequancy of the image
print(resize2.shape)


for name,methods in zip(metName, methods):
    resize3 = cv2.resize(img2, None, fx=0.6, fy=0.6, interpolation=methods) # img , width x hight x chanels fx,fy scaling factors
    restored = cv2.resize(resize3, (img2.shape[1], img2.shape[0]), interpolation=methods)
    cv2.imshow(f"Resize - {name}", resize3)
    cv2.imshow(f"Restored - {name}", restored)
    print(f"{name} -> Restored shape: {restored.shape}")
    
cv2.waitKey(0)
cv2.destroyAllWindows()    
'''


img2 = cv2.imread('lena1.png',0)

imgOver = np.clip(img2 * 0.4 + 100 , 0, 255).astype(np.uint8) #only int supported for floats crushes
imgUnder = np.clip(img2 * 0.4 - 100 , 0, 255).astype(np.uint8)
cv2.imshow('overexposed', imgOver)
cv2.imshow('underexposed', imgUnder)

eqOver = cv2.equalizeHist(imgOver)
eqUnder = cv2.equalizeHist(imgUnder)

#Use milder Gaussian blur for low frequency images
# Smaller kernel size preserves more frequency content while still creating low frequency version
low_freq =  cv2.GaussianBlur(img2, (21, 21), 0)
low_freq_over = cv2.GaussianBlur(imgOver, (21, 21), 2.0)  # Reduced from (21,21) to (9,9)
low_freq_under = cv2.GaussianBlur(imgUnder, (21, 21), 2.0)# removes noise from an img , kernel size pos+ and odd sigmax = sigmay/standrad deviation

cv2.imshow('low_freq', low_freq)

eqlow_freq_over = cv2.equalizeHist(low_freq_over)
eqlow_freq_under = cv2.equalizeHist(low_freq_under)

hist = cv2.calcHist([low_freq_over], [0], None, [255], [0, 255])
plt.hist(low_freq_over.ravel(), 255, [0, 255])# x = pixel intensity y = amount of pixels
hist1 = cv2.calcHist([low_freq_under], [0], None, [255], [0, 255])
plt.hist(low_freq_under.ravel(), 255, [0, 255])
plt.show()


#High freq kernel
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
high_freq_over = cv2.filter2D(imgOver, -1, kernel)
high_freq_under = cv2.filter2D(imgUnder, -1, kernel)

eq_high_freq_over = cv2.equalizeHist(high_freq_over)
eq_high_freq_under = cv2.equalizeHist(high_freq_under)


cv2.imshow('image2',img2)
cv2.imshow('high',img2)
cv2.imshow('low_freq over',low_freq_over)
cv2.imshow('low_freq under',low_freq_under)
cv2.imshow('high_freq over',high_freq_over)
cv2.imshow('high_freq under',high_freq_under)



hist2 = cv2.calcHist([high_freq_over], [0], None, [255], [0, 255])
plt.hist(high_freq_over.ravel(), 255, [0, 255])
hist3 = cv2.calcHist([high_freq_under], [0], None, [255], [0, 255])
plt.hist(high_freq_under.ravel(), 255, [0, 255])
plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()

