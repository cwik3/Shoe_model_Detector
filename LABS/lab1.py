import numpy as np
import cv2


img = cv2.imread('../img/lena1.png', 1)
img2 = cv2.imread('../img/lena1.png',0)
cv2.imshow('img',img)
cv2.imshow('image2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# task 2
"""
x1 = np.uint8([200])
y1 = np.uint8([25])
x2 = np.uint8([120])
y2 = np.uint8([2])

#add1 = cv2.add(img,x1)
#cv2.imshow('addition',add1)
#add2 = cv2.add(img,y1)
#cv2.imshow('addition2',add2)
#subb1 = cv2.subtract(img,x2)
#cv2.imshow('subb',subb1)
multi1 = cv2.multiply(img,y2)
cv2.imshow('multi',multi1)
#div1 = cv2.divide(img,y2)
#cv2.imshow('div',div1)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# task 3
'''
img = cv2.imread('../img/lena1.png', 1)
img2 = cv2.imread('../img/nike_purple_court.jpg',1)
print(img.shape)
print(img2.shape)

resized2 = cv2.resize(img2,(512,512))
print(resized2.shape)

blend = cv2.addWeighted(img,0.5,resized2,0.5,0)
cv2.imshow('img_blended',blend)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#task 4
'''
img = cv2.imread('../img/lena1.png', 1)
img2 = cv2.imread('../img/nike_purple_court.jpg',1)

resized2 = cv2.resize(img2,(512,512))

vertical1 = cv2.vconcat([img,resized2])
cv2.imshow('vconcat',vertical1)

horizontal1 = cv2.hconcat([img,resized2])
cv2.imshow('hconcat',horizontal1)
'''


'''
def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(img.shape[1] for img in img_list)

    im_list_resize = [cv2.resize(img, (w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation=interpolation)
                      for img in img_list]

    return cv2.vconcat(im_list_resize)

img_v_resize = vconcat_resize([img2, img, resized2])
cv2.imshow('Vertical Concatenated and Resized', img_v_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()


#print(cv2.__version__)
'''