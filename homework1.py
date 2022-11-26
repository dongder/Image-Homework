#homework1
#生成不同空间分辨率和灰度分辨率的lena图像
import cv2

#读取图像
img1=cv2.imread('lena.bmp',1)

#改变空间分辨率
img2 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
img3 = cv2.resize(img1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
cv2.waitKey(0)

# 改变灰度分辨率
def reduce_intensity_levels(img, level):
    img = cv2.copyTo(img, None)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            si = img[x, y]
            ni = int(level * si / 255 + 0.5) * (255 / level)
            img[x, y] = ni
    return img

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
result1 = reduce_intensity_levels(gray, 7)
result2 = reduce_intensity_levels(gray, 1)

cv2.imshow('gray',gray)
cv2.imshow('result1', result1)
cv2.imshow('result2', result2)
cv2.waitKey(0)


