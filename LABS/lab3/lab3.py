import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
img2 = cv2.imread('lena1.png', 0)

# low-freq image
low_freq = cv2.GaussianBlur(img2, (21, 21), 0)

# high-freq kernel
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
high_freq = cv2.filter2D(img2, -1, kernel)


#quanitization depending on divison
def quantize_image(img, div):
    pix = 256 //div
    quantized = (img//pix)*pix
    return quantized.astype(np.uint8)


# strech
def stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image.copy()
    stretched = (image.astype(np.float32)-min_val) /( max_val- min_val)
    stretched = (stretched *255).clip(0,255).astype(np.uint8)

    return stretched


quant_levels = [128, 64, 32]


fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle("Histogram Stretching on Quantized Images", fontsize=16)

for i, L in enumerate(quant_levels):
    low_q = quantize_image(low_freq, L)
    high_q = quantize_image(high_freq, L)
    axes[0, i].imshow(low_q, cmap='gray')
    axes[0, i].set_title(f"low-freq = {L}")
    axes[0, i].axis('off')
    axes[1, i].hist(low_q.ravel(), bins=256, range=[0, 255])
    axes[1, i].set_title(f"histogram for{L}")
    axes[2, i].imshow(high_q, cmap='gray')
    axes[2, i].set_title(f"high-freq {L}")
    axes[2, i].axis('off')
    axes[3, i].hist(high_q.ravel(), bins=256, range=[0, 255])
    axes[3, i].set_title(f"Histogram Q={L}")
plt.tight_layout()
plt.show()



fig2, axes2 = plt.subplots(4, 3, figsize=(15, 12))
fig2.suptitle("Before & After Histogram Stretching", fontsize=16)
for i, L in enumerate(quant_levels):
    low_q =quantize_image(low_freq, L)
    high_q =quantize_image(high_freq, L)
    low_s = stretching(low_q)
    high_s = stretching(high_q)
    axes2[0, i].imshow(low_q, cmap='gray')
    axes2[0, i].set_title(f"low-freq {L}")
    axes2[0, i].axis('off')
    axes2[1, i].imshow(low_s, cmap='gray')
    axes2[1, i].set_title(f"low-freq with stretch {L}")
    axes2[1, i].axis('off')
    axes2[2, i].imshow(high_q, cmap='gray')
    axes2[2, i].set_title(f"high_freq {L}")
    axes2[2, i].axis('off')
    axes2[3, i].imshow(high_s, cmap='gray')
    axes2[3, i].set_title(f"high-f stretch{L}")
    axes2[3, i].axis('off')
plt.tight_layout()
plt.show()

cv2.tr
cv2.imshow("Original", img2)

# final stats
print("Original Low-Freq: Min:", np.min(low_freq), "Max:", np.max(low_freq))
print("Original High-Freq: Min:", np.min(high_freq), "Max:", np.max(high_freq))

for L in quant_levels:
    low_q = quantize_image(low_freq, L)
    low_s = stretching(low_q)

    print(f"\nQuantization level {L}:")
    print("  Low-Freq Quantized Min/Max:", np.min(low_q), "/", np.max(low_q))
    print("  Low-Freq Stretched Min/Max:", np.min(low_s), "/", np.max(low_s))
'''
'''
img = cv2.imread("lena1.png", 0)
ret,thresh1 = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,60,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,60,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,60,255,cv2.THRESH_TOZERO_INV)
negative = 255 - img

hist1 = cv2.calcHist([thresh1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([thresh3], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([negative], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([img], [0], None, [256], [0, 256])


fig, axes = plt.subplots(4, 2, figsize=(15, 15))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Og")
axes[0, 0].axis('off')
axes[0, 1].hist(img.ravel(), bins=256, range=[0,256],color='blue',alpha=0.7)
axes[0, 1].set_title("histogram 1")
axes[1, 0].imshow(thresh1, cmap='gray')
axes[1, 0].set_title("binary")
axes[1, 0].axis('off')
axes[1, 1].hist(thresh1.ravel(), bins=256, range=[0,256],color='blue',alpha=0.7)
axes[1, 1].set_title("hist bin")
axes[2, 0].imshow(thresh3, cmap='gray')
axes[2, 0].set_title("trash norm")
axes[2, 0].axis('off')
axes[2, 1].hist(thresh3.ravel(), bins=256, range=[0,256],color='blue',alpha=0.7)
axes[2, 1].set_title("Hist trash norm")
axes[3, 0].imshow(negative, cmap='gray')
axes[3, 0].set_title("neagative")
axes[3, 0].axis('off')
axes[3, 1].hist(negative.ravel(), bins=256, range=[0,256],color='blue',alpha=0.7)
axes[3, 1].set_title("hist negative")
plt.tight_layout()
plt.show()

'''

img2 = cv2.imread('lena1.png', 0)
rows, cols = img2.shape
m = cv2.getOptimalDFTSize(rows)
n = cv2.getOptimalDFTSize(cols)
padded = cv2.copyMakeBorder(img2, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
complexI = cv2.merge(planes)
cv2.dft(complexI, complexI)
cv2.split(complexI, planes)
cv2.magnitude(planes[0], planes[1], planes[0])
magI = planes[0]
matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
cv2.add(matOfOnes, magI, magI)
cv2.log(magI, magI)
magI_rows, magI_cols = magI.shape
magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
cx = int(magI_rows / 2)
cy = int(magI_cols / 2)
q0 = magI[0:cx, 0:cy]
q1 = magI[cx:cx + cx, 0:cy]  # Top-Right
q2 = magI[0:cx, cy:cy + cy]  # Bottom-Left
q3 = magI[cx:cx + cx, cy:cy + cy]  # Bottom-Right

tmp = np.copy(q0)
magI[0:cx, 0:cy] = q3
magI[cx:cx + cx, cy:cy + cy] = tmp

tmp = np.copy(q1)
magI[cx:cx + cx, 0:cy] = q2
magI[0:cx, cy:cy + cy] = tmp
cv2.normalize(magI, magI, 0, 1, cv2.NORM_MINMAX)
cv2.imshow("lena", img2)
cv2.imshow(" lena spectrum", magI)

#b
idft = cv2.idft(complexI)
cv2.split(idft, planes)
reconstructed = cv2.magnitude(planes[0], planes[1])
reconstructed = reconstructed[0:rows, 0:cols]
cv2.normalize(reconstructed, reconstructed, 0, 255, cv2.NORM_MINMAX)
reconstructed = np.uint8(reconstructed)
cv2.imshow("reconstructed", reconstructed)

diff = cv2.absdiff(img2,reconstructed)
plt.figure(figsize=(12, 8))
plt.imshow(diff, cmap='plasma')
plt.title('Difference Image')
plt.colorbar()
plt.tight_layout()
plt.show()


#4
img1 = cv2.imread('kkk.jpg', 0)
dft = cv2.dft(np.float32(img1), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = 20 * np.log(magnitude + 1)
mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
mag_norm = np.uint8(mag_norm)
threshold = np.percentile(magnitude_spectrum, 99)
noise_elements = magnitude_spectrum > threshold
noise_plot = (noise_elements.astype(np.uint8) * 255)

img1_r = cv2.resize(img1, (500, 500))
mag_norm_r = cv2.resize(mag_norm, (500, 500))
noise_plot_r= cv2.resize(noise_plot, (500, 500))
cv2.imshow('fav klub',img1_r)
cv2.imshow("kkk spectrum", mag_norm_r)
cv2.imshow("noise", noise_plot_r)
cv2.waitKey()