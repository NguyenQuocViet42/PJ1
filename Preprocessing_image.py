import cv2
import numpy as np

def preprocessing_image(img):
    # set arguments
    gamma = 0.2
    alpha = 0.1
    tau = 3.0

    # gamma correction
    img_gamma = np.power(img, gamma)
    img_gamma2 = (255.0 * img_gamma).clip(0,255).astype(np.uint8)

    # DOG
    blur1 = cv2.GaussianBlur(img_gamma, (0,0), 1, borderType=cv2.BORDER_REPLICATE)
    blur2 = cv2.GaussianBlur(img_gamma, (0,0), 2, borderType=cv2.BORDER_REPLICATE)
    img_dog = (blur1 - blur2)
    # normalize by the largest absolute value so range is -1 to 1
    img_dog = img_dog / np.amax(np.abs(img_dog))
    img_dog2 = (255.0 * (img_dog + 0.5)).clip(0,255).astype(np.uint8)

    # contrast equalization equation 1
    img_contrast1 = np.abs(img_dog)
    img_contrast1 = np.power(img_contrast1, alpha)
    img_contrast1 = np.mean(img_contrast1)
    img_contrast1 = np.power(img_contrast1,1.0/alpha)
    img_contrast1 = img_dog/img_contrast1

    # contrast equalization equation 2
    img_contrast2 = np.abs(img_contrast1)
    img_contrast2 = img_contrast2.clip(0,tau)
    img_contrast2 = np.mean(img_contrast2)
    img_contrast2 = np.power(img_contrast2,1.0/alpha)
    img_contrast2 = img_contrast1/img_contrast2
    img_contrast = tau * np.tanh((img_contrast2/tau))

    # Scale results two ways back to uint8 in the range 0 to 255
    #img_contrastA = (255.0 * (img_contrast+0.5)).clip(0,255).astype(np.uint8)
    img_contrastB = (255.0 * (0.5*img_contrast+0.5)).clip(0,255).astype(np.uint8)
    return img_contrastB