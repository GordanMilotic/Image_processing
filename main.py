
import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.style.use('seaborn-v0_8-whitegrid')

image = cv2.imread('C:/Users/user/Pictures/sik1_slike/baloons_noisy.png') #ubaciti svoju putanju za sliku
if image is None:
    raise FileNotFoundError("Slika nije pronađena. Provjeri putanju ili naziv datoteke.")

denoise = cv2.fastNlMeansDenoisingColored(image, None, 50, 17, 71)

plt.subplot(121), plt.imshow(image) 
plt.subplot(122), plt.imshow(denoise) 
  
plt.show()

