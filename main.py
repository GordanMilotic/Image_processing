import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog

def prompt_and_apply_image_processing(image):
    options = {
        "1": "Denoising",
        "2": "Grayscale",
        "3": "Edge Detection",
        "4": "Gaussian Blur"
    }

    choice = simpledialog.askstring(
        "Odabir tehnike",
        "Odaberite tehniku korekcije grešaka:\n1. Denoising\n2. Grayscale\n3. Edge Detection\n4. Gaussian Blur"
    )

    if choice == "1":
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            # denosing
            denoised_image = cv2.fastNlMeansDenoisingColored(bgr_image, None, 10, 10, 7, 21)
            return cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB), "Denoised"
        except Exception as e:
            messagebox.showerror("Greška", f"Došlo je do greške prilikom primjene denoisinga: {e}")
            return None, None
        return cv2.fastNlMeansDenoisingColored(image, None, 50, 17, 71), "Denoised"
    elif choice == "2":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), "Grayscale"
    elif choice == "3":
        return cv2.Canny(image, 100, 200), "Edge Detection"
    elif choice == "4":
        return cv2.GaussianBlur(image, (15, 15), 0), "Gaussian Blur"
    else:
        messagebox.showerror("Greška", "Neispravan odabir.")
        return None, None

plt.style.use('seaborn-v0_8-whitegrid')

def select_and_process_image():
    file_path = filedialog.askopenfilename(
        title="Odaberi sliku",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    if not file_path:
        return  

    # BGR format koji se koverta u RGB za matplotlib
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Greška", "Slika nije pronađena ili je format neispravan.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    processed_image, technique = prompt_and_apply_image_processing(image)
    if processed_image is None:
        return

    if len(processed_image.shape) == 2:
        effect_only = processed_image
    else: 
        effect_only = cv2.absdiff(image, processed_image)

    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(image), plt.title('Original')
    plt.subplot(132), plt.imshow(effect_only, cmap='gray' if len(effect_only.shape) == 2 else None), plt.title('Effect only')
    plt.subplot(133), plt.imshow(processed_image, cmap='gray' if len(processed_image.shape) == 2 else None), plt.title(technique)
    
    plt.show()

root = tk.Tk()
root.withdraw() 

select_and_process_image()
