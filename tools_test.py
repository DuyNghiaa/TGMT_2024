import os
import numpy as np
from PIL import Image
import matplotlib.pillow as plt

def load_image (file_path):
    try:
        image = image.open(file_path)
        return image
    except Exception as e:
        print("Error! by: ", file_path, " ", e)
        return None

def is_image_file(file_path):
    extension =(".jpg", ".jpeg", ".png", ".gif", ".bmp")
    return file_path.lower().endswith(extension)

def get_image_list(folder_path):
    image_list = []
    if os.path.extist(folder_path) and os.path.isdir(folder_path):
        filenames = os.listdir(folder_path)
        for filename in filenames:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and is_image_file(file_path):
                image = load_image(file_path)
                image_list.append(image)
    return image_list

def histogram_equalization(image, nbs_bins=256):
    if image.mode != ('L'):
        image = image.convert('L')
    image_array = np.array(image)
    histogram, bins = np.histogram(image_array, bins = nbs_bins, range=(0,256), density= True)
    cdf = histogram.cumsum()
    cdf = cdf * 255 / cdf[-1]
    image_equalized = np.interp(cdf, image_array, bins[:-1])
    equalized_image = Image.formarray(image_equalized.astype('uint8')
    return equalized_image
    