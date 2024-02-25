import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import aspose.pdf as ap


import filter
from tensorflow.keras.models import load_model


model = load_model('models//wildfire_detection_model4.h5')
images = os.listdir('real_data')

test_images = []

for image in images:
    test = filter.JPEG('real_data\\' + image) 
    try:
        test.decode()  
    except:
        print("INVALID_DATA!")
        os.remove('real_data\\' + image)
        images.remove(image)
        continue
    img = cv2.imread('real_data\\' + image)
    test_images.append(img)
    
i = 0
for image in test_images:
    test_images[i] = tf.image.resize(test_images[i],(256,256))
    i = i + 1

results = []

i = 0
for image in test_images:
    predicted_y = model.predict(np.expand_dims(test_images[i] / 255, 0))
    results.append(predicted_y)
    i = i + 1


fig, ax = plt.subplots(nrows = len(test_images), figsize = (20,60))
for i,image in enumerate(images):
    ax[i].imshow(cv2.cvtColor(cv2.imread('real_data\\' + image), cv2.COLOR_BGR2RGB))
    if results[i] >= 0.5:
       ax[i].set_title(f'wildfire :{results[i]}', fontsize = 25)
    else:
       ax[i].set_title(f'no wildfire :{results[i]}', fontsize = 25)
       

reading_number = len(os.listdir('readings')) + 1
file_path = f'readings\\reading{reading_number}.pdf'

plt.savefig(file_path)

document = ap.Document(file_path)

textStamp = ap.TextStamp("Predictions")
textStamp.width = 600
textStamp.height = 70
textStamp.top_margin = 160
textStamp.horizontal_alignment = ap.HorizontalAlignment.CENTER
textStamp.vertical_alignment = ap.VerticalAlignment.TOP

document.pages[1].add_stamp(textStamp)

document.save(file_path)

os.startfile(file_path)