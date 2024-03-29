import os
import cv2
import json
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


# Загрузка модели
model = tf.keras.models.load_model("trained_model.h5")
# Чтение csv файла
test = pd.read_csv(r'C:\Users\user\Desktop\SamatmodulA\test.csv')

# Обработка изображений
def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # оттенки серого
    
    # Обрезка
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE) # поворот на 90 градусов

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 "
max_str_len = 9
num_of_characters = len(alphabets) + 1
num_of_timestamps = 64
batch_size = 128
# Функция, превращающая символ в цифру
def label_to_num(label):
    label_num = []
    for ch in label:
        
            label_num.append(alphabets.find(ch) if alphabets.find(ch)!=-1 else alphabets.find('-'))
        
    return np.array(label_num)
# Функция, превращающая цифру в символ
def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:
            break
        else:
            ret+=alphabets[ch]
    return ret
# обработка фотографии
def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_copy = image.copy() # Копия фотографии в оригинальном виде для вывода
    image = preprocess(image)
    image = image / 255.
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1],
                                       greedy=True)[0][0])
    result = num_to_label(decoded[0])
    return image_copy, result
# Выбор изображение
def select_image():
    file_path = filedialog.askopenfilename() # Конвертация изображения в путь
    if file_path:
        image, result = process_image(file_path)
        image = Image.fromarray(image)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        result_label.config(text=result)

        
# Создание окна
root = tk.Tk()
root.title("OCR Interface")
root.geometry("1000x500")

# Настройка цветов
root.configure(bg="#f0f0f0")
button_bg = "#4CAF50" 
button_fg = "white" 
# Кнопка выбора изображения
select_button = tk.Button(root, text="Выбрать изображение", command=select_image, bg=button_bg, fg=button_fg)
select_button.pack(pady=10)
# Формат вывода
result_label = tk.Label(root, text="", font=("Helvetica", 16), bg="#f0f0f0")
result_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)
# Запуск
root.mainloop()