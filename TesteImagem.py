from __future__ import division, print_function
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


app = Flask(__name__)

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

def grayscale(img):
    if len(img.shape) == 3 and img.shape[2] == 3:  # Verifica se a imagem é colorida
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    if len(img.shape) == 3 and img.shape[2] == 3:  # Verifica se a imagem é colorida
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img.shape) == 2:  # Verifica se a imagem está em escala de cinza
        img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    if len(img.shape) == 3 and img.shape[2] == 3:  # Verifica se a imagem é colorida
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = equalize(img)
    img = img/255
    return img



def getClassName(classNo):
    class_names = [
        'Limite de velocidade 20 km/h', 'Limite de velocidade 30 km/h',
        'Limite de velocidade 50 km/h', 'Limite de velocidade 60 km/h',
        'Limite de velocidade 70 km/h', 'Limite de velocidade 80 km/h',
        'Fim de limite de velocidade (80km/h)', 'Limite de velocidade 100 km/h',
        'Limite de velocidade 120 km/h', 'Ultrapassagem proibida',
        'Ultrapassagem proibida para veículos acima de 3.5 toneladas',
        'Cedência de passagem no próximo cruzamento', 'Estrada com prioridade',
        'Dê a preferência', 'Pare', 'Veiculos proibidos',
        'Veículos acima de 3.5 toneladas proibidos', 'Entrada proibida',
        'Atenção', 'Curva perigosa à esquerda', 'Curva perigosa à direita',
        'Curva dupla', 'Estrada irregular', 'Piso escorregadio',
        'strada estreita à direita', 'Obras na pista', 'Semaforização',
        'Pedestres', 'Crianças atravessando', 'Bicicletas atravessando',
        'Cuidado com gelo/neve', 'Animais silvestres atravessando',
        'Fim de todas as limitações de velocidade e ultrapassagem',
        'Curva à direita para frente', 'Curva à esquerda para frente',
        'Somente em frente', 'Siga em frente ou à direita',
        'Siga em frente ou à esquerda', 'Mantenha à direita',
        'Mantenha à esquerda', 'Rotatória obrigatória',
        'Fim da proibição de ultrapassagem',
        'Fim da proibição de ultrapassagem para veículos acima de 3.5 toneladas'
    ]
    return class_names[classNo]

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32), color_mode="grayscale")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalização
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    preds = getClassName(classIndex)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            basepath = os.path.dirname(__file__)
            uploads_dir = os.path.join(basepath, 'uploads')
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            file_path = os.path.join(uploads_dir, secure_filename(f.filename))
            f.save(file_path)
            preds = model_predict(file_path, model)
            return preds
    return "No file selected"

if __name__ == '__main__':
    app.run(port=5001, debug=True)
