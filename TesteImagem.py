from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH ='model.h5'

model = load_model(MODEL_PATH)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getClassName(classNo):
    if   classNo == 0: return 'Limite de velocidade 20 km/h'
    elif classNo == 1: return 'Limite de velocidade 30 km/h'
    elif classNo == 2: return 'Limite de velocidade 50 km/h'
    elif classNo == 3: return 'Limite de velocidade 60 km/h'
    elif classNo == 4: return 'Limite de velocidade 70 km/h'
    elif classNo == 5: return 'Limite de velocidade 80 km/h'
    elif classNo == 6: return 'Fim de limite de velocidade (80km/h)'
    elif classNo == 7: return 'Limite de velocidade 100 km/h'
    elif classNo == 8: return 'Limite de velocidade 120 km/h'
    elif classNo == 9: return 'Ultrapassagem proibida'
    elif classNo == 10: return 'Ultrapassagem proibida para veículos acima de 3.5 toneladas'
    elif classNo == 11: return 'Cedência de passagem no próximo cruzamento'
    elif classNo == 12: return 'Estrada com prioridade'
    elif classNo == 13: return 'Dê a preferência'
    elif classNo == 14: return 'Pare'
    elif classNo == 15: return 'Veiculos proibidos'
    elif classNo == 16: return 'Veículos acima de 3.5 toneladas proibidos'
    elif classNo == 17: return 'Entrada proibida'
    elif classNo == 18: return 'Atenção'
    elif classNo == 19: return 'Curva perigosa à esquerda'
    elif classNo == 20: return 'Curva perigosa à direita'
    elif classNo == 21: return 'Curva dupla'
    elif classNo == 22: return 'Estrada irregular'
    elif classNo == 23: return 'Piso escorregadio'
    elif classNo == 24: return 'strada estreita à direita'
    elif classNo == 25: return 'Obras na pista'
    elif classNo == 26: return 'Semaforização'
    elif classNo == 27: return 'Pedestres'
    elif classNo == 28: return 'Crianças atravessando'
    elif classNo == 29: return 'Bicicletas atravessando'
    elif classNo == 30: return 'Cuidado com gelo/neve'
    elif classNo == 31: return 'Animais silvestres atravessando'
    elif classNo == 32: return 'Fim de todas as limitações de velocidade e ultrapassagem'
    elif classNo == 33: return 'Curva à direita para frente'
    elif classNo == 34: return 'Curva à esquerda para frente'
    elif classNo == 35: return 'Somente em frente'
    elif classNo == 36: return 'Siga em frente ou à direita'
    elif classNo == 37: return 'Siga em frente ou à esquerda'
    elif classNo == 38: return 'Mantenha à direita'
    elif classNo == 39: return 'Mantenha à esquerda'
    elif classNo == 40: return 'Rotatória obrigatória'
    elif classNo == 41: return 'Fim da proibição de ultrapassagem'
    elif classNo == 42: return 'Fim da proibição de ultrapassagem para veículos acima de 3.5 toneladas'


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    # probabilityValue =np.amax(predictions)
    preds = getClassName(classIndex)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
