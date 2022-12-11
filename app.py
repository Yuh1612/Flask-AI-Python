from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import time

def current_milli_time():
    return round(time.time() * 1000)

app = Flask(__name__)


graph = tf.compat.v1.get_default_graph()

def model_predict(img_path):
    model = ResNet50(weights="imagenet")
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/videoclassification", methods=["GET"])
def videoclassification():
	return render_template("video.html")

@app.route("/imageclassification", methods=["GET"])
def imageclassfication():
    return render_template("image.html")

@app.route("/predict", methods=["GET", "POST"])
def upload():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            f = request.files['image']

            basepath = os.path.dirname(__file__)

            file_path = os.path.join(
                basepath, 'uploads', secure_filename(str(current_milli_time()) + "-" + f.filename))
            f.save(file_path)

            preds = model_predict(file_path)

            pred_class = decode_predictions(preds, top=1)
            result = str(pred_class[0][0][1])             
            return result
        return None

@app.route("/movementclassification", methods=["GET"])
def movementclassification():
	return render_template("movement.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
	http_server = WSGIServer(('0.0.0.0', 5000), app)
	http_server.serve_forever()

