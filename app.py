from flask import Flask, render_template,request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json

app=Flask(__name__)
model=load_model("vegdetsys.h5")
with open("class_labels.json","r") as f:
    class_labels=json.load(f)

@app.route("/")
def home_page():
    return render_template('index.html')

@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method == "POST":
        file=request.files['image']
        img=Image.open(file).convert("RGB").resize((128,128))
        img_array=np.expand_dims(image.img_to_array(img),axis=0)/255.0

        prediction=model.predict(img_array)
        predicted_class=class_labels[str(np.argmax(prediction))]
        confidence=np.max(prediction)*100
        return render_template('result.html',vegetable=predicted_class,confidence=confidence)
    
    