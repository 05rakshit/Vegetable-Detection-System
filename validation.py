import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image
from PIL import Image
model=tf.keras.models.load_model('vegdetsys.h5')
with open("class_labels.json","r") as f:
    class_labels=json.load(f)

test_dir='test'
results={}

def predict_image(filepath):
    try:    
        with Image.open(filepath) as img:
            img=img.convert("RGB")
            img=img.resize((128,128))
        img_array=image.img_to_array(img)
        img_array=np.expand_dims(img_array, axis=0)/255.0
    
        prediction=model.predict(img_array)
        predicted_class=class_labels[str(np.argmax(prediction))]
        confidence=np.max(prediction)*100
    except Exception as e:
        raise ValueError(f"Error processing {filepath}: {e}")

    return predicted_class,confidence

for folder in os.listdir(test_dir):
    folder_path=os.path.join(test_dir,folder)
    
    if not os.path.isdir(folder_path):
        continue

    results[folder]=[]

    for filename in os.listdir(folder_path):
        filepath=os.path.join(folder_path,filename)

        if not filename.lower().endswith(('.png','.jpg','.jpeg')):
            continue

        try:
            pred_class, conf=predict_image(filepath)
            results[folder].append({
                "image":filename,
                "predicted_class":pred_class,
                "confidence":f"{conf:.2f}%"
            })
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")

with open("results.json","w") as f:
    json.dump(results, f, indent=4)

print("Prediction results saved to results.json")