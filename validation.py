import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image
model=tf.keras.models.load_model('vegdetsys.h5')
with open("class_labels.json","r") as f:
    class_labels=json.load(f)

test_dir='test'
results={}

def predict_image(filepath):
    img=image.load_img(filepath, target_size=(128, 128))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array, axis=0)/255.0
    
    prediction=model.predict(img_array)
    predicted_class=class_labels[str(np.argmax(prediction))]
    confidence=np.max(prediction)

    return predicted_class,confidence

for folder in os.listdir(test_dir):
    folder_path=os.path.join(test_dir,folder)

    if os.path.isdir(folder_path):
        results[folder]=[]

        for filename in os.listdir(folder_path):
            filepath=os.path.join(folder_path,filename)

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