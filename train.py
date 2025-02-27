import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import json

# def convert_to_rgb(image):
#     if isinstance(image, np.ndarray):  
#         image=Image.fromarray(image)
#     if image.mode in ("RGBA", "P"):  
#         image = image.convert("RGB")
#     return np.array(image)

train_dir='train'
validation_dir='validation'
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    # preprocessing_function=convert_to_rgb,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen=ImageDataGenerator(
    rescale=1.0/255,
    # preprocessing_function=convert_to_rgb
    )
train_generator= train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

model=Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(36, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(train_generator, epochs=10, validation_data=validation_generator, verbose=1)
class_labels_reversed = {str(value): key for key, value in train_generator.class_indices.items()}
with open("class_labels.json","w") as f:
    json.dump(class_labels_reversed, f,indent=4)
model.save('vegdetsys.h5')