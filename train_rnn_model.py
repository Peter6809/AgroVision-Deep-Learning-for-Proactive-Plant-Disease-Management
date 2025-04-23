import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Set dataset path
DATASET_PATH = "data/plants_data"

# Parameters
IMG_SIZE = (128, 128)  # Reduced image size for faster training
BATCH_SIZE = 64        # Increased batch size
EPOCHS = 5            # Reduced epochs
MAX_IMAGES = 1000     # Limit number of images for faster training

# Load dataset and preprocess images
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class names
class_names = list(train_generator.class_indices.keys())
np.save("class_names.npy", class_names)
print("Class names saved:", class_names)

# Define RNN-based model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model checkpoint (saves best model)
checkpoint = ModelCheckpoint("outputs/rnn_best_model.keras", save_best_only=True, monitor='val_loss')

# Train model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint]  
)

# Save model
model.save("outputs/RNN_model.keras")
print("Model saved successfully!")

