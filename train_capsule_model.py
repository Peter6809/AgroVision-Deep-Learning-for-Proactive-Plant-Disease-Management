import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Set dataset path
train_data_dir = "/content/drive/MyDrive/Dataset/plants_data"  # Your dataset path
img_size = 128  # Image size
batch_size = 16  # Reduced batch size for better performance
max_images = 1000  # Maximum number of images for training

def squash(s, axis=-1):
    """Squash function for capsules"""
    squared_norm = tf.reduce_sum(tf.square(s), axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + 1e-8)
    return scale * s

def build_capsule_network(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Feature extraction with Conv layers
    x = layers.Conv2D(64, (3, 3), activation='swish', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='swish', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='swish', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Primary Capsule Layer
    x = layers.Conv2D(32 * 8, (3, 3), activation='swish', padding='same')(x)
    x = layers.Reshape((-1, 8))(x)
    x = layers.Lambda(squash)(x)

    # Fully Connected Layer
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="swish")(x)
    x = layers.Dropout(0.4)(x)  # Prevent overfitting

    # Output Layer
    capsule_output = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, capsule_output, name="CapsuleNet")
    return model

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

num_classes = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())

# Function to limit dataset to 1000 images
def limit_dataset(generator, max_images):
    image_list, label_list = [], []
    total_images = 0
    while total_images < max_images:
        x_batch, y_batch = next(generator)  # Load batch
        image_list.append(x_batch)
        label_list.append(y_batch)
        total_images += len(x_batch)
        if total_images >= max_images:
            break  # Stop once we reach 1000 images
    return np.vstack(image_list)[:max_images], np.vstack(label_list)[:max_images]

# Load only 1000 images for training
train_images, train_labels = limit_dataset(train_generator, max_images=1000)

# Build and compile model with Adam optimizer
capsule_model = build_capsule_network((img_size, img_size, 3), num_classes)
capsule_model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Learning rate adjustment and early stopping
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6)
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# Train model with 1000 images
capsule_model.fit(
    train_images, train_labels,
    epochs=10,  # Increased epochs for better training
    batch_size=batch_size,
    callbacks=[reduce_lr, early_stop]
)

# Save model and class labels
os.makedirs("outputs", exist_ok=True)
capsule_model.save("outputs/capsule_plant_disease.h5")
np.save("outputs/cap_lables.npy", class_labels)

print("Capsule Network trained and saved successfully.")
