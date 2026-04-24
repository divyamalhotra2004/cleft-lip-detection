import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import os

# ---------- CONFIG ----------
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5

DATASET_PATH = "dataset/train"

# ---------- DATA GENERATOR ----------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print("Classes:", train_data.class_indices)

# ---------- MODEL ----------
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze most layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Custom head
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# ---------- COMPILE ----------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------- TRAIN ----------
print("\n🚀 Starting Training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ---------- SAVE MODEL ----------
os.makedirs("model", exist_ok=True)
model.save("model/cleft_model.h5")

print("\n✅ Model trained and saved successfully!")