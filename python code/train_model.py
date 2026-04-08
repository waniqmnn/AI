import numpy as np
import matplotlib.pyplot as plt
import os
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ==============================
# Paths
# ==============================
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_SAVE_PATH = "models/leaf_cnn_model_improved.h5"
RESULTS_DIR = "results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# Data Augmentation (leaf-safe)
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes

print("\nDetected classes:")
print(train_generator.class_indices)

# ==============================
# MobileNetV2 Transfer Learning
# ==============================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# 🔒 Freeze base model completely
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# Callbacks
# ==============================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ==============================
# TRAIN (PHASE 1 ONLY)
# ==============================
print("\n=== TRAINING (NO PHASE 2) ===")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# ==============================
# TEST EVALUATION
# ==============================
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print("\n==============================")
print(f"TEST ACCURACY : {test_accuracy * 100:.2f}%")
print(f"TEST LOSS     : {test_loss:.4f}")
print("==============================")

# ==============================
# Plot Training Curves
# ==============================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.axhline(test_accuracy, color="r", linestyle="--", label="Test Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_results.png"), dpi=300)
plt.show()

# ==============================
# Save Class Indices
# ==============================
with open("models/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=4)

print("\n✓ TRAINING COMPLETE")
print(f"✓ Model saved at: {MODEL_SAVE_PATH}")
print("✓ Phase 2 intentionally removed to preserve accuracy")
