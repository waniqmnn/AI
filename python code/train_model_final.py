# pylint: skip-file

import numpy as np
import matplotlib.pyplot as plt
import os
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ==============================
# Paths
# ==============================
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_SAVE_PATH = "models/leaf_final_model.h5"
RESULTS_DIR = "results_final"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# Data Augmentation
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_indices_final = train_generator.class_indices
NUM_CLASSES = train_generator.num_classes

print("Detected classes:", class_indices_final)
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Save class mapping
with open(os.path.join("models", "class_indices_final.json"), 'w') as f:
    json.dump(class_indices_final, f, indent=4)

# ==============================
# Custom CNN Model
# ==============================
def build_improved_cnn(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model

# ==============================
# Transfer Learning Model
# ==============================
def build_transfer_learning_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model, base_model

# Choose model
USE_TRANSFER_LEARNING = True

if USE_TRANSFER_LEARNING:
    model, base_model = build_transfer_learning_model(NUM_CLASSES)
    print("Using Transfer Learning with MobileNetV2")
else:
    model = build_improved_cnn(NUM_CLASSES)
    print("Using Custom CNN")

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
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==============================
# Training
# ==============================
if USE_TRANSFER_LEARNING:
    # Phase 1
    print("\n=== Phase 1: Training with frozen base model ===")
    history_phase1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    # Phase 2: Fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n=== Phase 2: Fine-tuning unfrozen layers ===")
    history_phase2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=30,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    # Combine histories
    history = type('obj', (object,), {
        'history': {
            'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
            'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
            'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
            'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
        }
    })()
else:
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

# ==============================
# Evaluate on Test Set
# ==============================
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ==============================
# Plot Training Results
# ==============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
ax1.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy: {test_accuracy:.3f}')
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.set_title("Model Accuracy", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history["loss"], label="Training Loss", linewidth=2)
ax2.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Model Loss", fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_results.png"), dpi=300)
plt.show()

print("\n✓ Training complete!")
print(f"✓ Best model saved to: {MODEL_SAVE_PATH}")
print(f"✓ Class mapping saved to: models/class_indices_final.json")
