import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd

# ==============================
# Paths
# ==============================
TEST_DIR = "dataset/test"
MODEL_PATH = "models/leaf_final_model.h5"  # Use improved model
RESULTS_DIR = "results"
CLASS_INDICES_PATH = "models/class_indices_final.json"

# Update IMG_SIZE to match training (224x224 if using improved model)
IMG_SIZE = (224, 224)  # Changed from 128x128
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# Load trained model
# ==============================
print("Loading model...")
model = load_model(MODEL_PATH)
print("✓ Model loaded successfully")

# ==============================
# Load class mapping
# ==============================
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices_final = json.load(f)
    print("✓ Class mapping loaded")
else:
    print("⚠ Class mapping file not found, will use generator's mapping")
    class_indices_final = None

# ==============================
# Prepare test data
# ==============================
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

if class_indices_final is not None:
    class_labels = list(class_indices_final.keys())
else:
    # Fallback in case JSON mapping not found
    class_labels = list(test_generator.class_indices.keys())
num_classes = len(class_labels)


print(f"\nFound {test_generator.samples} test images")
print(f"Classes: {class_labels}")

# ==============================
# Model prediction
# ==============================
print("\nMaking predictions...")
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Get prediction confidence scores
y_pred_proba = np.max(predictions, axis=1)

# ==============================
# Overall Metrics
# ==============================
overall_accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print("\n" + "="*50)
print("OVERALL MODEL PERFORMANCE")
print("="*50)
print(f"Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1-Score: {f1:.4f}")
print(f"Average Confidence: {np.mean(y_pred_proba):.4f}")
print("="*50)

# ==============================
# Detailed Classification Report
# ==============================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    digits=4
)

print("\n" + report)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write("OVERALL METRICS\n")
    f.write("="*50 + "\n")
    f.write(f"Test Accuracy: {overall_accuracy:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1-Score: {f1:.4f}\n")
    f.write(f"Average Confidence: {np.mean(y_pred_proba):.4f}\n")
    f.write("="*50 + "\n\n")
    f.write("PER-CLASS METRICS\n")
    f.write("="*50 + "\n")
    f.write(report)

# ==============================
# Enhanced Confusion Matrix
# ==============================
cm = confusion_matrix(y_true, y_pred)

# Calculate per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Ini yang diubah, asal (16,7)

# Confusion Matrix (counts)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels,
    cmap="Blues",
    ax=ax1,
    cbar_kws={'label': 'Count'}
)
ax1.set_xlabel("Predicted Label", fontsize=12)
ax1.set_ylabel("True Label", fontsize=12)
ax1.set_title(f"Confusion Matrix (Counts)\nOverall Accuracy: {overall_accuracy:.2%}", 
              fontsize=14, fontweight='bold')

# Confusion Matrix (normalized)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2%",
    xticklabels=class_labels,
    yticklabels=class_labels,
    cmap="RdYlGn",
    ax=ax2,
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Percentage'}
)
ax2.set_xlabel("Predicted Label", fontsize=12)
ax2.set_ylabel("True Label", fontsize=12)
ax2.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# Per-Class Performance Bar Chart
# ==============================
precision_per_class, recall_per_class, f1_per_class, support_per_class = \
    precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(num_classes))

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(class_labels))
width = 0.25

bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
bars2 = ax.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Leaf Class', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_labels, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "per_class_performance.png"), dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# Confidence Distribution Analysis
# ==============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))#adjustment 1 asal(14,5)

# Overall confidence distribution
ax1.hist(y_pred_proba, bins=30, edgecolor='black', alpha=0.7)  
ax1.axvline(np.mean(y_pred_proba), color='r', linestyle='--', 
            label=f'Mean: {np.mean(y_pred_proba):.3f}')
ax1.axvline(np.median(y_pred_proba), color='g', linestyle='--', 
            label=f'Median: {np.median(y_pred_proba):.3f}')
ax1.set_xlabel('Prediction Confidence', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Correct vs Incorrect predictions confidence
correct_mask = (y_pred == y_true)
correct_conf = y_pred_proba[correct_mask]
incorrect_conf = y_pred_proba[~correct_mask]

ax2.hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
ax2.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
ax2.set_xlabel('Prediction Confidence', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Confidence: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confidence_analysis.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Correct predictions: {np.sum(correct_mask)} ({np.mean(correct_mask)*100:.2f}%)")
print(f"✓ Average confidence (correct): {np.mean(correct_conf):.4f}")
print(f"✗ Incorrect predictions: {np.sum(~correct_mask)} ({np.mean(~correct_mask)*100:.2f}%)")
print(f"✗ Average confidence (incorrect): {np.mean(incorrect_conf) if len(incorrect_conf) > 0 else 0:.4f}")

# ==============================
# Find Most Confused Class Pairs
# ==============================
print("\n" + "="*50)
print("MOST CONFUSED CLASS PAIRS")
print("="*50)

confused_pairs = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((class_labels[i], class_labels[j], cm[i, j]))

confused_pairs.sort(key=lambda x: x[2], reverse=True)

for true_class, pred_class, count in confused_pairs[:5]:
    print(f"{true_class} → {pred_class}: {count} times")

# ==============================
# Find Low Confidence Predictions
# ==============================
low_confidence_threshold = 0.5
low_conf_indices = np.where(y_pred_proba < low_confidence_threshold)[0]

if len(low_conf_indices) > 0:
    print(f"\n⚠ Found {len(low_conf_indices)} predictions with confidence < {low_confidence_threshold}")
    print("These samples may need review or additional training data.")
else:
    print(f"\n✓ All predictions have confidence ≥ {low_confidence_threshold}")

# ==============================
# Export Detailed Results to CSV
# ==============================
results_df = pd.DataFrame({
    'filename': test_generator.filenames,
    'true_class': [class_labels[i] for i in y_true],
    'predicted_class': [class_labels[i] for i in y_pred],
    'confidence': y_pred_proba,
    'correct': (y_pred == y_true)
})

results_df.to_csv(os.path.join(RESULTS_DIR, "detailed_predictions.csv"), index=False)
print(f"\n✓ Detailed predictions saved to: {RESULTS_DIR}/detailed_predictions.csv")

# ==============================
# Single Image Prediction Demo with Visualization
# ==============================
sample_image_path = "dataset/test/palmate/1493.jpg"

if os.path.exists(sample_image_path):
    print("\n" + "="*50)
    print("SINGLE IMAGE PREDICTION DEMO")
    print("="*50)
    
    img = load_img(sample_image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array_batch, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(prediction[0])[-3:][::-1]
    
    print(f"Image: {sample_image_path}")
    print(f"\nTop 3 Predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"{i}. {class_labels[idx]}: {prediction[0][idx]*100:.2f}%")
    
    # Visualize prediction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # adjustment 2 asal (12,5)
    
    # Show image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f"Predicted: {predicted_class}\nConfidence: {confidence*100:.2f}%",
                  fontsize=12, fontweight='bold')
    
    # Show prediction probabilities
    colors = ['green' if i == predicted_class_idx else 'gray' for i in range(num_classes)]
    bars = ax2.barh(class_labels, prediction[0], color=colors, alpha=0.7)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    # Highlight top prediction
    bars[predicted_class_idx].set_color('darkgreen')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_prediction.png"), dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "="*50)
print("EVALUATION COMPLETE")
print("="*50)
print(f"✓ All results saved to: {RESULTS_DIR}/")
print(f"  - classification_report.txt")
print(f"  - confusion_matrix.png")
print(f"  - per_class_performance.png")
print(f"  - confidence_analysis.png")
print(f"  - detailed_predictions.csv")
print(f"  - sample_prediction.png")