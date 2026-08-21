import json
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "dataset/test"

MODEL_PATH = "model/best_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("\n========================================")
print("NAKSH_AI CLASS NAMES")
print("========================================")

for i, name in enumerate(class_names):
    print(f"{i} -> {name}")


# ==========================================
# LOAD TEST DATASET
# ==========================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names
)

print("\nTest dataset loaded.")


# ==========================================
# LOAD BEST MODEL
# ==========================================

print("\nLoading best trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# BASIC TEST EVALUATION
# ==========================================

print("\n========================================")
print("TESTING MODEL")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ==========================================
# GET PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ==========================================
# CREATE CONFUSION MATRIX IMAGE
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "Naksh_AI - Confusion Matrix"
)

plt.tight_layout()

output_path = "model/confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=200
)

plt.close()

print(
    f"\nConfusion matrix saved to: {output_path}"
)


# ==========================================
# FINISHED
# ==========================================

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")
