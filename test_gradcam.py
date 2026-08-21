import tensorflow as tf
import json
from PIL import Image
from gradcam import make_gradcam_heatmap


MODEL_PATH = "model/best_model.keras"


# Load model

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# Load class names

with open(
    "model/class_names.json",
    "r"
) as file:

    class_names = json.load(file)


# ------------------------------------------------------------
# CHANGE THIS IMAGE PATH
# ------------------------------------------------------------

IMAGE_PATH = "dataset/test/planet"


# Find first JPG image

import os

image_file = None

for filename in os.listdir(IMAGE_PATH):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        image_file = os.path.join(
            IMAGE_PATH,
            filename
        )

        break


if image_file is None:

    raise FileNotFoundError(
        "No image found."
    )


print(
    "Testing image:",
    image_file
)


# Load image

image = Image.open(
    image_file
).convert("RGB")


# Generate heatmap

heatmap, predicted_index = make_gradcam_heatmap(
    image,
    model,
    "conv_1"
)


print(
    "\nPredicted:",
    class_names[predicted_index]
)

print(
    "Heatmap shape:",
    heatmap.shape
)

print(
    "Heatmap min:",
    heatmap.min()
)

print(
    "Heatmap max:",
    heatmap.max()
)


# Save heatmap

import numpy as np

heatmap_image = Image.fromarray(
    np.uint8(255 * heatmap)
)

heatmap_image = heatmap_image.resize(
    image.size
)

heatmap_image.save(
    "model/test_heatmap.png"
)


print(
    "\nHeatmap saved:"
)

print(
    "model/test_heatmap.png"
)
