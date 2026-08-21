import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

from gradcam import make_gradcam_heatmap


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/best_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

OUTPUT_PATH = "model/gradcam_overlay.png"

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Naksh_AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

import json

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:

    class_names = json.load(file)


# ============================================================
# SELECT IMAGE
# ============================================================

IMAGE_PATH = "dataset/test/star"


image_file = None

for filename in sorted(
    os.listdir(IMAGE_PATH)
):

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
        f"No image found in {IMAGE_PATH}"
    )


print(
    "Image:",
    image_file
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    image_file
).convert("RGB")


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print(
    "Generating Grad-CAM..."
)

heatmap, predicted_index = make_gradcam_heatmap(
    image,
    model,
    "conv_1"
)


predicted_class = class_names[
    predicted_index
]


print(
    "Prediction:",
    predicted_class
)


# ============================================================
# PREPARE ORIGINAL IMAGE
# ============================================================

original = np.array(
    image
)


# ============================================================
# RESIZE HEATMAP
# ============================================================

heatmap_image = Image.fromarray(
    np.uint8(
        heatmap * 255
    )
)

heatmap_image = heatmap_image.resize(
    image.size,
    Image.Resampling.BILINEAR
)

heatmap_resized = np.array(
    heatmap_image
) / 255.0


# ============================================================
# CREATE COLORED HEATMAP
# ============================================================

colormap = plt.get_cmap(
    "jet"
)

colored_heatmap = colormap(
    heatmap_resized
)

colored_heatmap = np.uint8(
    colored_heatmap[:, :, :3] * 255
)


# ============================================================
# CREATE OVERLAY
# ============================================================

overlay = (
    0.55 * original
    +
    0.45 * colored_heatmap
)

overlay = np.uint8(
    np.clip(
        overlay,
        0,
        255
    )
)


# ============================================================
# SAVE OVERLAY
# ============================================================

Image.fromarray(
    overlay
).save(
    OUTPUT_PATH
)


# ============================================================
# CREATE THREE-PANEL IMAGE
# ============================================================

plt.figure(
    figsize=(15, 5)
)


# Original

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    original
)

plt.title(
    "Original Image"
)

plt.axis(
    "off"
)


# Heatmap

plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    heatmap_resized,
    cmap="jet"
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis(
    "off"
)


# Overlay

plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    overlay
)

plt.title(
    f"AI Focus — {predicted_class.upper()}"
)

plt.axis(
    "off"
)


plt.tight_layout()


# Save three panel

plt.savefig(
    "model/gradcam_result.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 50)
print("GRAD-CAM COMPLETE")
print("=" * 50)

print(
    "Prediction:",
    predicted_class
)

print(
    "Overlay:",
    OUTPUT_PATH
)

print(
    "Three-panel result:",
    "model/gradcam_result.png"
)
