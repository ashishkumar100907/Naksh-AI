import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# NAKSH_AI - 4 CLASS MODEL
# ==========================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validate"

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# CHECK DIRECTORIES
# ==========================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not os.path.exists(VALIDATION_DIR):
    raise FileNotFoundError(
        f"Validation folder not found: {VALIDATION_DIR}"
    )

# ==========================================
# LOAD TRAINING DATA
# ==========================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================================
# LOAD VALIDATION DATA
# ==========================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# GET CLASS NAMES
# ==========================================

class_names = train_dataset.class_names

print("\n========================================")
print("NAKSH_AI CLASSES")
print("========================================")

for index, class_name in enumerate(class_names):
    print(f"{index} -> {class_name}")

print("========================================")

if len(class_names) != 4:
    raise ValueError(
        f"Expected 4 classes, but found {len(class_names)}"
    )

# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# ==========================================
# PRETRAINED MOBILENETV3
# ==========================================

print("\nLoading MobileNetV3Small...")

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze pretrained model
base_model.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v3.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    4,
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========================================")
print("MODEL CREATED")
print("========================================")

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath="model/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
]

# ==========================================
# TRAIN
# ==========================================

print("\n========================================")
print("STARTING NAKSH_AI TRAINING")
print("========================================\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

final_model_path = "model/naksh_ai_model.keras"

model.save(final_model_path)

# ==========================================
# SAVE CLASS NAMES
# ==========================================

with open(
    "model/class_names.json",
    "w"
) as file:

    json.dump(
        class_names,
        file
    )

# ==========================================
# COMPLETE
# ==========================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "Best model: model/best_model.keras"
)

print(
    "Final model: model/naksh_ai_model.keras"
)

print(
    "Classes:",
    class_names
)
