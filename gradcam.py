import tensorflow as tf
import numpy as np


IMAGE_SIZE = (224, 224)


def make_gradcam_heatmap(
    image,
    model,
    last_conv_layer_name="conv_1"
):

    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image
    ).astype(np.float32)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ========================================================
    # FIND THE TWO IMPORTANT PARTS
    # ========================================================

    preprocessing_model = None
    base_model = None

    for layer in model.layers:

        if (
            isinstance(layer, tf.keras.Model)
            and layer.name == "sequential"
        ):
            preprocessing_model = layer

        if (
            isinstance(layer, tf.keras.Model)
            and layer.name == "MobileNetV3Small"
        ):
            base_model = layer


    if preprocessing_model is None:

        raise ValueError(
            "Preprocessing/augmentation model "
            "was not found."
        )


    if base_model is None:

        raise ValueError(
            "MobileNetV3Small model "
            "was not found."
        )


    # ========================================================
    # FIND CONVOLUTIONAL LAYER
    # ========================================================

    last_conv_layer = base_model.get_layer(
        last_conv_layer_name
    )

    print(
        "Using Grad-CAM layer:",
        last_conv_layer.name
    )


    # ========================================================
    # CREATE MODEL INSIDE MOBILENET
    #
    # This model takes MobileNet's own input and returns:
    #
    # 1. conv_1 feature maps
    # 2. MobileNet final feature output
    # ========================================================

    base_grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )


    # ========================================================
    # GRADIENT CALCULATION
    # ========================================================

    with tf.GradientTape() as tape:

        # Apply the same preprocessing/augmentation
        # pipeline used by the original model.

        processed_image = preprocessing_model(
            image_array,
            training=False
        )


        # Run through MobileNet

        conv_outputs, base_output = base_grad_model(
            processed_image,
            training=False
        )


        # ----------------------------------------------------
        # Reproduce the classification head
        # ----------------------------------------------------

        x = base_output


        # Find the pooling, dropout and dense layers

        for layer in model.layers:

            if layer.name == "global_average_pooling2d":

                x = layer(x)

            elif layer.name == "dropout":

                x = layer(
                    x,
                    training=False
                )

            elif layer.name == "dense":

                x = layer(x)


        predictions = x


        # ----------------------------------------------------
        # Find predicted class
        # ----------------------------------------------------

        predicted_index = tf.argmax(
            predictions[0]
        )


        class_output = predictions[
            0,
            predicted_index
        ]


    # ========================================================
    # CALCULATE GRADIENTS
    # ========================================================

    grads = tape.gradient(
        class_output,
        conv_outputs
    )


    if grads is None:

        raise ValueError(
            "Gradients are None. "
            "Grad-CAM could not calculate gradients."
        )


    # ========================================================
    # GLOBAL AVERAGE POOLING OF GRADIENTS
    # ========================================================

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )


    # ========================================================
    # REMOVE BATCH DIMENSION
    # ========================================================

    conv_outputs = conv_outputs[0]


    # ========================================================
    # CREATE HEATMAP
    # ========================================================

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )


    # ========================================================
    # KEEP POSITIVE ACTIVATIONS
    # ========================================================

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # ========================================================
    # NORMALIZE HEATMAP
    # ========================================================

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value +
        tf.keras.backend.epsilon()
    )


    return (
        heatmap.numpy(),
        int(predicted_index.numpy())
    )
