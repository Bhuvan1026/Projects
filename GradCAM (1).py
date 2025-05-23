
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.layers import Conv2D

# ✅ Automatically get the name of the last Conv2D layer
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

# ✅ Generate Grad-CAM heatmap
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
        

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# ✅ Overlay heatmap on the original image
def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, original_img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    img = np.array(original_img)
    if img.shape[-1] == 4:  # remove alpha if present
        img = img[..., :3]

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# ✅ Full prediction pipeline with Grad-CAM
def predict_single_image_with_gradcam(image_path, model_path):
    model = load_model(model_path)
    last_conv_layer_name = get_last_conv_layer_name(model)

    # Load & preprocess
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "REAL" if prediction >= 0.5 else "FAKE"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    # Grad-CAM
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
    original_img = image.load_img(image_path)
    superimposed_img = overlay_heatmap(heatmap, original_img)

    # Show results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title(f"Grad-CAM: {result} (Confidence: {confidence:.2f})")
    plt.show()

    print(f"📌 Image: {image_path}")
    print(f"🔍 Prediction: {result}")
    print(f"✅ Confidence: {confidence:.4f}")

# ✅ Example usage
predict_single_image_with_gradcam(
    "/home/netweb/Documents/CIFAKE/0001 (6).jpg",
    "/home/netweb/Documents/CIFAKE/saved_models/best_model_64u_1l_f1_0.948.h5"
)

