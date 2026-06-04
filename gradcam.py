import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

def get_gradcam_layer(model):
    preferred = [
        'conv5_block3_out',
        'conv5_block3_3_bn',
        'conv5_block2_out',
        'conv4_block6_out',
    ]
    layer_names = [l.name for l in model.layers]
    for name in preferred:
        if name in layer_names:
            print(f"Grad-CAM target layer: {name}")
            return name
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Grad-CAM fallback layer: {layer.name}")
            return layer.name
    raise ValueError("No suitable conv layer found.")


def generate_gradcam(model, img_array, class_idx, layer_name=None):
    if layer_name is None:
        layer_name = get_gradcam_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs[0].numpy()

    # Weight each channel by its gradient
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)

    # ReLU — only positive activations matter
    heatmap = np.maximum(heatmap, 0)

    # ── Key fix 1: Gaussian smooth to reduce edge spike artifacts
    heatmap = gaussian_filter(heatmap, sigma=1)

    # Normalize
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    # ── Key fix 2: Threshold — suppress weak activations below 20%
    # This removes background noise and sharpens focus on true region
    heatmap[heatmap < 0.2] = 0

    # Re-normalize after threshold
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def overlay_heatmap(original_img_path, heatmap, output_path, alpha=0.5):
    img_bgr = cv2.imread(original_img_path)
    img_bgr = cv2.resize(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # ── Key fix 3: Apply brain mask — ignore pure black background pixels
    # Converts to grayscale, finds non-background pixels
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, brain_mask = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
    brain_mask = brain_mask.astype(np.float32)

    # Apply mask: zero out heatmap where there's no brain tissue
    heatmap_masked = heatmap_resized * brain_mask

    # Re-normalize after masking
    if heatmap_masked.max() != 0:
        heatmap_masked = heatmap_masked / heatmap_masked.max()

    heatmap_uint8   = np.uint8(255 * heatmap_masked)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_float     = img_rgb.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    overlay = (img_float * (1 - alpha) + heatmap_float * alpha).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#0d0d0d')

    panels = [
        (img_rgb,        None,  'Original MRI'),
        (heatmap_masked, 'jet', 'Grad-CAM Heatmap'),
        (overlay,        None,  'Overlay'),
    ]
    for ax, (data, cmap, title) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, color='white', fontsize=11, pad=6)
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='jet',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Attention', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    return output_path