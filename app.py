import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import time
from PIL import Image
import gdown

import warnings
warnings.filterwarnings("ignore")


@register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    iou = tf.where(K.equal(union, 0), 0.0, intersection / union)
    return iou
@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
@register_keras_serializable()
def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.6 * dice + 0.4 * bce
    
file_id = "1RgisjfzT7Xkj23PH7r3ysMjX9nCqPBhZ/view?usp=sharing"  # Thay thế bằng ID thực tế của bạn
url = f"https://drive.google.com/drive/folders/{file_id}"
model_path = "model_final.h5"

if not os.path.exists(model_path):
    st.write("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

@st.cache_resource()
def load_model_func():
    model = load_model(model_path)
    return model
with st.spinner('Model is being loaded..'):
    model=load_model_func()

# Image processing
def process_image(image):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0 
    image = tf.image.random_brightness(image, max_delta=0.1)  # Điều chỉnh độ sáng ngẫu nhiên
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Điều chỉnh độ tương phản ngẫu nhiên 
    
    return image

# Predict mask
def predict_mask(image):
    # Dự đoán mask
    mask = model.predict(tf.expand_dims(image, axis=0))
    mask = np.squeeze(mask)  # Loại bỏ batch dimension
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, (image.shape[1], image.shape[0]))  
    mask = tf.cast(mask > 0.5, tf.float32) 
    
    return mask

# mask to image
def apply_mask(image, mask):
    mask_colored = tf.image.grayscale_to_rgb(mask)
    alpha = 0.1
    mask_color = [0, 255, 0]
    mask_colored = tf.multiply(mask_colored, mask_color)

    # (1 - alpha) * image + alpha * mask_colored
    masked_image = tf.add(tf.multiply(image, (1 - alpha)), tf.multiply(mask_colored, alpha))
    return masked_image

# Streamlit
st.title('Cегментированных полипов в колоноскопии')
st.write("Загрузить фотографии или видео эндоскопии")


# Upload image or video
file = st.file_uploader("", type=["jpg", "png", "mp4", "avi"])

col1, col2 = st.columns(2)

if file is None:
    st.text("Please upload an image file")
else:
    start_time = time.time()
    if file.type == "image/png" or file.type == "image/jpg" :
        image = Image.open(file)
        image_show = image.resize((256, 256))
        with col1:
            st.image(image_show, caption = 'Оригинальное фото', use_container_width=True)
        
        # add mask to image
        image = process_image(image)
        mask = predict_mask(image)
        masked_image = apply_mask(image, mask)

        # EagerTensor to PIL image
        masked_image = masked_image.numpy()
        masked_image = np.clip(masked_image * 255.0, 0 , 225).astype(np.uint8)
        masked_image = Image.fromarray(masked_image)

        # show output
        with col2:
            st.image(masked_image, caption = 'Расположение полипа', use_container_width=True)
        end_time = time.time()
        execution_time = end_time - start_time

        st.text(f"Время обработки: {round(execution_time, 2)} секунд")

    elif file.type == "video/mp4" or file.type == "video/avi":
        video = cv2.VideoCapture(file)

        frames = []
        processed_frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_image(frame_rgb)
            mask = predict_mask(processed_frame)
            masked_frame = apply_mask(processed_frame, mask)

            # Convert processed frame to BGR for display
            masked_frame_bgr = np.clip(masked_frame.numpy() * 255.0, 0, 255).astype(np.uint8)
            masked_frame_bgr = cv2.cvtColor(masked_frame_bgr, cv2.COLOR_RGB2BGR)

            frames.append(frame)
            processed_frames.append(masked_frame_bgr)

        # Display original and processed video
        for orig_frame, processed_frame in zip(frames, processed_frames):
            with col1:
                st.image(orig_frame, caption="Оригинальное видео", use_container_width=True)
            
            with col2:
                st.image(processed_frame, caption="Расположение полипа", use_container_width=True)
