


import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt') 

st.title("YOLO Object Detection")
st.markdown("Upload an image to detect fruits and see the count of each type.")

image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if image_file is not None:
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, channels="RGB", use_column_width=True)

    results = model(img)
    item_dict = {}

    for result in results:
        detections = result.boxes.xyxy
        class_ids = result.boxes.cls
        names = result.names

        for class_id in class_ids:
            item_name = names[int(class_id)]
            if item_name in item_dict:
                item_dict[item_name] += 1
            else:
                item_dict[item_name] = 1

    st.subheader("Detected Items:")
    if item_dict:
        for item_name, count in item_dict.items():
            st.markdown(f"The number of **{item_name}** is **{count}**.")
    else:
        st.write("No items detected.")
