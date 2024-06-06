import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO  # Assuming YOLO from ultralytics is used
from streamlit_image_comparison import image_comparison
import wget

# Configurations
CFG_MODEL_PATH = "best_yolov8nano.pt"
CFG_ENABLE_URL_DOWNLOAD = True
if CFG_ENABLE_URL_DOWNLOAD:
    # Configure this if you set CFG_ENABLE_URL_DOWNLOAD to True
    url = "https://archive.org/download/best_yolov8nano/best_yolov8nano.pt"
# End of Configurations

# Custom CSS to change the sidebar background color
st.markdown(
    """
    <style>
    .css-1d391kg {  /* This class may need to be updated if Streamlit changes their class names */
        background-color: black;
        color: white;
    }
    .css-17eq0hr {
        background-color: black;
        color: white;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def download_model():
    if not os.path.exists(CFG_MODEL_PATH):
        wget.download(url, out=CFG_MODEL_PATH)

@st.cache_resource
def load_model():
    return YOLO(CFG_MODEL_PATH)

def detect_people(image, model):
    image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes.cpu()).astype("float")

    person_count = 0
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for index, row in px.iterrows():
        x1, y1, x2, y2, conf, cls_id = map(int, row)
        class_name = class_list[cls_id]
        if 'body' in class_name:
            person_count += 1
            cv2.rectangle(image_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_color, class_name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            mask[y1:y2, x1:x2] = 255

    return image_color, mask, person_count

def blur_background_func(image, mask, kernel_size, sigma):
    mask_inv = cv2.bitwise_not(mask)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    focused_area = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(blurred_image, blurred_image, mask=mask_inv)
    final_image = cv2.add(focused_area, background)
    return final_image

def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        download_model()
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error('Model not found, please configure if you wish to download model from URL set `CFG_ENABLE_URL_DOWNLOAD = True`', icon="⚠️")

    model = load_model()
    class_list = ['body']

    st.set_page_config(page_title="FocusAI: People Detection and Background Blur", page_icon="👤", layout="wide")

    states = ['FaceFleet', 'About Us']
    curr_state = states[0]

    # Sidebar for file upload and controls
    img = Image.open('logo.jpg').convert("RGB")
    st.sidebar.image(img, caption='', use_column_width=True)

    # Create two columns
    col1, col2 = st.sidebar.columns(2)
    if col1.button('&nbsp;&nbsp;&nbsp;&nbsp;FocusAI&nbsp;&nbsp;&nbsp;&nbsp;', key='FocusAI'):
        curr_state = states[0]
    if col2.button('&nbsp;&nbsp;&nbsp;&nbsp;About Us&nbsp;&nbsp;&nbsp;&nbsp;', key='About Us'):
        curr_state = states[1]

    st.sidebar.markdown('---')

    processed_image = None
    blurred_image = None

    if curr_state == states[0]:
        st.sidebar.title('Quick start:')
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"], label_visibility="collapsed")
        blur_background = st.sidebar.checkbox("Blur Background Without Boxes", value=False)
        kernel_size = st.sidebar.slider('Kernel size for blurring:', min_value=3, max_value=99, value=21, step=2)
        sigma = st.sidebar.slider('Sigma value for blurring:', min_value=0, max_value=100, value=10)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")  # Ensure the image is in RGB format
            image = np.array(image)
            cv2.imwrite('original_image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            u_col1, u_col2, u_col3 = st.columns(3)
            with u_col2:
                st.subheader("Uploaded Image:")
                st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.sidebar.button('Detect People'):
                processed_image, mask, person_count = detect_people(image.copy(), model)
                cv2.imwrite('processed_image.jpg', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                if blur_background:
                    blurred_image = blur_background_func(image.copy(), mask, kernel_size, sigma)
                    cv2.imwrite('blurred_image.jpg', cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))

                st.subheader("People Detection vs Focus AI" if blur_background else "Original vs People Detection")
                image_comparison(
                    img1='processed_image.jpg' if blur_background else 'original_image.jpg',
                    img2='blurred_image.jpg' if blur_background else 'processed_image.jpg',
                    label1="Detection" if blur_background else "Original",
                    label2="FocusAI" if blur_background else "Detection",
                    width=1200,
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True,
                )

    if curr_state == states[1]:
        st.title(curr_state)
        st.caption("This section is about us")

        team_col1, team_col2, team_col3 = st.columns(3)
        team_col1.image(Image.open('mihwa.jpeg').convert("RGB"), caption='Mihwa Kim', use_column_width=True)
        team_col2.image(Image.open('ahmad.jpeg').convert("RGB"), caption='Ahmad Faizi', use_column_width=True)
        team_col3.image(Image.open('dylan.jpeg').convert("RGB"), caption='Gia (Dylan) Bach Le', use_column_width=True)
        st.subheader("The objective")
        st.markdown("The objective is to enhance privacy and focus in visual content. By identifying and isolating individuals in the image, the model ensures that the viewer's attention is drawn primarily to the subjects. The blurring of the background reduces distractions, providing a cleaner, more professional appearance. This can be particularly useful in applications such as video conferencing, photography, and social media, where the focus is often on the individuals present and not the environment around them. Additionally, by blurring the background, the model can help protect sensitive or private information that might inadvertently be captured in the image.")

        st.subheader("Methodology")
        st.markdown("1. Dataset Preparation: The first step in the process was to prepare the dataset for training. In this case, the CrowdHuman dataset was used. This dataset is a large and rich dataset that contains images of people in various real-world scenarios, making it ideal for training an AI model to identify people in images.")
        st.markdown("2. Model Selection and Training: The YOLOv5 (You Only Look Once version 5) model was chosen for this task. YOLOv5 is a state-of-the-art, real-time object detection system that has been widely used in similar tasks due to its speed and accuracy. The model was trained on the prepared CrowdHuman dataset.")
        st.markdown("3. Model Evaluation: After training, the model was evaluated to ensure it was accurately identifying people in images and could effectively separate them from the background.")
        st.markdown("4. Application Development: Once the model was trained and evaluated, it was integrated into an application prototype. Streamlit, a fast and easy-to-use open-source framework for building machine learning and data science web applications, was used for this purpose.")
        st.markdown("5. Testing and Iteration: The prototype was then tested to ensure it was working as expected. Feedback from these tests was used to make any necessary adjustments to the model or application.")
        st.markdown("6. Deployment: Once testing was complete and any necessary adjustments were made, the application was ready for deployment.")

        st.subheader("Applications")
        st.markdown("Security Systems: In security applications, the model can be used to protect the identities of individuals in the background of surveillance footage, focusing only on the subjects of interest.")
        st.markdown("Augmented Reality (AR): In AR applications, this model can be used to separate real-world objects from their backgrounds, allowing for more immersive and realistic interactions with digital content.")
        st.markdown("Photography and Videography: The model can be used to automatically create a depth-of-field effect, or ‘bokeh’, which is a popular technique in photography and videography. This effect helps to draw attention to the subject by blurring the background.")

    if curr_state == states[0] and uploaded_file is None:
        a_c1, a_c2, a_c3 = st.columns(3)
        a_c2.subheader("Camera Feed")
        picture = a_c2.camera_input("Take a picture", label_visibility="collapsed")

        if picture is not None:
            file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
            picture = cv2.imdecode(file_bytes, 1)
            picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)  # Convert to RGB

            b_c1, b_c2 = st.columns(2)
            processed_image, mask, person_count = detect_people(picture.copy(), model)
            cv2.imwrite('processed_image.jpg', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            blurred_image = blur_background_func(picture.copy(), mask, kernel_size, sigma)
            cv2.imwrite('blurred_image.jpg', cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
            image_comparison(
                img1='processed_image.jpg',
                img2='blurred_image.jpg',
                label1="Detection",
                label2="FocusAI",
                width=1200,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )

if __name__ == '__main__':
    main()
