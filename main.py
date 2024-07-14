import streamlit as st
import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Define the path to the users.csv file
USERS_FILE = 'users.csv'

# Function to create the users.csv file if it does not exist or is empty
def create_users_file():
    if not os.path.exists(USERS_FILE) or os.stat(USERS_FILE).st_size == 0:
        df = pd.DataFrame(columns=['username', 'password'])
        df.to_csv(USERS_FILE, index=False)

# Define functions for loading and processing images
def load_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def anomaly_detection(img_array):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Load pre-trained weights if available
    # model.load_weights('path_to_pretrained_weights.h5')
    
    prediction = model.predict(img_array)
    anomaly_score = prediction[0][0]
    return anomaly_score

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    hashed_password = hash_password(password)
    create_users_file()
    user_data = pd.read_csv(USERS_FILE)
    if username in user_data['username'].values:
        return False, "Username already exists."
    new_user = pd.DataFrame([[username, hashed_password]], columns=['username', 'password'])
    user_data = pd.concat([user_data, new_user], ignore_index=True)
    user_data.to_csv(USERS_FILE, index=False)
    return True, "Registration successful."

def login_user(username, password):
    hashed_password = hash_password(password)
    create_users_file()
    user_data = pd.read_csv(USERS_FILE)
    if username in user_data['username'].values:
        stored_password = user_data[user_data['username'] == username]['password'].values[0]
        if stored_password == hashed_password:
            return True, "Login successful."
        else:
            return False, "Incorrect password."
    else:
        return False, "Username not found."

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame
        return frame

def main():
    st.title('Anomaly Detection in Medical Images')
    
    menu = ["Home", "Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ''
    
    if choice == "Home":
        if st.session_state.logged_in:
            st.subheader(f"Welcome, {st.session_state.username}!")
            upload_method = st.radio("Choose upload method", ("Upload Image", "Capture from Camera"))
            
            if upload_method == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
                
                if uploaded_file is not None:
                    # Perform anomaly detection
                    img_path = './uploaded_image.jpg'  # Temporarily save uploaded image
                    with open(img_path, 'wb') as f:
                        f.write(uploaded_file.read())
                    
                    img, img_array = load_image(img_path)
                    anomaly_score = anomaly_detection(img_array)
                    
                    # Determine anomaly detection result
                    threshold = 0.5
                    if anomaly_score > threshold:
                        st.error("Anomaly Detected!")
                    else:
                        st.success("No Anomaly Detected.")
                    
                    # Display the uploaded image
                    st.image(img, caption='Uploaded Image.', use_column_width=True)
            
            elif upload_method == "Capture from Camera":
                ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
                if ctx.video_transformer:
                    if st.button("Capture"):
                        frame = ctx.video_transformer.frame.to_ndarray(format="bgr24")
                        if frame is not None:
                            img_path = './captured_image.jpg'
                            plt.imsave(img_path, frame)
                            
                            img, img_array = load_image(img_path)
                            anomaly_score = anomaly_detection(img_array)
                            
                            # Determine anomaly detection result
                            threshold = 0.5
                            if anomaly_score > threshold:
                                st.error("Anomaly Detected!")
                            else:
                                st.success("No Anomaly Detected.")
                            
                            # Display the captured image
                            st.image(frame, caption='Captured Image.', use_column_width=True)
        
        else:
            st.warning("Please log in to use the application.")
    
    elif choice == "Login":
        st.subheader("Login")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            success, msg = login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(msg)
            else:
                st.error(msg)
    
    elif choice == "Register":
        st.subheader("Register")
        
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type='password')
        if st.button("Register"):
            success, msg = register_user(new_username, new_password)
            if success:
                st.success(msg)
            else:
                st.error(msg)

if __name__ == '__main__':
    main()
