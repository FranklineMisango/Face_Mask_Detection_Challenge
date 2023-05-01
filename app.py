import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.models import load_model

st.title("Face Mask Detector")
image = Image.open('images/cover.png')
st.image(image, caption='Mask detector')

model = load_model('Mask_detection.h5', compile=False)
model.build((None, 224, 224, 3))

def predict_mask(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = np.expand_dims(frame, axis=0)
    prediction = model.predict(frame)
    # Use the pre-trained model to classify the image
    prediction = model.predict(frame)
    prediction = prediction.flatten()
    
    # Return the prediction result
    if prediction[0] > 0.5:
        return "Mask"
    else:
        return "No Mask"

# Define the webcam_mask_detection function
def webcam_mask_detection():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        
        # Apply the mask detection algorithm
        result = predict_mask(frame)
        
        # Draw the result on the frame and display it
        cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Mask Detection', frame)
        
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release the webcam and destroy the window
    cap.release()
    cv2.destroyAllWindows()

# Define the app behavior
def app():
    # Define the app text
    st.write("Choose an option to detect whether a person is wearing a mask or not.")

    # Allow the user to choose between uploading an image or using their webcam
    option = st.radio("Select an option:", ("Upload an image", "Use Webcam"))

    if option == "Upload an image":
        # Allow the user to upload an image
        image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

        if image_file is not None:
            # Load the image and preprocess it
            image = Image.open(image_file)
            image = image.resize((128, 128))
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)

            # Use the pre-trained model to classify the image
            prediction = model.predict(image_array)
            prediction = prediction.flatten()

            # Display the prediction result
            if prediction[0] > 0.5:
                st.write("The person is wearing a mask.")
            else:
                st.write("The person is not wearing a mask.")

    elif option == "Use Webcam":
        # Display a button to start the webcam mask detection
        if st.button("Start Webcam"):
            webcam_mask_detection()

# Run the app
if __name__ == '__main__':
    app()
