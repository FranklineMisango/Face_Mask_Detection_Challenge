import cv2
import h5py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from keras.models import load_model

st.title("Face Mask Detector")
image = Image.open('images/cover.png')
st.image(image, caption='Mask detector')


#model = load_model('Mask_detection.h5', compile=False)  Testing the azure blob storage function and uncomment for remote testing 

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Initialize Azure Blob Storage connection
connection_string = "add string"
container_name = "dlmisangobeta"
blob_name = "Mask_detection.h5"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
blob_client = container_client.get_blob_client(blob_name)

# Load the model from Azure Blob Storage
model_bytes = blob_client.download_blob().readall()
#model = load_model(io.BytesIO(model_bytes), compile=False)
model = tf.keras.models.load_model(io.BytesIO(model_bytes))
model.build((None, 224, 224, 3))

# Use the model in your Streamlit app
st.write("Model loaded successfully from Azure Blob Storage")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def predict_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face = frame[y:y+h, x:x+w]
            # Resize the face ROI to 128x128
            face = cv2.resize(face, (128, 128))
            # Preprocess the face ROI
            face = np.expand_dims(face, axis=0)
            # Use the pre-trained model to classify the face ROI
            prediction = model.predict(face)
            prediction = prediction.flatten()
            # Draw a rectangle around the face
            if prediction[0] > 0.5:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "No Mask", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Mask", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Return the frame with the face detection and classification results
    return frame



# Define the webcam_mask_detection function
def webcam_mask_detection():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        
        # Apply the mask detection algorithm
        result = predict_mask(frame)
        
        # Display the result on the frame and show it
        cv2.imshow('Mask Detection', result)
        
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release the webcam and destroy the window
    cap.release()
    cv2.destroyAllWindows()

# Define the video_mask_detection function
def video_mask_detection(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        # Capture a frame from the video
        ret, frame = cap.read()

        if ret:
            # Apply the mask detection algorithm
            result = predict_mask(frame)

            # Write the result to the output video
            out.write(result)

            # Display the result on the frame and show it
            cv2.imshow('Mask Detection', result)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    # Release the video and destroy the window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Define the app behavior
def app():
    # Define the app text
    st.write("Choose an option to detect whether a person is wearing a mask or not.")

    # Allow the user to choose between uploading an image, using their webcam, or uploading a video
    option = st.radio("Select an option:", ("Upload an image", "Use Webcam", "Upload a video"))

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
            if prediction[0] < 0.5:
                st.write("The person is wearing a mask.")
            else:
                st.write("The person is not wearing a mask.")

    elif option == "Use Webcam":
        # Display a button to start the webcam mask detection
        if st.button("Start Webcam"):
            webcam_mask_detection()

    elif option == "Upload a video":
        # Allow the user to upload a video
        video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

        if video_file is not None:
            # Save the video to disk
            with open('uploaded_video.mp4', 'wb') as f:
                f.write(video_file.read())

            # Call the video_mask_detection function
            video_mask_detection('uploaded_video.mp4')

# Run the app
if __name__ == '__main__':
    app()
