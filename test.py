import tensorflow as tf
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import io

# Initialize Azure Blob Storage connection
connection_string = "add_string"
container_name = "dlmisangobeta"
blob_name = "Mask_detection.h5"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
blob_client = container_client.get_blob_client(blob_name)

# Load the model from Azure Blob Storage
model_bytes = blob_client.download_blob().readall()
model = tf.keras.models.load_model(io.BytesIO(model_bytes))

# Test if the model is loaded successfully
print("Model loaded successfully from Azure Blob Storage.")
