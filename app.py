# Import necessary packages from TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load your Keras model
loaded_model = load_model('ashwin.h5')

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define route for uploading images
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        
        image_file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if image_file.filename == '':
            return 'No selected file'
        
        try:
            # Use PIL to open the image and convert it to RGB format
            img = Image.open(image_file).convert('RGB')
            img = img.resize((28, 28))  # Resize the image to the target size

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Rescale to values between 0 and 1

            # Make predictions
            predictions = loaded_model.predict(img_array)

            # Get the class label (assuming binary classification)
            class_label = "No Tumor Found" if predictions[0][0] > 0.5 else "Tumor Found"

            # Print the predictions
            print("Predictions:", predictions)
            print("Class Label:", class_label)
            return render_template('upload.html', prediction=class_label) 
        except Exception as e:
            return f'Error processing image: {e}'
    return render_template('upload.html')  # Render the upload page if method is GET

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("Home.html")

# Define route for about us page
@app.route('/aboutus', methods=['GET', 'POST'])
def about():
    return render_template("AboutUs.html")

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
