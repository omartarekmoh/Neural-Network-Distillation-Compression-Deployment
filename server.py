from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import time
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from models.models import ModelLoader  # Import ModelLoader

app = Flask(__name__)

# Define class labels for CIFAR-10 dataset
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize ModelLoader
model_loader = ModelLoader()

# Load Teacher (ResNet50) and Student (ResNet18) models (PyTorch)
teacher_model = model_loader.load_pytorch_model("teacher", "models/best_resnet50_model.pth")
student_model = model_loader.load_pytorch_model("student", "models/best_student.pth")

# Load TFLite model
tflite_interpreter = model_loader.load_tflite_model("models/best_resnet18_model_light.tflite")

# Set PyTorch models to evaluation mode
teacher_model.eval()
student_model.eval()

# ImageNet Normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Image transformations for test images
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def predict(model, image):
    """Make a prediction using a PyTorch model."""
    image = transform_test(image).unsqueeze(0)  # Transform and add batch dimension

    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
    inference_time = time.time() - start_time

    _, predicted = torch.max(outputs, 1)
    return CIFAR10_CLASSES[predicted.item()], inference_time


@app.route('/')
def index():
    return render_template('index.html')


def get_multiple_predictions(model, image, num_predictions=5):
    """Runs multiple predictions to calculate average inference time for PyTorch models."""
    softmax_outputs = []
    times = []

    for _ in range(num_predictions):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        inference_time = time.time() - start_time
        times.append(inference_time)

    softmax_output = F.softmax(outputs, dim=1)
    softmax_outputs.append(softmax_output.cpu().numpy())

    avg_time = sum(times) / len(times)
    return softmax_outputs, avg_time


def predict_tflite(interpreter, image):
    """Make an inference using the TensorFlow Lite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']  # Example: (1, 224, 224, 3) or (1, 3, 224, 224)
    input_dtype = input_details[0]['dtype']  # Expected data type

    # Apply PyTorch Transformations
    img_tensor = transform_test(image)  # Shape: (3, 224, 224)

    # Convert to NumPy and remove batch dimension
    img_array = img_tensor.numpy()

    # Handle different input formats (NCHW vs NHWC)
    if input_shape[-1] == 3:  # Model expects (1, 224, 224, 3) -> Convert from (3, 224, 224)
        img_array = np.transpose(img_array, (1, 2, 0))  # Convert from CHW to HWC
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    else:  # Model expects (1, 3, 224, 224), so just add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

    # Ensure dtype matches the model
    img_array = img_array.astype(input_dtype)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Get model output
    output_data = interpreter.get_tensor(output_details[0]['index']).copy()

    softmax_output = F.softmax(torch.tensor(output_data), dim=1).numpy().flatten().tolist()

    print(softmax_output)

    # Get predicted class
    predicted_class = np.argmax(softmax_output)

    return CIFAR10_CLASSES[predicted_class], inference_time, softmax_output


@app.route('/predict', methods=['POST'])
def predict_route():
    """API endpoint for model inference."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file).convert('RGB')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Transform image to tensor
    image_tensor = transform_test(image).unsqueeze(0)

    # Get predictions for PyTorch models
    finetuned_softmax, finetuned_time = get_multiple_predictions(teacher_model, image_tensor)
    distilled_softmax, distilled_time = get_multiple_predictions(student_model, image_tensor)

    # Convert softmax outputs to lists
    finetuned_softmax = [softmax[0].tolist() for softmax in finetuned_softmax]
    distilled_softmax = [softmax[0].tolist() for softmax in distilled_softmax]

    # Get prediction from TensorFlow Lite model
    tflite_pred, tflite_time, tflite_softmax = predict_tflite(tflite_interpreter, image)

    # Prepare JSON response
    return jsonify({
        "prediction_finetuned": CIFAR10_CLASSES[np.argmax(finetuned_softmax[0])],
        "finetuned_percentages": finetuned_softmax[0],
        "finetuned_avg_time": finetuned_time,

        "prediction_distilled": CIFAR10_CLASSES[np.argmax(distilled_softmax[0])],
        "distilled_percentages": distilled_softmax[0],
        "distilled_avg_time": distilled_time,

        "prediction_tflite": tflite_pred,
        "tflite_percentages": tflite_softmax,  # List of softmax values
        "tflite_avg_time": tflite_time
    })


if __name__ == '__main__':
    app.run(debug=True)
