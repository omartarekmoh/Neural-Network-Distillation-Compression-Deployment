import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import tensorflow as tf
import numpy as np


class ModelLoader:
    def __init__(self, device=None):
        """
        Initializes the model loader.
        :param device: Device for PyTorch models (default is auto-detect)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pytorch_model(self, model_type, checkpoint_path, num_classes=10):
        """
        Load a PyTorch model from a checkpoint.
        :param model_type: "teacher" for ResNet50, "student" for ResNet18
        :param checkpoint_path: Path to the model checkpoint file
        :param num_classes: Number of output classes
        :return: Loaded PyTorch model
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if model_type == "teacher":
            model = CustomResNet(10,
                     checkpoint["hyperparameters"]["intermediate_dim"],
                     checkpoint["hyperparameters"]["dropout1_rate"],
                     checkpoint["hyperparameters"]["dropout2_rate"]
                     )
            
            optimizer = torch.optim.Adam(model.parameters(), checkpoint["hyperparameters"]['lr'])
        elif model_type == "student":
            model = StudentResNet(10,
                     checkpoint["hyperparameters"]["intermediate_dim"],
                     checkpoint["hyperparameters"]["dropout1_rate"],
                     checkpoint["hyperparameters"]["dropout2_rate"]
                     )
            optimizer = torch.optim.SGD(model.parameters(), checkpoint["hyperparameters"]['lr'])
        else:
            raise ValueError("Invalid model type. Use 'teacher' or 'student'.")

        # Load hyperparameters
        model = model.to(self.device)

        # Load state dictionaries
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']

        print(f"{model_type.capitalize()} model loaded from epoch {epoch} with best validation accuracy: {best_val_acc}")

        return model

    def load_tflite_model(self, tflite_path):
        """
        Load a TensorFlow Lite model.
        :param tflite_path: Path to the TFLite model
        :return: Loaded TFLite interpreter
        """
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        print("TFLite model loaded successfully.")
        print("Model Input Details:", interpreter.get_input_details())
        print("Model Output Details:", interpreter.get_output_details())

        return interpreter

    def run_tflite_inference(self, interpreter, input_data):
        """
        Runs inference using a TensorFlow Lite model.
        :param interpreter: Loaded TFLite interpreter
        :param input_data: Input data matching the model's expected input shape
        :return: Model output
        """
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Ensure input is in the correct shape and type
        input_tensor_index = input_details[0]['index']
        interpreter.set_tensor(input_tensor_index, input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_tensor_index = output_details[0]['index']
        output_data = interpreter.get_tensor(output_tensor_index)

        return output_data


class CustomResNet(nn.Module):
    def __init__(self, num_classes, intermediate_dim, dropout1_rate, dropout2_rate):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, intermediate_dim),
            nn.Dropout(p=dropout1_rate),
            nn.ReLU(),
            nn.Dropout(p=dropout2_rate),
            nn.Linear(intermediate_dim, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

class StudentResNet(nn.Module):
    def __init__(self, num_classes, intermediate_dim, dropout1_rate, dropout2_rate):
        super(StudentResNet, self).__init__()
        # Using ResNet18 as student (smaller than ResNet50)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features

        # Same architecture as teacher for the final layers
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, intermediate_dim),
            nn.Dropout(p=dropout1_rate),
            nn.ReLU(),
            nn.Dropout(p=dropout2_rate),
            nn.Linear(intermediate_dim, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
