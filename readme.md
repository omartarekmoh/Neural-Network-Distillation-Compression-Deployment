# 🧠 Multi-Stage Model Compression & Deployment

## 🚀 Overview
This repository implements a **multi-stage model compression pipeline** for deep learning models. It includes:
1. **Big Model (Teacher)** – A high-performance, large-scale deep learning model.
2. **Distilled Model (Student)** – A smaller, optimized version of the teacher model.
3. **TFLite Model** – A further compressed version of the student model for mobile and edge deployment.

This pipeline demonstrates **knowledge distillation**, **model compression**, and **TensorFlow Lite conversion** for scalable AI applications.

---

## 📂 Project Structure

```
📦 Multi-Stage Model Compression
├── models/  # Pre-trained and distilled models
│   ├── best_resnet50_model.pth  # Teacher Model (ResNet50)
│   ├── best_student.pth  # Distilled Model (ResNet18)
│   ├── best_resnet18_model_light.tflite  # TFLite Model
├── server.py  # Flask API for model inference
├── frontend/  # Web interface for testing
│   ├── index.html  # Web UI
│   ├── static/
│   │   ├── style.css  # CSS Styling
│   │   ├── script.js  # JavaScript for UI updates
├── data/  # Data directory (ignored in .gitignore)
├── requirements.txt  # Python dependencies
├── README.md  # Documentation (this file)
└── .gitignore  # Git ignore rules
```

---

## 🎯 Features
- ✅ **Knowledge Distillation** – Compresses a large model into a lightweight student model.
- ✅ **TFLite Conversion** – Deploys the model efficiently on mobile and edge devices.
- ✅ **Flask API** – A REST API for serving predictions from all models.
- ✅ **Web Interface** – An interactive UI for testing models.
- ✅ **Class Probability Visualization** – Displays softmax class probabilities using Chart.js.

---

## 🛠️ Setup & Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scriptsctivate      # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🔥 Model Training & Knowledge Distillation

The teacher model (ResNet50) is trained on CIFAR-10, and a distilled student model (ResNet18) is obtained using knowledge distillation.

### Train the Teacher Model
```bash
python train_teacher.py --epochs 50 --batch_size 64 --lr 0.001
```

### Train the Student Model with Distillation
```bash
python train_student.py --epochs 50 --batch_size 64 --lr 0.001 --temperature 4.0
```

### Convert Student Model to TFLite
```bash
python convert_tflite.py --model best_student.pth --output best_resnet18_model_light.tflite
```

---

## 🌐 Running the Flask API

Start the backend server to serve model predictions via an API.

```bash
python server.py
```

The API will be available at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Endpoints:
- **POST** `/predict` – Accepts an image and returns predictions.

---

## 🖥️ Web UI for Model Testing

A web-based UI allows users to upload images and view predictions and class distributions.

### **1️⃣ Start the Web Server**
```bash
cd frontend
python -m http.server 8080
```

### **2️⃣ Open in Browser**
Go to: [http://127.0.0.1:8080](http://127.0.0.1:8080)

---

## 📡 API Endpoints

### 📌 Predict Image Class

#### 🔹 Request
```http
POST /predict
Content-Type: multipart/form-data
```

| Parameter | Type  | Description          |
|-----------|-------|----------------------|
| file      | image | The input image file |

#### 🔹 Response
```json
{
    "prediction_finetuned": "dog",
    "finetuned_percentages": [0.1, 0.05, 0.03, 0.02, 0.7, 0.04, 0.02, 0.02, 0.01, 0.01],
    "finetuned_avg_time": 0.0123,
    
    "prediction_distilled": "dog",
    "distilled_percentages": [0.12, 0.03, 0.04, 0.02, 0.65, 0.06, 0.03, 0.02, 0.02, 0.01],
    "distilled_avg_time": 0.0087,

    "prediction_tflite": "dog",
    "tflite_percentages": [0.09, 0.06, 0.02, 0.03, 0.75, 0.02, 0.01, 0.01, 0.005, 0.005],
    "tflite_avg_time": 0.0054
}
```

---

## 🏗️ Deployment Options

### **1️⃣ Deploy with Docker**
```bash
docker build -t model-compression-api .
docker run -p 5000:5000 model-compression-api
```

### **2️⃣ Deploy to Cloud (e.g., AWS, GCP, Heroku)**
- AWS Lambda + API Gateway
- Google Cloud Run
- Heroku with Gunicorn

---

## 🛠️ Technologies Used
- **PyTorch** – Model Training & Knowledge Distillation
- **TensorFlow Lite** – Model Conversion for Edge Deployment
- **Flask** – API for Serving Predictions
- **Chart.js** – Data Visualization
- **HTML, CSS, JavaScript** – Web UI

---

## 🤝 Contributing
Contributions are welcome! Follow these steps:

1. Fork the repo
2. Create a feature branch
3. Commit changes
4. Push and open a pull request

---

## ⚖️ License
This project is licensed under the **MIT License**.

---

## 📝 Acknowledgments
- **PyTorch & TensorFlow** for model training and deployment tools.
- **Flask & Chart.js** for API and UI components.

---

## 🚀 Contact & Links
📩 **Email:** your.email@example.com  
🔗 **GitHub Repo:** Your Repo Link  
🌍 **Live Demo (if deployed):** Live Link
