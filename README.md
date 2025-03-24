# Handwritten Number Recognition System
![GitHub Repo stars](https://img.shields.io/github/stars/Uni-Creator/Handwritten_Number_Recognition_System?style=social)  ![GitHub forks](https://img.shields.io/github/forks/Uni-Creator/Handwritten_Number_Recognition_System?style=social)

## 📌 Overview
The **Handwritten Number Recognition System** is a deep learning model built from scratch to classify handwritten digits using a custom neural network. It is trained on the MNIST dataset and utilizes NumPy-based implementation for forward and backward propagation.

## 🚀 Features
- **Custom-built Neural Network**: Implemented using NumPy without deep learning frameworks like TensorFlow or PyTorch.
- **Forward & Backward Propagation**: Includes ReLU and Tanh activation functions for optimization.
- **Model Training & Evaluation**: Supports training on both CPU and GPU.
- **Manual & Automated Testing**: Test the model using predefined test images or random inputs.
- **Performance Metrics**: Displays accuracy and prediction confidence.

## 🏗️ Tech Stack
- **Python**
- **NumPy** (for matrix computations)
- **Matplotlib** (for visualization)
- **Pandas** (for data handling)
- **GPU Support** (via CUDA for optimized training)

## 📂 Project Structure
```
Handwritten_Number_Recognition_System/
│── data/                     # Dataset storage
│── main.py                   # Loads trained model and tests data
│── model.npz                 # Saved model parameters
│── trainer.py                # Trains the neural network model
│── trainOnGPU.py             # Optimized training for GPU acceleration
│── README.md                 # Project documentation
```

## 📦 Installation & Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/Uni-Creator/Handwritten_Number_Recognition_System.git
   cd Handwritten_Number_Recognition_System
   ```
2. **Install dependencies**
   ```sh
   pip install numpy pandas matplotlib
   ```
3. **Train the model (if needed)**
   ```sh
   python trainer.py
   ```
4. **Run the model for testing**
   ```sh
   python main.py
   ```

## 📊 How It Works
1. The model loads pre-trained weights from `model.npz` or `model.pth`.
2. A test image is provided for prediction.
3. The model outputs a digit classification with confidence score.
4. The prediction is displayed along with the corresponding test image.

## 🛠️ Future Improvements
- Implement CNN-based architecture for improved accuracy.
- Add a web interface for user-uploaded handwritten digit classification.
- Support for different datasets beyond MNIST.

## 🤝 Contributing
Contributions are welcome! Feel free to open an **issue** or submit a **pull request**.

## 📄 License
This project is licensed under the **Apache-2.0 license**.

---

