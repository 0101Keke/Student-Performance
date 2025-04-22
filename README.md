# 🎓 Student Grade Prediction Web App

This is a full-stack machine learning application that predicts a student's **grade class** using both traditional and deep learning models. The web app is built with **Dash**, and models are trained using **scikit-learn** and **TensorFlow/Keras**.

## 📊 Features

- ✅ Input form to collect student data
- 🔄 Option to choose between:
  - Random Forest Classifier
  - Feedforward Neural Network (Keras)
- 📈 Visual output of prediction probabilities for neural network
- 🧠 Deep learning component (TensorFlow-based model) integrated into the backend
- 🌐 Deployed using Render

## 💡 Tech Stack

| Component         | Technology                |
|------------------|---------------------------|
| Backend ML Model | scikit-learn, TensorFlow  |
| Frontend UI      | Dash, Dash Bootstrap      |
| Deployment       | Render.com                |
| Data Scaling     | StandardScaler (sklearn)  |


## 🧠 Deep Learning (Step 8 - Requirement)

- A **Feedforward Neural Network (FNN)** was built in a separate Jupyter Notebook.
- Trained using Keras and saved as `feedforward_nn_model.keras`.
- It was integrated into the Dash web app using `tf.keras.models.load_model`.
- The neural network model provides both:
  - Predicted grade class
  - A bar chart visualization of prediction probabilities

This satisfies **Step 8: Apply a deep learning algorithm.**

**Dash Application:** https://student-performance-9.onrender.com
