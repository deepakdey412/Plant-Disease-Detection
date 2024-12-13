# Plant Disease Prediction Mini Project 

## About the Project
Plant diseases can significantly impact agricultural productivity, leading to major economic losses and food insecurity. This mini project aims to address the issue by developing a **Plant Disease Prediction system** **[Detect diseases in Apple, Potato, Tulsi, Tomato, and Rose plants]**. It leverages advanced machine learning and deep learning techniques to identify plant diseases from images, providing farmers and researchers with an accurate and efficient solution for disease detection.

## What the Project Does
This project uses images of plant leaves to predict whether a plant is healthy or affected by a disease. The primary functionalities include:
- Detecting diseases in plants from images.
- Classifying the specific type of disease.
- Providing accurate predictions with the help of trained models.

## Motivation
The project was created as part of a **college mini-project** to demonstrate the practical application of machine learning in solving real-world problems. Agriculture is a crucial sector, and early detection of plant diseases can prevent widespread damage, saving time, money, and resources. This project is a step toward automating the process of disease detection and contributing to smarter agricultural practices.

## Technologies Used
The following technologies, libraries, and frameworks were used in the project:
- **Programming Language**: Python
- **Frontend:** Streamlit (for creating an interactive user interface)
- **Frameworks**: TensorFlow, Keras
- **Libraries**: OpenCV, NumPy, Pandas, Matplotlib, scikit-learn)
- **Other Tools**: Jupyter Notebook (for development and experimentation)

## Models Implemented
- **Convolutional Neural Networks (CNNs)**: Trained for image classification and disease prediction.
- Multiple models were trained and evaluated, optimized for accuracy and performance.
- Transfer learning techniques (e.g., using pre-trained models like ResNet or VGG) were applied to enhance performance on the dataset.

## How It Works Internally
1. **Dataset Preparation**:
   - The dataset consists of categorized folders containing images of healthy and diseased plants.
   - Images are preprocessed (resizing, normalization, and augmentation) to improve model performance.

2. **Model Training**:
   - A CNN model was built and trained on the preprocessed dataset.
   - The dataset was split into training, validation, and test sets.
   - Loss functions and optimizers were applied to enhance the model’s learning.

3. **Prediction Mechanism**:
   - The trained model accepts a new image as input.
   - The model processes the image and predicts the likelihood of diseases based on its learned parameters.

## How to Run the Project
Follow these steps to run the project on another device:

### Prerequisites
1. Install Python 3.8+.
2. Install the following libraries using pip:
   ```bash
   pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn flask
   ```
3. Ensure the dataset is available in the required format (organized in folders for training and testing).

### Running the Project
1. Clone the project repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd plant-disease-prediction
   ```
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. Access the application in your browser at `http://127.0.0.1:5000`.

## Customization Guide
You can adapt the project for custom datasets by following these steps:

### Dataset Requirements
- Organize the dataset in the following structure:
  ```plaintext
  dataset/
    train/
      category_1/
      category_2/
      ...
    valid/
      category_1/
      category_2/
      ...
    test/
      category_1/
      category_2/
      ...
  ```
- Each category should contain images relevant to that class.

### Steps for Customization
1. **Prepare the Dataset**:
   - Ensure the images are labeled correctly and preprocessed (resizing, etc.).

2. **Modify the Model**:
   - Adjust the model architecture in the script to suit your dataset.
   - Change the number of output neurons to match the number of categories in your dataset.

3. **Retrain the Model**:
   - Run the training script to train the model on your dataset.
   - Evaluate and save the trained model for future use.

4. **Integrate the Model**:
   - Replace the old model file in the Flask application with your trained model.

5. **Test the Application**:
   - Start the Flask server and test predictions with images from your dataset.

---

This README should provide clear guidance for users to understand, use, and modify the Plant Disease Prediction project for their purposes. 😊


