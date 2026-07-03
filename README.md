# Brain-tumor-detection

A deep learning-based brain tumor detection system that classifies MRI brain scans into **Tumor** and **No Tumor** categories using **EfficientNetB0** with transfer learning. The model leverages TensorFlow and Keras to achieve high classification accuracy while reducing training time through pre-trained ImageNet weights.


##  Overview

Brain tumors are one of the most critical neurological disorders that require early diagnosis for effective treatment. Manual analysis of MRI scans is time-consuming and depends on expert radiologists. This project automates brain tumor detection using a Convolutional Neural Network (CNN) based on **EfficientNetB0**, providing fast and reliable predictions from MRI images.

The project includes:

- Automatic dataset download from Kaggle
- Image preprocessing and augmentation
- Transfer learning using EfficientNetB0
- Fine-tuning for improved performance
- Model evaluation using multiple metrics
- Single image prediction
- Saving trained models in both `.keras` and `.h5` formats


##  Features

- MRI image classification (Tumor / No Tumor)
- Automatic dataset download using KaggleHub
- Transfer Learning with EfficientNetB0
- Data augmentation to improve generalization
- Early stopping and learning rate scheduling
- Model checkpointing
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - Classification Report
  - Confusion Matrix
- Predicts individual MRI images
- Saves trained models for future use

---

##  Dataset

**Dataset:** Brain MRI Images for Brain Tumor Detection

The dataset contains two classes:

- **yes** → Brain Tumor
- **no** → No Brain Tumor

The dataset is automatically downloaded using KaggleHub.

Dataset Structure:

```
brain_tumor_dataset/
│
├── yes/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── no/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```
##  Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Scikit-learn
- KaggleHub

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
```

Navigate to the project directory:

```bash
cd brain-tumor-detection
```

Install the required packages:

```bash
pip install tensorflow numpy scikit-learn kagglehub
```

---

##  Running the Project

Run the Python script:

```bash
python main.py
```

The script will automatically:

- Download the dataset
- Prepare training, validation, and testing datasets
- Train the EfficientNetB0 model
- Fine-tune the network
- Evaluate model performance
- Save trained models
- Predict a sample MRI image

---

##  Model Architecture

The project uses **EfficientNetB0** as the feature extractor.

Architecture:

```
Input Image (224 × 224 × 3)
            │
Data Augmentation
            │
EfficientNetB0 (ImageNet Weights)
            │
Global Average Pooling
            │
Dropout (0.35)
            │
Dense (128, ReLU)
            │
Dropout (0.25)
            │
Dense (1, Sigmoid)
            │
Prediction


##  Evaluation Metrics

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- Classification Report
- Confusion Matrix

Example output:

```
Final Test Results

Accuracy : XX.XX %

Precision: XX.XX %

Recall   : XX.XX %
```

---

##  Single Image Prediction

To predict an MRI image:

```python
predict_single_image("sample_mri.jpg")
```

Example Output:

```
==============================
Single Image Prediction Result
==============================
Image: sample_mri.jpg
Result: Tumor Detected
Confidence: 98.74%
==============================
```

---


##  Project Structure

```
Brain-Tumor-Detection/
│
├── main.py
├── sample_mri.jpg
├── brain_tumor_detection_model.h5
├── brain_tumor_detection_model.keras
├── best_brain_tumor_model.keras
├── README.md
└── requirements.txt
```
 Future Improvements

- Multi-class brain tumor classification
- Tumor segmentation using U-Net
- Web application using Flask or Streamlit
- Explainable AI (Grad-CAM)
- Support for DICOM medical images
- Hyperparameter optimization

---

##  Contributing

Contributions are welcome. Feel free to fork this repository, create a feature branch, and submit a pull request.

---

##  License

This project is developed for educational and research purposes.



