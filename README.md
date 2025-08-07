#  Chest X-Ray Pneumonia Classification using CNN

This project uses deep learning to classify chest X-ray images as **NORMAL** or **PNEUMONIA**. It is based on the publicly available dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).


##Dataset

- The dataset contains chest X-ray images categorized into:
  - **NORMAL**
  - **PNEUMONIA**
- The data is split into `train`, `val`, and `test` folders.


## Project Goals

- Load and preprocess X-ray images.
- Explore class distribution and visualize samples.
- Build a Convolutional Neural Network (CNN) for binary classification.
- Evaluate model performance using various metrics.
- Apply data augmentation to reduce overfitting.
- (Optional) Experiment with transfer learning using pre-trained models like **VGG16**, **ResNet**, or **MobileNet**.


## Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn


## 🧪 Model Architecture

```text
Conv2D → ReLU → MaxPooling
Conv2D → ReLU → MaxPooling
Conv2D → ReLU → MaxPooling
Flatten → Dropout → Dense → Output (Sigmoid)
