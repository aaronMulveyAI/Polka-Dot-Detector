# Polka Dot Detector: README

This README provides a detailed explanation of the **Polka Dot Detector** data science project. The project implements a basic image classification system for detecting **dysplastic nevi** and **spitz nevus** skin lesions. The project is implemented using Python, leveraging `Tkinter`, `NumPy`, `OpenCV`, `Matplotlib`, and basic linear algebra for machine learning modeling.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Workflow](#workflow)
   - [Data Collection](#data-collection)
   - [Feature Extraction](#feature-extraction)
   - [Visualization](#visualization)
   - [Model Training](#model-training)
   - [Prediction](#prediction)
3. [Code Structure](#code-structure)
4. [How to Run the Project](#how-to-run-the-project)
5. [Dependencies](#dependencies)
6. [Screenshots](#screenshots)

---

## Project Overview

The Polka Dot Detector project aims to classify images of skin lesions into two categories:
1. **Dysplastic Nevi**
2. **Spitz Nevus**

This is achieved by extracting RGB features from the lesion images, visualizing the feature space, calculating the loss function, and determining the optimal hyperplane using a simple linear model.

The system provides:
- A visualization of the feature space
- A loss function surface
- An optimal hyperplane for classification
- A prediction interface for unseen images

---

## Workflow

### 1. Data Collection

The dataset consists of images of two types of skin lesions:
- **Dysplastic Nevi** (training and testing images)
- **Spitz Nevus** (training and testing images)

Images are loaded from the following folder structure:
```
Polka Dot Detector/data/
    datasetLunares/
        dysplasticNevi/
            train/
            test/
        spitzNevus/
            train/
            test/
```

Each image is read using `OpenCV` and processed for feature extraction.

---

### 2. Feature Extraction

For each image, the **RGB features** are extracted using a mask applied to grayscale images. The process is as follows:
1. Convert the image to grayscale.
2. Apply the Otsu threshold to generate a binary mask.
3. Use the mask to compute the mean of the **R**, **G**, and **B** channels.


Code (located in `FeatureExtraction.py`):
```python
class FeatureExtraction:
    @staticmethod
    def get_features(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        mask = np.uint8(1 * (gray < threshold))

        B = (1 / 255) * np.sum(img[:, :, 0] * mask) / np.sum(mask)
        G = (1 / 255) * np.sum(img[:, :, 1] * mask) / np.sum(mask)
        R = (1 / 255) * np.sum(img[:, :, 2] * mask) / np.sum(mask)
        return [B, G, R]
```

---

### 3. Visualization

The system generates three key visualizations:

#### a) **Feature Space**
- A 3D scatter plot of the RGB features of all images.
- Red points represent **Spitz Nevus**; black points represent **Dysplastic Nevi**.

![Feature Space](https://github.com/aaronMulveyAI/Polka-Dot-Detector/blob/main/images/features.gif?raw=true)

#### b) **Loss Function**
- A 3D surface plot showing a quadratic loss function based on two weights (`w1`, `w2`).



![Loss Function](https://raw.githubusercontent.com/aaronMulveyAI/Polka-Dot-Detector/main/images/Loss.png)


#### c) **Optimal Hyperplane**
- A normalized feature space with the optimal hyperplane plotted.
- The hyperplane separates the two classes using linear algebra.

![Optimal Hyperplane](https://github.com/aaronMulveyAI/Polka-Dot-Detector/blob/main/images/hyperplane.gif?raw=true)

---

### 4. Model Training

The project implements a linear model for classification, determined as follows:
- Data is normalized using `StandardScaler`.
- The optimal hyperplane is computed using the closed-form solution for weights in linear regression:

Code snippet:
```python
A = np.zeros((4, 4))
b = np.zeros((4, 1))
for i, feature_row in enumerate(features_normalized):
    x = np.append([1], feature_row).reshape((4, 1))
    y = self.labels[i]
    A = A + x @ x.T
    b = b + x * y

invA = np.linalg.inv(A)
W = np.dot(invA, b)
```

The weights `W` are then used to define the hyperplane.

---

### 5. Prediction

The GUI provides an interface to predict unseen images. A simple sign function is used for classification based on the extracted RGB features:
```python
prediction = np.sign(0.5 * features[0] + 0.5 * features[1] - 0.5)
```
![Prediction Interface]("https://raw.githubusercontent.com/aaronMulveyAI/Polka-Dot-Detector/blob/main/images/Prediction.png")

#### Results:
- If the prediction result is `-1`, the image is classified as **Dysplastic Nevi**.
- Otherwise, it is classified as **Spitz Nevus**.

![Prediction Result]("https://raw.githubusercontent.com/aaronMulveyAI/Polka-Dot-Detector/blob/main/images/Class1.png")

---

## Code Structure

The project directory is structured as follows:
```
Polka Dot Detector/
├── data/
│   ├── datasetLunares/
│   │   ├── dysplasticNevi/
│   │   └── spitzNevus/
├── FeatureExtraction.py
├── PolkaDotDetectorApp.py
└── README.md
```
- **data/**: Contains the image dataset.
- **FeatureExtraction.py**: Contains the feature extraction logic.
- **main.py**: The main script implementing the GUI and model.
- **README.md**: Project documentation.

---

## How to Run the Project

1. Install the required dependencies (see below).
2. Ensure the dataset folder is structured correctly.
3. Run the `PolkaDotDetectorApp.py` script:
   ```bash
   python PolkaDotDetectorApp.py
   ```

---

## Dependencies

Install the following Python libraries:
```bash
pip install numpy opencv-python-headless matplotlib scikit-learn pillow
```


---

## Notes
- Ensure that image paths are updated according to your local directory.
- The prediction model is a basic linear classifier and can be improved using more advanced machine learning algorithms.

---

