# Plant Disease Detection using CNN

A deep learning image classification project that detects plant diseases from leaf images, built to support early diagnosis in precision agriculture.

## 🌱 Problem

Manual inspection of crops for disease is slow, inconsistent, and doesn't scale — by the time a farmer spots visible symptoms, the disease may have already spread. An automated system that can flag disease from a simple leaf photo can catch problems earlier and reduce crop loss.

## 🧠 Approach

- Used **Convolutional Neural Networks (CNN)** with **MobileNetV2 transfer learning** to classify leaf images by disease type.
- Built the full pipeline: data preprocessing and augmentation → model training → validation → evaluation.
- Implemented in **TensorFlow / Keras**.

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- NumPy, Pandas, Matplotlib

## 📊 Pipeline

1. **Data Preprocessing** — resized and normalized leaf images, applied augmentation (rotation, flip, zoom) to improve generalization.
2. **Model Building** — used MobileNetV2 as a pretrained base, added custom classification layers on top.
3. **Training & Validation** — trained on the labeled dataset with a train/validation split, monitored accuracy and loss curves to avoid overfitting.
4. **Evaluation** — assessed performance using accuracy, precision, recall, and confusion matrix.

## 📈 Results

Achieved strong classification accuracy across disease categories, demonstrating that lightweight transfer-learning models like MobileNetV2 can be practical for real-world, on-device plant disease detection.

## 🚀 Impact

This kind of model can be deployed in low-cost mobile or edge devices, giving farmers a quick, accessible way to screen crops for disease and act before it spreads — supporting precision agriculture at scale.

## 📁 Project Structure

```
├── data/                # Leaf image dataset (train/val/test)
├── notebooks/           # Jupyter notebooks for EDA & experimentation
├── model/                # Saved model weights
├── train.py             # Training script
├── evaluate.py           # Evaluation script
└── README.md
```

## 🔧 How to Run

```bash
git clone https://github.com/ARYANTYAGI2003/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
python train.py
```

## 👤 Author

**Aryan Tyagi**
B.Tech Data Science Engineering, Manipal University Jaipur
