
# 👁️ Eye Disease Detection Using Convolutional Neural Networks (CNNs)

## 🔍 Project Overview
This project demonstrates the use of **Convolutional Neural Networks (CNNs)** for the **automated detection of common eye diseases** using **fundus images**. The goal is to bridge AI research with practical healthcare solutions through a **lightweight, efficient, and deployable system** for disease classification.

### 🚀 Technologies Used
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Seaborn / Matplotlib**
- **Flask (for GUI Integration)**

## 🏥 Diseases Detected
- **Normal**
- **Cataract**
- **Glaucoma**
- **Diabetic Retinopathy**

## 📊 Project Highlights
- Lightweight CNN architecture (~1.8M parameters)
- Achieved **~92% accuracy** on test data
- Training metrics: **Precision, Recall, F1-Score**
- GUI for easy interaction by non-technical users
- Data preprocessing includes: **Resizing, Normalization, Augmentation**
- Desktop deployment through **Flask-based GUI**
- Evaluation with confusion matrix and accuracy/loss curves

## 💻 Project Structure
```
├── app.py               # Flask app for GUI
├── model.h5             # Trained CNN model
├── templates/           # HTML templates for GUI
├── static/              # CSS / JS / Images for GUI
├── dataset/             # Fundus images dataset (organized by class)
├── preprocessing.ipynb  # Data preprocessing steps
├── training.ipynb       # Model training notebook
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## 🛠️ How to Run
### 1️⃣ Clone this Repository
```bash
git clone https://github.com/Likuna06/EyeDiseaseDetection.git
cd EyeDiseaseDetection
```

### 2️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python app.py
```
Access the GUI on `http://localhost:5000` in your browser.

## 📂 Dataset
The dataset is organized into folders:
```
/dataset
    /Normal
    /Cataract
    /Glaucoma
    /DiabeticRetinopathy
```

## 📈 Visual Results
- Training vs. Validation Accuracy / Loss Curves
- Confusion Matrix
- GUI Output Screenshots (Upload Image → Get Disease Prediction)

## 🔮 Future Improvements
- Mobile app integration (Android/iOS)
- TensorFlow Lite for offline access
- Expanded datasets for robustness
- Integration with hospital information systems (EHR)
- Explainability features (Grad-CAM)

## 🔗 Research Paper
**[👉 View Detailed Research on ResearchGate](https://www.researchgate.net/publication/393778069_Eye_Disease_Detection_Using_Convolutional_Neural_Networks_A_Research-Based_Project_on_the_Application_of_Deep_Learning_for_Medical_Image_Classification_Declaration)**

## 🤝 Connect with Me
Feel free to connect or collaborate:
- **LinkedIn:** [https://www.linkedin.com/in/likuna-swain-b25208346/]
- **Email:** likunaswain2@gmail.com

## 📄 License
This project is licensed under the **MIT License** - see the LICENSE file for details.
