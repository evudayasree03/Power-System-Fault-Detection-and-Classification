# Power-System-Fault-Detection-and-Classification

# ⚡ Power System Fault Detection and Classification

## 📌 Overview

Power distribution systems are critical infrastructure that require continuous monitoring to ensure reliability and stability. Faults such as **Transformer Failures, Line Breakages, and Overheating** can lead to power outages, equipment damage, and increased operational costs.

This project presents an **AI-powered Fault Detection and Classification System** that utilizes Machine Learning techniques to analyze electrical, environmental, and equipment health parameters for intelligent fault identification. The solution helps utility operators detect faults quickly, improve maintenance planning, and enhance overall grid reliability.

---

## 🎯 Problem Statement

Design a machine learning model to detect and classify faults in a power distribution system using electrical measurements and operational data.

The system should:

- Detect abnormal operating conditions.
- Classify different fault types.
- Analyze electrical, environmental, and maintenance-related parameters.
- Support preventive maintenance and decision-making.
- Improve power grid reliability and reduce downtime.

---

## 📊 Dataset Description

The dataset contains information related to power system faults, including:

### Electrical Parameters
- Voltage (V)
- Current (A)
- Power Load (MW)

### Environmental Factors
- Temperature (°C)
- Wind Speed
- Weather Condition

### Asset Information
- Maintenance Status
- Component Health

### Fault Information
- Fault Type
- Fault Duration
- Downtime

### Location Data
- Latitude
- Longitude

---

## 🔍 Fault Categories

The model classifies the following fault types:

- Transformer Failure
- Line Breakage
- Overheating

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Machine Learning
- Scikit-Learn
- XGBoost

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### AI Platform
- IBM watsonx.ai
- IBM Granite Models

### Workflow Orchestration
- LangFlow

### Dashboard
- Streamlit

---

## 🏗️ System Architecture

```text
Power System Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
XGBoost Classification Model
        │
        ▼
Fault Detection & Classification
        │
        ▼
Visualization Dashboard
```

---

## ⚙️ Machine Learning Workflow

### Data Preprocessing
- Missing value handling
- Categorical encoding
- Feature scaling
- Data cleaning

### Feature Engineering
- Voltage-Current Ratio
- Load-Current Ratio
- Temperature-Load Interaction
- Downtime Severity Metrics

### Model Training
- Train-Test Split
- Cross Validation
- Hyperparameter Tuning
- XGBoost Classification

### Model Evaluation
- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## ✨ Features

- Automated Fault Detection
- Fault Type Classification
- Real-Time Prediction Capability
- Data Visualization Dashboard
- Predictive Maintenance Support
- Explainable AI using Feature Importance Analysis

---

## 📂 Project Structure

```text
Power-System-Fault-Detection/
│
├── dataset/
│   └── fault_data.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── prediction.py
│
├── models/
│   └── fault_detection_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── presentation.pptx
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/power-system-fault-detection.git
cd power-system-fault-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the Model

```bash
python train_model.py
```

### Run Predictions

```bash
python prediction.py
```

### Launch Dashboard

```bash
streamlit run app.py
```

---

## 📈 Results

The model is trained using historical power system fault data and evaluated using standard classification metrics.

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Feature Importance Analysis

---

## 💡 Novelty and Uniqueness

- Intelligent Machine Learning-Based Fault Classification
- Multi-Parameter Analysis Using Electrical and Environmental Data
- Real-Time Fault Detection Capability
- Predictive Maintenance Support
- Scalable and Explainable AI Solution

---

## 🔮 Future Scope

- Smart Grid and IoT Sensor Integration
- AI-Based Predictive Maintenance
- Advanced Deep Learning Models
- SCADA System Integration
- Fault Localization and Root Cause Analysis

---

## 👨‍💻 Author

**Project Title:** Power System Fault Detection and Classification

**Developed As Part of:** IBM University Engagement Program

---

## 📜 License

This project is developed for academic and educational purposes under the IBM University Engagement Program.
