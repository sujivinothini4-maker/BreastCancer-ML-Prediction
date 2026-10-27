# Breast Cancer Prediction Using Machine Learning

## Project Overview

This project demonstrates a machine learning application that classifies tumor samples as benign or malignant using numerical features from the Breast Cancer Wisconsin Diagnostic dataset.

The application is developed using Python, Scikit-learn, Pandas, and Streamlit.

## Objectives

* Explore and analyze a public breast cancer dataset.
* Train and evaluate machine learning classification models.
* Compare Logistic Regression and Random Forest.
* Build an interactive web application for predictions.
* Display prediction results and model output probabilities.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through Scikit-learn.

The dataset contains 569 samples and 30 numerical features.

The target classes are benign and malignant.

## Project Files

* `train_model.py` — trains and evaluates the machine learning models.
* `app.py` — provides the Streamlit web interface.
* `breast_cancer_model.pkl` — stores the trained model and its metadata.
* `requirements.txt` — lists the required Python packages.

## How to Run the Project

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Start the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Model Evaluation

The training script compares Logistic Regression and Random Forest using a held-out test set.

It reports accuracy, precision, recall, F1-score, and a confusion matrix.

Model performance should be assessed using multiple metrics rather than accuracy alone.

## Limitations

This project is an educational machine learning prototype. It has not been validated for clinical use and must not be used to diagnose cancer or guide medical treatment.

## Future Enhancements

* Add more detailed data visualizations.
* Improve model evaluation and validation.
* Add explainability tools for feature analysis.
* Improve the user interface.
* Deploy the application online.

## Author

Developed as a student machine learning project.
