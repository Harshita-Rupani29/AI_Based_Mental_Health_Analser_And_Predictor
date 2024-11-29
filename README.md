
---

# **AI-Based Mental Health Analyzer and Predictor**

## **Overview**
This project is a comprehensive AI-driven application designed to analyze and predict mental health trends. It includes a **Streamlit dashboard** for user interaction and **machine learning models** to make predictions based on mental health-related data. The project demonstrates the integration of **Exploratory Data Analysis (EDA)**, **feature engineering**, and multiple predictive models for meaningful insights.

---

## **Features**

### **Dashboard (`app.py`)**
- **Mental Fitness Score Prediction**: 
  - Allows users to input mental health-related statistics (e.g., schizophrenia, anxiety rates) and predict a **Mental Fitness Score**.
  - Provides insights into **Disability Adjusted Life Years (DALYs)**.
- **Future Trend Forecasting**:
  - Forecasts mental health trends for specific countries using the **Prophet model**.
  - Visualizes historical data and predictions with confidence intervals.

### **Machine Learning Pipeline (`ai.ipynb`)**
- **Exploratory Data Analysis (EDA)**:
  - Detailed analysis of the dataset to identify trends and patterns.
- **Feature Engineering**:
  - Applied **Principal Component Analysis (PCA)** for dimensionality reduction.
- **Model Training and Evaluation**:
  - Implemented and fine-tuned multiple models:
    - **Linear Regression**
    - **Lasso Regression**
    - **Random Forest**
    - **Multi-Layer Perceptron (MLP)**
  - Compared model performance with and without PCA.

---

## **Technologies Used**
- **Python Libraries**:
  - `streamlit`: For the interactive dashboard.
  - `pandas`, `numpy`: For data manipulation and analysis.
  - `matplotlib`, `seaborn`: For data visualization.
  - `Prophet`: For time-series forecasting.
  - `joblib`: For model saving and loading.
  - `scikit-learn`: For machine learning models and PCA.
- **Models**:
  - Random Forest, Linear Regression, Lasso Regression, Multi-Layer Perceptron (MLP).

---

## **How to Run the Project**

### **1. Clone the Repository**
```bash
git clone https://github.com/Harshita-Rupani29/AI_Based_Mental_Health_Analyzer_And_Predictor.git
cd AI_Based_Mental_Health_Analyzer_And_Predictor
```

### **2. Install Dependencies**
Install the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

### **3. Run the Dashboard**
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

### **4. Explore the Notebook**
Open `ai.ipynb` in a Jupyter Notebook or equivalent to:
- Perform EDA.
- Train and compare models with and without PCA.

---

## **Project Structure**
```
AI_Based_Mental_Health_Analyzer_And_Predictor/
├── app.py                     # Streamlit dashboard script
├── ai.ipynb                   # Notebook for EDA, feature engineering, and modeling
├── dff_output.csv             # Dataset used for predictions
├── rf_model.pkl               # Trained Random Forest model
├── requirements.txt           # Required Python dependencies
└── README.md                  # Project documentation
```

---

## **Future Enhancements**
- Integrate additional mental health parameters for more accurate predictions.
- Extend time-series forecasting to include more robust models.
- Deploy the dashboard on a public platform like **Heroku** or **Streamlit Sharing**.

---

## **License**
This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---
