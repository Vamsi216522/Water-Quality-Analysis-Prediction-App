
# Water Quality Analysis & Prediction App

This project is a **Streamlit-based application** for analyzing and predicting key water quality parameters — **Dissolved Oxygen (DO)** and **Conductivity** — using a **multi-output machine learning model**. It provides an interactive platform for environmental monitoring, water quality assessment, and decision-making.

## Features
- 📤 **Data Upload** — Supports CSV and Excel files with water quality measurements.  
- 🤖 **Machine Learning Model** — Implements **XGBoost** with multi-output regression to predict DO and Conductivity simultaneously.  
- 📊 **Interactive EDA** — Automatically generates descriptive statistics and visualizations using **Plotly**.  
- 📈 **Predictions** — Compares predicted values with actual data for validation.  
- 🎯 **Model Training** — Retrain the model with new datasets directly in the app.  
- 💾 **Pre-trained Model** — Includes a ready-to-use `multioutput_water_model.pkl`.  

## Tech Stack
- **Frontend/UI:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** scikit-learn, XGBoost  
- **Visualization:** Plotly Express  
- **Model Storage:** Joblib  

## Workflow
1. **Upload Dataset** — Import CSV/Excel files containing water quality attributes.  
2. **EDA & Visualization** — View summary statistics, correlation heatmaps, and scatter plots.  
3. **Model Training** — Train or retrain the regression model on your data.  
4. **Prediction** — Generate predictions for DO and Conductivity.  
5. **Decision Support** — Use visual insights for water quality management.  

## Applications
- Environmental research and conservation projects  
- Water treatment facility quality control  
- Academic and student projects in data science & environmental engineering  
- Government agencies monitoring water standards  

## Repository Structure
```
water_analysis/
│── analysis.ipynb              # Jupyter notebook for experiments & EDA  
│── app.py                      # Streamlit application script  
│── WaterQuality data.xlsx      # Sample dataset  
│── multioutput_water_model.pkl # Pre-trained ML model  
│── .venv/                      # Virtual environment (optional)  
```

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/water_quality_app.git
cd water_quality_app/water_analysis

# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the Streamlit app
streamlit run app.py
```


