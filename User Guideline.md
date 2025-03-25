# Inpatient Mortality Risk Prediction System User Guide

## System Overview
This system is an AI-powered tool for predicting inpatient mortality risk, aiding healthcare professionals in early risk assessment and decision-making. It supports both single-patient data entry and batch prediction via CSV upload.

## Environment Setup

### Hardware Requirements
- **RAM**: Minimum 4GB, 8GB or more recommended.
- **Storage**: Sufficient space for data and models.
- **Display**: Minimum resolution 1280x720.

### Software Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Browser**: Modern browsers like Chrome, Firefox, or Edge
- **Python Environment**: Install required libraries using pip or conda.

| Library       | Minimum Version | Purpose                         |
| ------------- | --------------- | ------------------------------- |
| numpy         | 1.21.0          | Basic data processing           |
| pandas        | 1.3.0           | Data analysis                   |
| scikit-learn  | 0.24.2          | Machine learning tools          |
| xgboost       | 1.5.0           | Gradient boosting framework     |
| lightgbm      | 3.3.0           | Lightweight gradient boosting   |
| tensorflow    | 2.8.0           | Deep learning framework         |
| streamlit     | 1.10.0          | Web app framework               |
| matplotlib    | 3.4.3           | Data visualization              |
| seaborn       | 0.11.2          | Enhanced data visualization     |
| cloudpickle   | 2.0.0           | Model serialization             |
| pathlib       | 1.0.1           | Path operations                 |
| python-dotenv | 0.19.0          | Environment variable management |
| jupyter       | 1.0.0           | Notebook environment (optional) |
| notebook      | 6.4.0           | Notebook server (optional)      |

### Setup Instructions
1. **Install Python**: Download from [python.org](https://www.python.org/), version 3.8+ recommended.
2. **Install Anaconda**: Download from [anaconda.com](https://www.anaconda.com/), helps manage Python environments.
3. **Create Virtual Environment**:
   ```bash
   conda create -n patient_risk_env python=3.9
   conda activate patient_risk_env
   ```
4. **Install Dependencies**:
   ```bash
   pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=0.24.2 xgboost>=1.5.0 lightgbm>=3.3.0 tensorflow>=2.8.0 streamlit>=1.10.0 matplotlib>=3.4.3 seaborn>=0.11.2 cloudpickle>=2.0.0 pathlib>=1.0.1 python-dotenv>=0.19.0 jupyter>=1.0.0 notebook>=6.4.0
   ```

### Launching the System
- **Command Line**: Run `streamlit run .\patient_risk_interface.py`
- **Quick Launch**: Use the provided `start_web.bat` file.

## Usage Instructions

### Accessing the System
Enter the system URL in your browser to access the interface.

### Choosing Data Input Method
Select between:
- **Manual Entry**: For single-patient prediction.
- **CSV Upload**: For batch prediction.

### Manual Data Entry
1. Choose "Manual Entry".
2. Input patient features as prompted.
3. Click "Predict Mortality Risk" to get results.

### Batch Prediction via CSV
1. Select "CSV Upload".
2. Prepare a CSV file with required columns.
3. Upload the file and initiate prediction.

## Result Interpretation
Results display mortality probability (0-1) and survival outcome ("Survived" or "Deceased").

## Important Notes
- Ensure data accuracy for reliable predictions.
- Predictions are for reference only; clinical decisions should be made by healthcare professionals.

  

