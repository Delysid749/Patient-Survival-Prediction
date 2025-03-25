# Inpatient Mortality Risk Prediction System User Guide

## I. System Overview
This system is an inpatient mortality risk prediction tool developed based on machine learning technology. It can help medical staff quickly assess the mortality risk of patients in the early stages of hospitalization, thereby providing data support for clinical decision-making. The system offers two data input methods to meet different usage scenarios and needs.

Please download our project through the following methods:

Click the download link: <https://codeload.github.com/Delysid749/Patient-Survival-Prediction/zip/refs/heads/main>
Clone this project using git (recommended): `git clone https://github.com/Delysid749/Patient-Survival-Prediction.git`

## II. Environment Preparation

### Hardware Requirements
- **Memory**: It is recommended to have more than 4GB RAM.
- **Storage**: Sufficient disk space to store data and model files.

### Software Requirements
- **Operating System**: Windows 10/11
- **Browser**: Modern browsers such as Google Chrome, Mozilla Firefox, or Microsoft Edge
- **Python Environment**: Please ensure that the following dependent libraries are installed. It is recommended to use Anaconda for environment management.

| Library Name  | Version Requirement | Purpose                                        |
| ------------- | ------------------- | ---------------------------------------------- |
| numpy         | >=1.21.0            | Basic data processing and scientific computing |
| pandas        | >=1.3.0             | Data processing and analysis                   |
| scikit-learn  | >=0.24.2            | Machine learning algorithms and tools          |
| xgboost       | >=1.5.0             | Implementation of boosting tree algorithm      |
| lightgbm      | >=3.3.0             | Lightweight gradient boosting framework        |
| tensorflow    | >=2.8.0             | Deep learning framework (for DNN models)       |
| streamlit     | >=1.10.0            | Web application framework                      |
| matplotlib    | >=3.4.3             | Data visualization                             |
| seaborn       | >=0.11.2            | Enhanced data visualization                    |
| cloudpickle   | >=2.0.0             | Model serialization and deserialization        |
| pathlib       | >=1.0.1             | Path operations                                |
| python-dotenv | >=0.19.0            | Environment variable management                |
| jupyter       | >=1.0.0             | Notebook environment (optional)                |
| notebook      | >=6.4.0             | Notebook server (optional)                     |

### Environment Setup Steps
1. **Install Python**: Download and install Python from the [Python official website](https://www.python.org/). It is recommended to use Python version 3.8 or higher.
2. **Install Anaconda**: Download and install Anaconda from the [Anaconda official website](https://www.anaconda.com/). This will help you manage the Python environment and dependent packages more conveniently.
3. **Create a Virtual Environment**: Run the following command in the Anaconda Prompt to create a new virtual environment:
   ```
   conda create -n patient_risk_env python=3.9
   ```
   Then activate the virtual environment:
   ```
   conda activate patient_risk_env
   ```
4. **Install Dependent Packages**: In the virtual environment, use the following command to install the required dependent packages:
   ```
   pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=0.24.2 xgboost>=1.5.0 lightgbm>=3.3.0 tensorflow>=2.8.0 streamlit>=1.10.0 matplotlib>=3.4.3 seaborn>=0.11.2 cloudpickle>=2.0.0 pathlib>=1.0.1 python-dotenv>=0.19.0 jupyter>=1.0.0 notebook>=6.4.0
   ```


### Launch the System
- **Command Line Launch**: Run the following command in the command prompt or terminal:
  ```
  streamlit run .\patient_risk_interface.py
  ```
- **Shortcut Launch**: We also provide a batch file `start_web.bat`. You can simply double-click this file to quickly launch the system.

## III. Operating Procedures

### (i) Access the System
Enter the system URL in the browser address bar to access the system page.

### (ii) Select Data Input Method
The system offers two data input methods. You can choose according to the actual situation:
- **Manual Input for a Single Patient**: Suitable for predicting the risk of a single patient.
- **Batch Prediction by Uploading CSV File**: Suitable for predicting the mortality risk of multiple patients in batches.

### (iii) Manually Input Information for a Single Patient
1. **Select Input Method**: Click on the "Manual Input for a Single Patient" option on the left.
2. **Input Patient Feature Information**: Enter the relevant feature information of the patient item by item according to the system prompt, including numerical and categorical features.
3. **Initiate Prediction**: Click the "Predict Patient Mortality Risk" button, and the system will automatically perform the prediction.
4. **Field Introduction**: For detailed information, please refer to the project file `col_description.md`.

### (iv) Batch Prediction by Uploading CSV File
1. **Select Input Method**: Click on the "Batch Prediction by Uploading CSV File" option on the left.
2. **Prepare CSV File**: Ensure that the CSV file contains the fields required by the model and that the data format is correct.
3. **Upload File**: Click the "Select CSV File" button, choose the prepared CSV file from your local machine and upload it.
4. **Initiate Prediction**: The system will automatically perform the prediction and display the results after the prediction is completed.

## IV. Result Interpretation
The prediction results will be displayed on the page, including the predicted mortality risk probability and survival outcome. The mortality risk probability is a value between 0 and 1, indicating the likelihood of the patient's death; the survival outcome will be directly displayed as "Survived" or "Deceased".

## V. Precautions
1. **Data Accuracy**: The input data should be as accurate and complete as possible to ensure the reliability of the prediction results.
2. **Result Reference**: The prediction results are for reference only. The final medical decision should be made by professional medical staff based on the actual situation.