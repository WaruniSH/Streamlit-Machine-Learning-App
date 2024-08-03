# Streamlit-Machine-Learning-App

### Overview
This repository contains a Streamlit application designed to predict the likelihood of lung cancer in users based on their input data. The app utilizes a machine learning model to analyze user-provided health metrics and risk factors to determine the probability of lung cancer.

### Features
* User-Friendly Interface: A simple and intuitive interface built with Streamlit to facilitate easy user interaction.
* Machine Learning Model: A trained machine learning model that predicts the likelihood of lung cancer based on user inputs.
* Real-Time Predictions: Instantly receive a prediction result upon entering the required information.
* Data Security: Ensures user data privacy and security throughout the process.
* Multiple Pages: Navigate through different sections of the app:
    * Home Page: Introduction and overview of the application.
    * Data Visualization Page: Visualize various health metrics and risk factors.
    * Cancer Prediction Page: Enter user data and receive lung cancer prediction.

### How It Works
* User Input: Users provide necessary health information, such as age, smoking history, family medical history, etc.
* Prediction Model: The app leverages a pre-trained machine learning model to analyze the input data.
* Result: The app displays the prediction result, indicating whether the user is likely to have lung cancer or not.

### Pages
* Home Page: Provides an introduction and overview of the Lung Cancer Prediction App, explaining its purpose and how it works.
* Data Visualization Page: Allows users to explore and visualize various health metrics and risk factors related to lung cancer.
* Cancer Prediction Page: The main page for entering user data and getting the lung cancer prediction result.

### Installation

To run the application locally, follow these steps:
1. Clone the repository:
```
git clone https://github.com/WaruniSH/Streamlit-Machine-Learning-App.git
cd Streamlit-Machine-Learning-App
```
2. Create a virtual environment:
```
python3 -m venv env
source env/bin/activate   # On Windows, use `env\Scripts\activate`
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Run the Streamlit app:

```
streamlit run main.py
```
### Usage

Once the application is running, open your web browser and navigate to the provided local URL (usually http://localhost:8501). Use the navigation menu to access the Home Page, Data Visualization Page, and Cancer Prediction Page. Enter the required health information on the Cancer Prediction Page and click the 'Predict' button to see the results.

### Contributing
We welcome contributions to enhance the functionality and accuracy of the Lung Cancer Prediction App. Please feel free to submit pull requests or report issues.

### Acknowledgements
We would like to thank the contributors of the open-source libraries and datasets that made this project possible.
