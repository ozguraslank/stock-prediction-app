![Python versions](https://img.shields.io/badge/python_3.11+-blue)
# Stock Price Prediction App
Stock price prediction app provides future price predictions for the S&P 100 Index stocks for next 90 days by using artificial intelligence models. <br>
It also provides comprehensive analysis about the statistics of a spesific stock in a desired time interval.

## Application Overview

### Stock Price Prediction Page (Home Page)
![prediction_page](images/home_page.png)

### Stock Analysis Page
![prediction_page](images/analysis_page.png)

## How to Launch The App?

### 1. Clone The Repository to Your Local
` git clone https://github.com/ozguraslank/stock-prediction-app.git `

### OPTIONAL: Docker (Skip step 2, 3, 4 and 5 If you are going to use Docker for launch)
` docker build -t <IMAGE_NAME> . ` <br>
` docker run -p 8501:8501 <IMAGE_NAME> `


### 2. Create a New Python environment to Avoid Conflicts
` python3 -m venv <ENV_NAME> `

### 3. Change The Environment
#### --Windows--
` .\<ENV_NAME>\Scripts\activate `

#### --Linux--
` source <ENV_NAME>/bin/activate `

### 4. Install The Required Packages Given In requirements.txt
` pip3 install -r requirements.txt ` 

### 5. Go to Source Folder and Run The Application
` cd src ` <br>
` streamlit run app.py ` 

---------------------------------
After these steps, your web app will be deployed on web, both in localhost and network. <br>
URL for both localhost and network will be shown in the console after executing the command given in step 5.
