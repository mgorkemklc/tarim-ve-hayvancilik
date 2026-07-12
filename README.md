🌍 Universal AutoML & Time-Series Forecasting Dashboard

Hi there! Welcome to my repository. 👋

What started as a specific data science case study for predicting agricultural milk yields has completely evolved into a fully-fledged, Universal AutoML (Automated Machine Learning) Dashboard.

I built this interactive web application to allow anyone to upload any time-series dataset, dynamically map their columns, and instantly pit 5 different machine learning algorithms against each other to forecast future trends. No coding required.

The core philosophy of this project is flexibility. It doesn't matter if you upload dairy sales, weather data, or stock prices—the system will understand it, clean it, and forecast it.

✨ Key Features & Upgrades

🔓 Universal Data Ingestion: Say goodbye to hardcoded column names! Upload any .csv or .xlsx file. The UI dynamically reads your headers and lets you map your Date, Target, and Category columns on the fly.

🤖 The AutoML Engine: The system automatically splits the data, trains, and evaluates 5 distinct models simultaneously. The competitor lineup includes:

-Holt-Winters (Exponential Smoothing for baseline seasonality)

-Random Forest Regressor (Ensemble learning)

-Gradient Boosting Regressor (Advanced boosting)

-Linear Regression (Statistical baseline)

-Support Vector Regression (SVR) (Margin-based predictions)

🎯 Dynamic Target Forecasting: Don't just settle for a generic "next 30 days" view. Users can select any specific future date using an interactive sidebar calendar, and the system will dynamically extend its forecast horizon up to that exact day.

📅 Flight-Booking Style Calendar UI: A custom-built HTML/CSS interactive calendar strip displays day-by-day AI predictions leading up to your target date—heavily inspired by the intuitive UI of flight booking engines (like Skyscanner).

📊 Real-Time Metrics & Logging: Dynamically generates comparison bar charts (using Seaborn) based on Mean Absolute Percentage Error (MAPE). It automatically crowns the winning model and displays a transparent, step-by-step AI Action Log so you can see exactly what the engine did behind the scenes.

🚀 How to Run the Project Locally

It is incredibly easy to get this project up and running on your own machine. Just follow these 3 simple steps:

1. Clone the repository

git clone [https://github.com/mgorkemklc/tarim-ve-hayvancilik.git](https://github.com/mgorkemklc/tarim-ve-hayvancilik.git)
cd tarim-ve-hayvancilik


2. Install the required dependencies
Make sure you have Python 3.8+ installed. Then, install all the necessary libraries by running:

pip install -r requirements.txt


3. Run the application
Start the Streamlit server by running the following command:

streamlit run app.py


(The dashboard will automatically open in your default web browser at http://localhost:8501)

🛠️ Troubleshooting Tip

Is Streamlit asking for an email? If Streamlit pauses in the terminal to ask for an email for their newsletter, you can just leave it blank and press Enter. To bypass this prompt completely in the future, run the app with this flag:
streamlit run app.py --browser.gatherUsageStats False

💡 How to Use the Dashboard

Upload: Use the left sidebar to upload your dataset. (You can test the system right away using the ciftlik_bazli_sut_verimi (1).xlsx or hayvan_bazli_sut_verimi_yeni.csv files provided in this repository).

Map Columns: Tell the system which column represents the Date, which one is the Target Value to predict, and which one acts as the Category/ID.

Pick a Target Date: Choose any future date from the interactive calendar.

Launch: Click "🚀 Yapay Zekayı Başlat" (Launch AI) and watch the algorithms compete to give you the most accurate daily predictions!

Developed as an advanced technical case study demonstrating AutoML concepts, dynamic data mapping, and interactive UI/UX design in Python.