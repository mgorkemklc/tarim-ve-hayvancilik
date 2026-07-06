🐄 AI-Powered Milk Yield Forecasting Dashboard

Hi there! Welcome to my repository. I built this interactive web dashboard as a data science case study to predict future milk production using machine learning.

The application analyzes historical agricultural data and provides future projections (e.g., "What will be the milk yield next Wednesday?") at both the macro (farm) and micro (individual animal) levels.

🧠 The Approach & Machine Learning Model

For this project, I chose the Holt-Winters (Exponential Smoothing) algorithm for time-series forecasting.

If you look at the historical data in the charts, you'll notice a lot of sharp zig-zags. In real-world agriculture, this daily volatility is often just "noise" caused by temporary environmental factors (weather, animal stress, slight delays in milking times, etc.).

If the model tried to replicate these extreme spikes, it would suffer from overfitting. Instead, my model is designed to filter out this daily noise to capture the actual underlying trend and 7-day seasonality. That's why the projected forecast lines appear much smoother and more stable—it provides a realistic and reliable expectation rather than an overfitted guess.

✨ Features

Farm-Level Forecasting: Analyzes macro-level data to predict the average daily milk yield of an entire farm.

Individual Animal Tracking: Processes specific animal IDs/tags to forecast individual performance.

Real-Time Accuracy Metrics: Calculates and displays the Mean Absolute Error (MAE) dynamically.

Interactive UI: Built with Streamlit, allowing users to simply upload their .xlsx or .csv files and instantly get visual dashboards without touching any code.

🚀 How to Run the Project Locally

It's very easy to get this up and running on your own machine. Just follow these steps:

1. Clone the repository

git clone https://github.com/mgorkemklc/tarim-ve-hayvancilik.git

cd tarim-ve-hayvancilik


2. Install the required dependencies
Make sure you have Python installed. Then, run:

pip install streamlit pandas matplotlib statsmodels openpyxl


(Note: openpyxl is required for Pandas to read Excel files properly.)

3. Run the application
Start the Streamlit server by running:

streamlit run app.py


The dashboard should automatically open in your default web browser at http://localhost:8501.

🛠️ Troubleshooting

Streamlit asking for an email? If Streamlit pauses in the terminal to ask for an email for their newsletter, you can just leave it blank and press Enter. To bypass it completely, run the app with this command instead:
streamlit run app.py --browser.gatherUsageStats False

📂 Data Format

To test the application, use the sidebar to upload the provided dataset files:

Upload ciftlik_bazli_sut_verimi (1).xlsx to the Farm Data section.

Upload hayvan_bazli_sut_verimi_yeni.csv to the Animal Data section.

Developed as a task-oriented technical case study for AI applications in agriculture and livestock.
