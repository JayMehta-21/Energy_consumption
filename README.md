⚡ Energy Consumption Optimization (ML-Based Prediction)

📘 Overview This project focuses on analyzing and predicting household energy consumption patterns using machine learning. The main goal is to understand energy usage trends and build a model that can optimize power consumption for better efficiency. Phase 1 (Data Cleaning & Preprocessing) has been completed. Phase 2 (Model Training & Interpretation) has also now been completed.

🧩 Phase 1 – Data Understanding & Preprocessing ✔ Tasks Completed: Data loading and inspection Replaced missing values and converted data types Unified date and time into a DateTime index Resampled data to hourly level to reduce noise and learn clear patterns

Feature extraction: hour, day, day_of_week, month, is_weekend Outlier and distribution exploration Correlation heatmap and feature relationship analysis

Feature encoding and scaling where required

🤖 Phase 2 – Model Building & Evaluation 
RandomForest Regressor (Final) MAE:- 0.109, RMSE:- 0.154, R²:-0.780 Captures non-linear usage behavior

Interpretation: The RandomForest model clearly performed better, showing that energy usage is non-linear and depends heavily on behavior + appliance usage patterns.

🔍 Key Insights 
Hour of the day strongly influences consumption → Evening usage peaks between 7 PM – 10 PM
Weekend vs Weekday difference is small → Daily routines matter more than weekly patterns

📂 Dataset This project uses the Household Electric Power Consumption Dataset. Download from: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Dataset is not included in this repository due to GitHub file size limits. Place the dataset in a folder named data/ inside the project directory.
