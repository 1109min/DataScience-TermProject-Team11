# DataScience-TermProject-Team11

# End-to-End Process for Illegal Parking Prediction

This project is part of the Data Science course at the Software Engineering Department of Gachon University. The purpose of this project is to predict and identify factors related to illegal parking issues in Seongnam City, Gyeonggi Province, South Korea.

The dataset used in this project is sourced from the South Korea Open Data Portal. If any data is missing in the project folder, you can download it from the following link.

---

## Analysis Overview

The project explores the relationship between the number of illegal parking incidents at the neighborhood (dong) level and commercial district information. To address the issue of limited records when analyzing at the dong level, we incorporated market-level data instead. Additionally, we included data from Seoul City, which shares geographical and cultural similarities with Seongnam City, to enhance the analysis.

Please note that the current dataset contains only around 190 records, which may not be sufficient for comprehensive analysis.

---

## Usage

1. Run `preprocessing_seoul_data.py` to preprocess and merge the Seoul City data located in the additional_data folder. This will generate the file `서울시_동별_단속현황.xlsx`.
2. Execute `final.py` to obtain the project results.
3. Enter the neighborhood (dong) you want to predict:
   - Example: "야탑1동"
4. Provide the population growth rate and the year for prediction:
   - Example: "10 2024"

The prediction will be based on the provided information.

---

## Linear Ridge Lasso Regression Model

To predict the specific count of illegal parking incidents, we employed the Linear Ridge Lasso regression model. The model was evaluated using metrics such as Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE). The results indicated that the Ridge model achieved the highest accuracy.

---

## Categorization of Illegal Parking Incidents

To categorize the count of illegal parking incidents as "low" or "high," we utilized several machine learning models: Decision Tree, K-Nearest Neighbors (KNN), Random Forest, and XGBoost. The performance evaluation included metrics such as Accuracy, Confusion Matrix, Precision, Recall, F1 Score, and K-Fold Cross Validation. Based on the K-Fold evaluation results, the Random Forest model demonstrated the highest accuracy.

The decision tree structure can be visualized in the file `final_decision_tree.png`.
