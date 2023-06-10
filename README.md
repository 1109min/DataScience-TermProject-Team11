# DataScience-TermProject-Team11

# End-to-End Process for Illegal Parking Prediction

This project is part of the Data Science course at the Computer Science Department of Gachon University. The purpose of this project is to predict and identify factors related to illegal parking issues in Seongnam-si, Gyeonggi-do, South Korea.

The dataset used in this project is sourced from the South Korea Open Data Portal. If any data is missing in the project folder, you can download it from the following link.

To complete the project, the following data sets must be downloaded from the link provided and stored in a folder location such as 'final.py'
* 경기도 성남시_주정차 위반 단속 위치 현황: <https://www.data.go.kr/data/15037104/fileData.do>
* 소상공인시장진흥공단_상가(상권)정보: <https://www.data.go.kr/data/15083033/fileData.do>
* 경기도 성남시_인구및세대_현황: <https://www.data.go.kr/data/15007386/fileData.do>

The '추가데이터'(Additional Data) folder may include datasets such as:
* 서울특별시 강동구_동별 불법주정타 단속현황: <https://www.data.go.kr/data/15081659/fileData.do>
* 서울특별시 강서구_불법주정차 단속 현황: <https://www.data.go.kr/data/15083768/fileData.do>
* 서울특별시 구로구_주정차단속현황: <https://www.data.go.kr/data/15034492/fileData.do>
* 서울특별시 서초구_주정차 단속 현황: <https://www.data.go.kr/data/15087185/fileData.do>
* 서울특별시 성북구_불법주정차 동별 데이터: <https://www.data.go.kr/data/15113658/fileData.do>
* 서울특별시 송파구_주정차단속건수정보: <https://www.data.go.kr/data/15048835/fileData.do>
* 서울특별시 영등포구_주정차단속현황: <https://www.data.go.kr/data/15034483/fileData.do>
* 서울특별시 용산구_불법주정차단속현황: <https://www.data.go.kr/data/15084175/fileData.do>
* 서울특별시 종로구_불법주정차 통계: <https://www.data.go.kr/data/15100293/fileData.do>
* 서울특별시_강남구_불법주정차단속현황: <https://www.data.go.kr/data/15048827/fileData.do>
* 서울특별시_서대문구_주정차 단속 현황: <https://www.data.go.kr/data/15034465/fileData.do>
* 서울 인구수 데이터: <https://www.data.go.kr/data/15046938/fileData.do>

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

## Regression Analysis & Evaluation

To predict the specific count of illegal parking incidents, we employed the Linear Ridge Lasso regression model. The model was evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE). The results indicated that the Ridge model achieved the highest accuracy.

---

## Classification Analysis & Evaluation

To categorize the count of illegal parking incidents as "low" or "high," we utilized several machine learning models: Decision Tree, K-Nearest Neighbors (KNN), Random Forest, and XGBoost. The performance evaluation included metrics such as Accuracy, Confusion Matrix, Precision, Recall, F1 Score, and K-Fold Cross Validation. Based on the K-Fold evaluation results, the Random Forest model demonstrated the highest accuracy.

The decision tree structure can be visualized in the file `final_decision_tree.png`.
