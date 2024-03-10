< 프로젝트 진행 개요 >

1. Load Data
2. Preprocessing
3. Feature Selection
4. Split
5. Model Training, Scoring by Regression
6. Use Ensemble
7. Feature Importance
8. Evaluation
9. Predict new data
10. Summary

< 파일 구성 >

1. 01

< 컬럼 분석 >

- Gender : 성별
- Age : 나이
- Height : 키 m
- Weight : 몸무게 kg
- family_history_with_overweight : 가족력 -> one-hot 필요
- Frequent consumption of high caloric food 고열량 식품 빈도(FAVC)
- Frequency of consumption of vegetables 채소 빈도(FCVC)
- Number of main meals 하루 끼니 횟수(NCP)
- Consumption of food between meals 간식(CAEC)
- SMOKE : 흡연 유무
- Consumption of water daily 물 섭취(CH20)
- The attributes related with the physical condition are: Calories consumption monitoring 칼로리 소비(SCC)
- Physical activity frequency 활동량(FAF)
- Time using technology devices 서마터폰 사용량(TUE),
- Consumption of alcohol 음주빈도(CALC)
- Transportation used 주이동수단(MTRANS)

[ 1. Feature Labeling ]
CAEC
('Always', 0) ('Frequently', 1) ('Sometimes', 2) ('no', 3)

CALC
('Always', 0) ('Frequently', 1) ('Sometimes', 2) ('no', 3)

NObeyesdad
('Insufficient_Weight', 0) ('Normal_Weight', 1) ('Overweight_Level_I', 2)
('Overweight_Level_II', 3) ('Obesity_Type_I', 4) ('Obesity_Type_II', 5)
('Obesity_Type_III', 6)

Gender
('Female', 0) ('Male', 1)

family_history_with_overweight
('yes', 0) ('no', 1)

FAVC
('no', 0) ('yes', 1)

SMOKE
('no', 0) ('yes', 1)

SCC
('no', 0) ('yes', 1)

[ 2. 분석 결과 ]

1. 남녀 분포는 비슷하고 20대의 수가 제일 많다.
2. BMI와 비교할 때 키는 상관계수가 아주 낮아 보이고 몸무게는 절대적으로 관련이 깊다
3. 가족력은 긍정 비율이 낮고 그 중에서도 비만은 적다
4. FAVC 고열량 섭취 빈도는 오히려 0 값의 분포가 높다
5. FCVC 채소 섭취 빈도는 2 값의 분포가 높다
6. NCP 하루 끼니 수는 3 값이 많지만 BMI 와 관련은 없다


[ 모델링 고민 ]
- 절대적으로 한 값이 많은 컬럼을 사용하는 게 좋을까? : 일단 제거해본다

    - Gender : 성별
    - Age : 나이
    - Height : 키 m
    - # Weight : 몸무게 kg
    - family_history_with_overweight : 가족력 -> one-hot 필요
    - Frequent consumption of high caloric food 고열량 식품 빈도(FAVC)
    - Frequency of consumption of vegetables 채소 빈도(FCVC)
    - Number of main meals 하루 끼니 횟수(NCP)
    - Consumption of food between meals 간식(CAEC)
    - # SMOKE : 흡연 유무
    - Consumption of water daily 물 섭취(CH20)
    - # The attributes related with the physical condition are: Calories consumption monitoring 칼로리 소비(SCC)
    - Physical activity frequency 활동량(FAF)
    - Time using technology devices 서마터폰 사용량(TUE),
    - Consumption of alcohol 음주빈도(CALC)
    - # Transportation used 주이동수단(MTRANS)-> 자동차, 대중교통만

- 선형은 정규화 진행

