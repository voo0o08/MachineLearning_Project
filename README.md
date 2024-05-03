# KDT5기 머신러닝 프로젝트 : 

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>

<img src="https://img.shields.io/badge/scikitlearn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/> <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/> <img src="https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/>


## team2 탈모와 비만 데이터를 통한 예측

<img width="446" alt="image" src="https://cdn.kormedi.com/wp-content/uploads/2022/09/unnamed-file-113.jpg">

현대인의 필수고민 비만과 탈모... 그렇다면 나는?? 머신러닝을 통해 예측해보자!!


## 비만 팀 소개 (클릭 시 상세내용 확인 가능)
<details>
<summary> 이윤서 : voo0o08 <a href="https://github.com/voo0o08" height="5" width="10" target="_blank"><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : 분류를 통한 비만 예측 / ppt 01p~19p</summary>

  ## 1. 데이터 전처리

- **라벨링**
    
    순서 상관 없는 데이터 → Label encoding
    
    순서 상관 있는 데이터 → Label encoding 함수 생성
    
    교통 수단은 5가지 → get_dummies(one_hot encoding)
    

- **샘플링 노이즈 제거**
    
    업샘플링은 데이터 원본 논문 참고
    
    [](https://www.sciencedirect.com/science/article/pii/S2352340919306985)
    
    |  | 업샘플링 전 | 업샘플링 후 | 처리 방식 |
    | --- | --- | --- | --- |
    | 범주형 데이터 | 좋음 3 / 보통 2 / 싫음 1 | 3 / 2.2548 / 1 / 2.4554| 범주형은 무조건 정수여야하기 때문에 round 처리 |
    | 수치형 데이터 | 36.5 / 12.4 / 89.4| 45.1648345 / 14.415612 / 36.5 | 데이터를 수집할 때 수집한 소수점 자리까지 round처리 → 소수점이 너무 많을 시, 중복 값 처리에서 걸리지지 않음 |
- **Regression을 위한 taget column 생성**
    
    <img width="887" alt="스크린샷 2024-03-20 170415" src="https://github.com/voo0o08/MachineLearning_Project/assets/155411941/21f9e367-9139-413c-a22f-7e0e3ca90dce">

    

- **중복 제거 및 클래스 균형 확인**
    
    <img width="829" alt="스크린샷 2024-03-20 170533" src="https://github.com/voo0o08/MachineLearning_Project/assets/155411941/e2589346-340c-4f65-bd88-b77921748552">

    
    중복 제거 후에도 클래스 균형 크게 달라지지 않음
    

- **상관관계 확인 후 불필요한 col 제거**
    
   <img width="920" alt="스크린샷 2024-03-20 170834" src="https://github.com/voo0o08/MachineLearning_Project/assets/155411941/b9d87bce-59fc-4545-98bd-928298901bfb">

    
    BMI의 경우 피쳐인 키와 몸무게의 제곱으로 구해진 값으로 공선성을 고려하여 키 feature제거
    

## 2. 분류 모델 분석 및 선택

- 실험 1. 단일 모델만 사용해보기
    
    <img width="926" alt="스크린샷 2024-03-20 171853" src="https://github.com/voo0o08/MachineLearning_Project/assets/155411941/c95c4c70-3524-40f2-914e-b7e4b4971632">

    
    **결과**
    
    상관관계가 0.1 이하인 column을 제거했을 때와 모두 포함했을 때의 결과로, 상관관계가 낮다고 무조건 불필요한 요소가 아님을 알 수 있다. 0.1이하인 데이터를 전부 drop했을 때 되려 값이 감소하였음 
    
- 실험 3. Logistic 알고리즘과 SGD
    
    
    | model | train | test |
    | --- | --- | --- |
    | LogisticRegression | 0.584 | 0.571 |
    | SGDClassifier(loss=”log_loss”) | 0.504 | 0.486 |
    
    **결과**
    
    LogisticRegression모델의 경우 전체 데이터를 학습하고, SGD의 경우 random하게 샘플을 뽑아 학습하기 때문에  점수가 더 낮을 수 밖에 없음. 따라서 데이터의 크기가 작거나 전체 데이터를 학습 시킬 여건이 된다면 확률적 경사하강법을 추천하지 않음
    
- 실험 2. 앙상블 사용해보기
    
    <img width="1271" alt="스크린샷 2024-03-20 172915" src="https://github.com/voo0o08/MachineLearning_Project/assets/155411941/8c793849-8f54-45c8-bff4-638503a882d0">

    
    앙상블의 경우 여러 모델들이 관여하기 때문에 단일 모델에 비해 성능이 좋음을 알 수 있음. 또한 병렬방식으로 학습하는 배깅보다 직렬 방식으로 가중치를 업데이트하는 부스팅이 성능이 더 좋음을 알 수 있음. 결과가 전체적으로 정상적이지 않아 XGB모델을 최종 모델로 선택하였음
    
- 실험 3. XGBClassifier
    <img width="889" alt="image" src="https://github.com/voo0o08/MachineLearning_Project/assets/155411941/6e3af301-bafe-4a26-9185-b4fa75b15646">

  좌측 : XGB의 중요한 피쳐 / 우측 : 상관계수 결과 
    
    **결과**
    
    XGB는 키가 중요한 피쳐임을 알아냈지만 상관계수와는 다른 결과임을 알 수 있음. 실제로 키는 BMI에 중요한 피쳐이기 때문에 XGB의 featrue importance가 적합하게 출력됨. 따라서 상관계수만으로 피쳐의 중요성을 판단해서는 안됨.
    

## 3. 새로운 데이터를 통한 예측
<img width="928" alt="스크린샷 2024-03-11 021223" src="https://github.com/voo0o08/MachineLearning_Project/assets/155411941/d13d8d6b-a17d-41bd-858d-53e56b5efd67">

나의 음주, 통학 수단, 식습관 등 17개의 feature를 입력하여 예측한 결과 정상 체중으로 잘 예측됨
</details>


<details>
<summary> 전진우 <a href="https://github.com/zeeenoo11" height="5" width="10" target="_blank">
	<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : 회귀를 통한 비만 예측 / ppt 20p~36p </summary>
<div markdown="1">
내용 채워주세요!!
</div>
</details>


| 자료 출처 | 링크 |
| ------------ | ------------- |
| Obesity or CVD risk (Classify/Regressor/Cluster) | https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster  |
| Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico | https://www.sciencedirect.com/science/article/pii/S2352340919306985 |

## 탈모 팀 소개 
<details>
<summary> 고우석 <a href="https://github.com/Gowooseo" height="5" width="10" target="_blank">
	<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : 탈모 예측 / ppt 37p~59p </summary>
<div markdown="1">
내용 채워주세요!!
</div>
</details>


- [김동현](https://github.com/KDT-05-Machine-Learning/KDT-05_ML_project/tree/main/DongHyun) : ?? <a href="https://github.com/??" height="5" width="10" target="_blank">
	<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : 탈모 예측 / ppt 60p~87p

<details>
<summary> 김동현 <a href="https://github.com/DongHyunKKK" height="5" width="10" target="_blank">
	<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : 탈모 예측 / ppt 60p~87p </summary>
<div markdown="1">
내용 채워주세요!!
</div>
</details>
