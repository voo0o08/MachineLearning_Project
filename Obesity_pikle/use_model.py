from joblib import load
import pandas as pd


# --------------------------------- 윤서 ---------------------------------------
# 모델 로딩
model_file = "obesity.pkl"
model = load(model_file)


obesity_list = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

# 로딩된 모델 확인
# print(model.classes_)
print('\n'*10)
if True:
    # data_list = datas.split(",")

    # d = datas.split(",")
    # ret = list(map(float, datas.split(",")))
    my_data = pd.DataFrame(
        [[0, -0.37025452, -1.18250493, 1, 1, 1, 2, 2, 0, 2, 0, 2, 0, 3, 0, 0, 0, 0, 1]],
        columns=[
            "Gender",
            "Age",
            "Height",
            "family_history_with_overweight",
            "FAVC",
            "FCVC",
            "NCP",
            "CAEC",
            "SMOKE",
            "CH2O",
            "SCC",
            "FAF",
            "TUE",
            "CALC",
            "MTRANS_Automobile",
            "MTRANS_Bike",
            "MTRANS_Motorbike",
            "MTRANS_Public_Transportation",
            "MTRANS_Walking",
        ],
    )

    # 입력된 정보에 해당하는 비만도 알려주기
    obesity = model.predict(my_data)

    proba = model.predict_proba(my_data)
    print(f"분류 방식 예측 결과 : {max(proba[0]*100):.2f}%의 확률로 {obesity_list[obesity[0]]}입니다.")
else:
    print("입력된 정보가 없습니다.")

# ------------------------------- 진우 ---------------------------------------
# 모델 로딩
model_file_r = "rfbs.pkl"
model_r = load(model_file_r)

# 모델 확인
# print(model_r)

# ['성별', '나이', '키', '가족력', '고열량 섭취 빈도', '채소 섭취 빈도', '하루 섭취 끼니 수', '간식 섭취 빈도',
# '물 섭취량', '활동량', '스마트폰 사용량', '음주 빈도', '이동 수단 : 자가용', '이동수단 : 대중교통', 'BMI']

# 예측
# -
# 1	0	21	1.520	0	0	3	3	2	3	3	0	2	0	1

# age 정규화
# scaler : scale_ ,     mean
# (array([5.97140927]), array([24.13378131]))
age = 25
height, weight = 170, 70
BMI = weight / ((height/100)**2)
scaled_age = (age - 24.13378131) / 5.97140927


jw_data = pd.DataFrame(
    [[1, scaled_age, height*0.01, 0, 0, 2, 3, 3, 3, 3, 1, 2, 0, 1]],
    columns=[
        "성별",
        "나이",
        "키",
        "가족력",
        "고열량 섭취 빈도",
        "채소 섭취 빈도",
        "하루 섭취 끼니 수",
        "간식 섭취 빈도",
        "물 섭취량",
        "활동량",
        "스마트폰 사용량",
        "음주 빈도",
        "이동 수단 : 자가용",
        "이동수단 : 대중교통",
    ],
)
jw_obesity = model_r.predict(jw_data)
print(f"회귀 방식 예측 결과 : 87.57%의 확률로 {jw_obesity[0]:.2f}입니다.")
print(f'실제 BMI : {BMI:.2f}')

