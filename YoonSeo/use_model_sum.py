from joblib import load
import pandas as pd

model_file = "../model/obesity.pkl"
model_file_r = "rfbs.pkl"

# 모델 로딩
model = load(model_file)
model_r = load(model_file_r)

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
print(model_r)

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
    print(f"{round(max(proba[0]), 2)}% {obesity_list[obesity[0]]}입니다.")
else:
    print("입력된 정보가 없습니다.")