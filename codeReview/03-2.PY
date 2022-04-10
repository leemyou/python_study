# 선형 회귀
# KNN의 단점을 보완할 수 있음

import numpy as np
import matplotlib.pyplot as plt

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )

from sklearn.model_selection import train_test_split

# 훈련 세트와 테스트 세트로 나눕니다
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
# 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)



# 선형회귀 모델 LinearRegression
# 데이터를 기준으로 직선으로 선을 그어서 데이터가 어떤건지 예측함.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# 선형 회귀 모델 훈련
lr.fit(train_input, train_target)

# lr객체의 안에는 coef_(기울기 = 가중치), intercept_(절편) 속성이 있음
print(lr.coef_, lr.intercept_)      # [39.01714496] -709.0186449535477

print(lr.score(train_input, train_target))      #0.939846333997604
print(lr.score(test_input, test_target))        #0.8247503123313558 KNN에 비해 상당히 높은 정확도를 보여준다.



# 다항회귀
# 선형회귀와는 다르게 데이터를 기준으로 곡선으로 선을 그어서 데이터를 예측함.

# 대신 2차 방정식의 그래프를 그려야하기 때문에 특성을 제곱한 항이 훈련 세트에 추가되어야함
# 원래 특성으로 있었던 길이를 제곱하여 왼쪽 열에 추가했기 때문에 shape찍어보면 열이 2개로 늘어나 있음것이다.
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)        # 학습

print(lr.coef_, lr.intercept_)          # [  1.01433211 -21.55792498] 116.0502107827827
                                        # 무게 = 1.01 * 길이^2 - 21.6 * 길이 + 116.05


# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다
# 곡선 = 무수히 많은 직선들이 모이고 모여서 생긴거라서 배열 나눠준다고 생각하면 편할 듯.
point = np.arange(15, 50)
# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
# 15에서 49까지 2차 방정식 그래프를 그립니다
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly, train_target))   # 0.9706807451768623
print(lr.score(test_poly, test_target))     # 0.9775935108325122 정확도가 더 높아진 것을 알 수 있다.

# 딱히 어떤 회귀가 더 좋다라는게 아니라 그래프 모양을 보고 그때그때 맞는 회귀모델을 사용하면 됨