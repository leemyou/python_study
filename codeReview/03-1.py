# KNN
# K-최근접 이웃 회귀
# 방금까지 했던건 KNN 분류, 지금 하는건 회귀

import numpy as np
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
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

# KNN 회귀 -> KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()

# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)

from sklearn.metrics import mean_absolute_error
# 테스트 세트에 대한 예측을 만듭니다
test_prediction = knr.predict(test_input)
# 테스트 세트에 대한 평균 절댓값 오차를 계산합니다
mae = mean_absolute_error(test_target, test_prediction)
print(mae)   # 19.157142857142862 -> 예측이 평균적으로 19g정도 타깃값과 다르다.


#과대적합 vs 과소적합.
# 훈련을 너무 많이 시키면 훈련 데이터에서는 높은 정확도를 보이지만,
# 테스트 데이터에서는 낮은 정확도를 보이는 것 => 과대적합
# => 과대적합일 경우 모델을 덜 복잡하게 만들어야함(KNN에서는 K값을 늘림.)

# 과소적합: 훈련세트보다 테스트 세트의 점수가 높거나 두 점수 모두 너무 낮은 경우
# => 과소적합은 보통 모델이 너무 단순하여 훈련 세트에 적절히 훈련되지 않은 경우(데이터의 크기가 너무 작은 경우) 발생한다.
# -> KNN에서는 K값을 줄이는 겻이 해결방안.
    
# 이웃의 갯수를 3으로 설정합니다
knr.n_neighbors = 3
# 모델을 다시 훈련합니다
knr.fit(train_input, train_target)

print(knr.score(train_input, train_target)) # 0.9804899950518966
print(knr.score(test_input, test_target))   # 0.9746459963987609



# BUT, 가장 가까운 N개의 데이터를 기준으로 분류와 회귀를 진행하기 때문에
# 기존의 데이터보다 많이 크거나 작은 데이터를 분류하는데에는 어려움이 있음