from cgi import test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

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

# 길이 무게를 그냥 점으로 찍어봤음
plt.scatter(perch_length, perch_weight)
plt.xlabel("length")
plt.ylabel("weight")
# plt.show()

# 훈련 세트와 테스트 세트로 나눔
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

# print(train_input.shape, test_input.shape)

test_array = np.array([1,2,3,4])
print(test_array.shape)

test_array = test_array.reshape(2, 2)
print(test_array.shape)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

print(train_input.shape, test_input.shape)


#결정 계수 R^2
knr = KNeighborsRegressor()
# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)
# 회귀에서 score는 R^2값을 의미함 예측이 타겟에 가까워지면 1에 가까워짐
knr.score(test_input, test_target)
print(knr.score(test_input, test_target))

# 테스트 세트에 대한 예측을 만듦
test_prediction = knr.predict(test_input)
# 테스트 세트에 대한 평균 절댓값 오차를 계산
mae = mean_absolute_error(test_target, test_prediction)
print(test_target.shape)
print(test_prediction.shape)

print(mae)

#과대적합 과소적합
# print(knr.score(train_input, train_target))
# 이웃의 갯수를 3으로 설정합니다
knr.n_neighbors = 3
# 모델을 다시 훈련합니다
knr.fit(train_input, train_target)
# print(knr.score(train_input, train_target))
# print(knr.score(test_input, test_target))

# k-최근접 이웃 회귀 객체 생성
x  = np.arange(5, 45).reshape(-1, 1)

# k=1 ,5, 10일 때 예측 결과
for k in range(1,11): #1~10까지 연속적 range
    knr = KNeighborsRegressor(k) #for문 안으로 들어와서 쌓여야함
    
    knr.n_neighbors = k
    knr.fit(train_input, train_target)
    knr.score(test_input, test_target)
    #비어있는 리스트를 만들어서 for문을 돌고 끝나면 리스트 안에 값 누적 append
    # 이 끝나면 한번에 그려
    prediction = knr.predict(x)
    

#훈련 세트와 예측 결과 그래프 그리기
plt.scatter(train_input, train_target)
plt.plot(x, prediction) #여기서 x는 x축 y축과는 관련이 없는건가용? 위에 x 선언한거 있잖아;;
plt.title('n_neighbors = {}'.format(k))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()