import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


#  농어의 길이와 무게 데이터
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트 & 데이터 세트
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

train_input = train_input.reshape(-1, 1) #x값이 하나인지 2개인지 3개인지 모를 때에는 3개를 묶어서 한번에 묶어서(묶어주는게 하나의 단위) 2차원 배열이 될 수 있음
test_input = test_input.reshape(-1, 1)

array_train = []
array_test = []

#k(x축)에 따른 train, test -> R2(y축)의 변화 그래프

for k in range(1, 5):
    knr = KNeighborsRegressor(k)
    knr.n_neighbors = k
    
    knr.fit(train_input, train_target)
    
    # R2값
    train_r2 = knr.score(train_input, train_target)
    test_r2 = knr.score(test_input, test_target)
    
    array_train.append(train_r2)
    array_test.append(test_r2)
    # prediction = knr.predict()

print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

#plt.scatter(k범위, R2범위값)

plt.scatter(range(1, 5), array_train)
plt.scatter(range(1, 5), array_test)
plt.plot(range(1, 5), array_train)
plt.plot(range(1, 5), array_test)
plt.title('n_neighbors = {}'.format(k))
plt.xlabel("k값")
plt.ylabel("R2")
plt.show()