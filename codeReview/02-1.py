# 평가에 사용하는 데이터 = 테스트 세트
# 훈련에 사용되는 데이터 = 훈련 세트

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]     # 샘플
fish_target = [1]*35 + [0]*14       # 처음 35개를 훈련세트로 나머지 14개를 테스트 세트로 이용하려고 0,1로 나눠줌


# 파이썬은 데이터 슬라이싱을 할 수 있음
train_input = fish_data[:35]
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]


# 갑자기 정확도가 0이 되었다. 왜?
# 샘플링 편향(위에 fish_data는 앞에 35개는 빙어, 14개는 도미로 나눠져있음. 그냥 앞뒤로 테스트 세트를 나누면 
# 앞에서 학습한 빙어에 대한 데이터는 하나도 들어있지 않아서 빙어없이 빙어를 분류하게 되었음. 
# 그래서 훈련 결과가 그지같이 나왔다.)이 일어났기 때문이다.
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)   # 0.0


# 샘플링 편향을 막아주려면 -> 데이터를 섞어주면 된다.
# 넘파이에서 2차원 배열로 바꿔준 후에 random.shuffle()함수를 이용해주면 된다.
import numpy as np
input_arr = np.array(fish_data)     # input_arr은 2차원 배열임.
target_arr = np.array(fish_target)

np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)        

kn.fit(train_input, train_target)
kn.score(test_input, test_target)   # 이젠 잘 섞임!!