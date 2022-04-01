# 도미 데이터셋
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터셋
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 그래프를 그리기 위한 맷플롯 라이브러리
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight) # 도미의 길이별 무게
plt.scatter(smelt_length, smelt_weight) # 빙어의 길이별 무게
plt.xlabel('length')                    # x축 = 길이
plt.ylabel('weight')                    # y축 = 무게
plt.show()
# 산점도 그래프가 일직선에 가까운 형태로 나타나는 경우 = 선형(linear)


# 첫번째 머신러닝 프로그램
length = bream_length+smelt_length
weight = bream_weight+smelt_weight                      # 도미와 빙어의 리스트를 각각 길이, 무게로 하나로 합침

fish_data = [[l, w] for l, w in zip(length, weight)]    # zip()함수: 나열된 리스트 각각에서 하나씩 원소를 꺼내 반환.
print(fish_data)                                        # zip()과 리스트 내포 구문을 이용해 길이와 무게 리스트를 2차원 리스트로 만듦.

fish_target = [1]*35 + [0]*14                           # 위에서 도미와 빙어 데이터를 순서대로 나열했기 때문에 앞에 35개는 1(도미), 14개는 0(빙어)
print(fish_target)                                      # 찾는 대상 = 1 = 도미

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()                             
kn.fit(fish_data, fish_target)                          # fish데이터를 학습
kn.score(fish_data, fish_target)                        # fish데이터의 정확도 계산


plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')                        # 예측하고자하는 데이터(30, 600)을 맷플롯에 찍음.
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.predict([[30, 600]])                                 # (30, 600)인 값의 예상값
print(kn._fit_X)
print(kn._y)
kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
print(35/49)                                            