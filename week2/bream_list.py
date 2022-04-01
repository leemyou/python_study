bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 머신러닝 프로그램
lenght = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(lenght, weight)]
fish_target = [1] * 35 + [0] * 14

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)     # fit() = training method
kn.score(fish_data, fish_target)     # score() = evaluate method(0~1)
kn.predict([[30,600]])       # 길이 30, 무게 600인 생선은 도미일까 광어일까?

kn49 = KNeighborsClassifier(n_neighbors=49)    # 참고 데이터를 49개로 한 모델
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
# Question - 기본값 5~49 중 정확도가 1.0 이하인 이웃의 수는?
for n in range(5,50):
  kn.n_neighbors = n
  score = kn.score(fish_data, fish_target)
  if score < 1:
    print(n, score)