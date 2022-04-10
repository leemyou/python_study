#도미 데이터 준비하기
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 3]
                
# 빙어 데이터 준비
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length+smelt_length
weight = bream_weight+smelt_weight

# 2차원 배열로 만들어줌
fish_data = [[l, w] for l, w in zip(length, weight)]

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)      #KNN으로 학습
kn.score(fish_data, fish_target)    #학습한 결과(정확도)출력


# KNN -> 가까운 몇 개의 데이터를 참고할지 선택(하이퍼파라메터)할 수 있음
# 기준은 n_neighbors=? 매개 변수로 바꿀 수 있음.
# 숫자를 너무 높게 설정하면 정확도에 안좋은 영향을 끼칠 수 있음 -> 적당한 갯수를 찾는 것도 중요(근데 보통 그냥 기본값 5사용)