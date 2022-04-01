# [문제 2] kaggle등에서 관심 분야 데이터셋(csv파일)을 구해서 3가지 모델로 예측(회귀)을 하고 비교를 하시오
    #   2.1 단순선형회귀 
    #   2.2 다항선형회귀 
    #   2.3 다중선형회귀 - 다중 선형 회귀는 데이터시트 상 불가능해서.. 못했습니다..
    #   2.4 모델 비교(맨 아래)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

#데이터 재가공
study_grade = pd.read_csv('week3/Student_Grade.csv')
df = pd.DataFrame(study_grade)

study_time = df.iloc[:, 0]
study_grade = df.iloc[:, 1]

study_time = np.array(study_time)
study_grade = np.array(study_grade)

# 훈련세트 / 테스트세트 분리
train_input, test_input, train_target, test_target = train_test_split(
    study_time, study_grade, random_state=40
)

train_input = train_input.reshape(-1, 1)    #2차원 배열로 만들어주는 과정
test_input = test_input.reshape(-1, 1) #2차원 배열로 만들어주는 과정

lr = LinearRegression()

    #다항 회귀

train_poly = np.column_stack((train_input ** 2, train_input)) #제곱한걸 나란히 붙여서 2차원 배열로
test_poly = np.column_stack((test_input ** 2, test_input))

lr.fit(train_poly, train_target)

time_set2 = 100
result_grades2 = lr.predict([[time_set2**2, time_set2]])

print(lr.coef_, lr.intercept_)  #[-1.64574230e-05  2.84394754e-02] 0.870738362867649

print(f"공부 시간이 {time_set2}분 일 때, 예상 성적은 {result_grades2}입니다.")


#구간(50분~500분)별 직선을 찍기 위해서 구간의 정수 배열을 만들어
point = np.arange(50, 500)
plt.scatter(train_input, train_target)  
plt.plot(point, -1.64574230e-05*point**2 + 2.84394754e-02*point + lr.intercept_)
# plt.ylim([0, 15])
plt.scatter(time_set2, result_grades2, marker='^')
plt.xlabel("Study Time In Minutes")
plt.ylabel("Credit Grade")
plt.show()

#비교
# 단순 선형 회귀는 1차 함수(y=ax+b)로 이루어져있어서 직선으로 즉, x와 y가 일정하게 증가 혹은 감소하는 데이터를
#     예측할 때 사용하기 좋다.
# 다항 회귀는 2차함수, 즉 곡선으로 이루어져 있다. 
# 다중 회귀는 여러 개의 특성을 사용한 선형 회귀를 말한다.
#     예를 들어 특성이 2개라면 평면을 학습한다. 