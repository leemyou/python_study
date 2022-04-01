# 1. Scikitlearn 라이브러리 이용하기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#데이터 준비

s_time = np.array([2, 4, 6, 8])      # x
s_grade=np.array([81, 93, 91, 97]) # y

s_time = s_time.reshape(-1, 1)
s_grade = s_grade.reshape(-1, 1)


lr = LinearRegression()
lr.fit(s_time, s_grade)

#fit() 메소드로 모델을 생성하면 pridict함수를 사용하여 예측값을 얻을 수 있다.
predict = lr.predict(s_time)
    #time = 2 일 때 예측값
time_model = lr.predict([[2]])[0][0]
print(time_model)



#선형 회기
plt.scatter(s_time, s_grade)
plt.xlabel('Time')
plt.ylabel('grade')
plt.plot(s_time, lr.predict(s_time))
    #굳이 없어도 되지만 있으면 보기 편한거
plt.grid(True)
plt.show()