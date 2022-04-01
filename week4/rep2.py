# 2. Numpy를 이용해 구현하기
import numpy as np
import matplotlib.pyplot as plt

#데이터 준비
s_time = np.array([2, 4, 6, 8])    # x
s_grade=np.array([81, 93, 91, 97]) # y

a = 0 #모두 0으로된 배열 생성
b = 0

lr = 0.005 #학습률: 한번 학습할 때 얼만큼 변화를 주는지에 대한 상수
#0.001 0.005 0.1 각각 대입해보았으나 0.005일 때 가장 올바른 것 같다
epochs = 2001 #에포크: 학습의 반복 횟수


#직선의 방정식은 y = ax + b라고 가정
# 예측값과 실제값의 차이 = error
for i in range(0, epochs):
    y_prediction = a*s_time + b     #예측 데이터
    error = s_grade - y_prediction  #실제data - 예측data
    
    a_diff = ((-2 * (1/len(s_time))) * sum(s_time * error))
    b_diff = ((-2 * (1/len(s_time))) * sum(s_grade - y_prediction))
    
    a = a - lr * a_diff
    b = b - lr * b_diff
    
    
print(a)
print(b)



s_time = s_time.reshape(-1, 1)
s_grade = s_grade.reshape(-1, 1)

#선형 회기
plt.scatter(s_time, s_grade)
plt.xlabel('Time')
plt.ylabel('grade')
plt.plot(s_time, y_prediction)
    #굳이 없어도 되지만 있으면 보기 편한거
plt.grid(True)
plt.show()