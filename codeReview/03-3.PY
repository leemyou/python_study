# 특성 공학과 규제

# 다중회귀: 여러 개의 특성을 사용한 선형회귀(도미의 길이, 무게, 두께를 이용하여 선형회귀를 진행..)
# 특성 공학: 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업

import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data')   # 이번엔 농어 데이터를 읽어와봅시다.
perch_full = df.to_numpy()      # print나 shape로 찍어보면 2차원 배열로 되어있다는 것을 알 수 있음.

import numpy as np

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
    perch_full, perch_weight, random_state=42)
# 여기까지는 기본 작업


# 변환기: 특성을 만들거나 전처리를 하기 위한 클래스
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(include_bias=False)   # include_bias의 기본값은 true인데, true로 하면 선형 방정식의 절편값을 지정을 해줘야 하는데 사이킷런에서는 자동으로 절편을 추가해주므로 필요없어서 false써줬음.
poly.fit([[2, 3]])                  # 새롭게 만들 특성 조합을 찾음 -> 변환기는 타깃 데이터 없이 입력 데이터를 변환.
print(poly.transform([[2, 3]]))     # 실제로 데이터를 변환함.
                                    # 절편을 위한 항이 제거되고 특성의 제곱과 특성끼리 곱한 항만 추가된다.

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# 9개의 특성이 각각 어떤 입력의 조합으로 만들었는지 알 수 있음.
poly.get_feature_names_out()


# 다중 회귀 모델 훈련하기
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)    # shape로 찍어보면 배열의 열의 갯수 = 55 = 특성의 갯수
test_poly = poly.transform(test_input)

lr.fit(train_poly, train_target)    # 학습
                                    # 특성의 갯수를 늘림 -> 훈련세트에 대해 거의 완벽하게 학습 -> 과대적합
                                    # => 규제를 걸어줘야 함.
                                    
        
                                    
# 규제
# 과대적합을 잡아내기 위한 릿지, 라소 모델
# 선형 회귀 모델의 경우 특성에 곱해지느 계수(가중치)의 크기를 작게 만들어주는 것
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)       # 훈련세트로 학습한 변환기를 사용하여 테스트 세트까지 변환해야함.





# 릿지
from sklearn.linear_model import Ridge
ridge = Ridge()

ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))      # 0.9896101671037343
print(ridge.score(test_scaled, test_target))        # 0.9790693977615391

# 릿지와 라쏘 모델을 사용할 때 규제의 양을 임의로 조절할 수 있음(하이퍼파라메터)
# alpha값이 크면 규제 강도가 세지므로 계수값을 줄이고 과소적합 되도록 유도할 수 있음

import matplotlib.pyplot as plt

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
    
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
    # => 적절한 alpha값은 두 그레프가 가장 가깝고 테스트 세트의 점수가 가장 높은 0.1

# 최적의 알파값으로 모델을 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)

print(ridge.score(train_scaled, train_target))      # 0.9903815817570365
print(ridge.score(test_scaled, test_target))        # 0.9827976465386884






# 라쏘
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)

# 최적의 알파값 찾기
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 모델을 만듭니다
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 라쏘 모델을 훈련합니다
    lasso.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
    
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
     # => 라쏘 모델은 대략 떨어지기 직전 값이 최적의 알파값...인가?
     # 그래프의 왼쪽은 과대적합, 오른쪽은 과소적합.

# 최적의 알파값으로 라쏘 모델 훈련
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)

print(lasso.score(train_scaled, train_target))      # 0.9888067471131867
print(lasso.score(test_scaled, test_target))        # 0.9824470598706695

# 라쏘모델은 계수값을 아예 0으로 만들 수 있음