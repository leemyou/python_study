import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#데이터 재가공
purchase_target = pd.read_csv('week2/Social_Network_Ads.csv')
df = pd.DataFrame(purchase_target)

pur_age = [age for age in purchase_target['Age']]
pur_salary = [salary for salary in purchase_target['EstimatedSalary']]
pur_gender = [gender for gender in purchase_target['Gender']]

#남자 여자 분리
#print(purchase_target[purchase_target['Gender'] == 'Male'])
#if purchase_target[purchase_target['Gender'] == 'Male']:
#    df[df['Gender'] == 'Male']
#else :
#    print("2")


purchase_data = np.column_stack((pur_gender, pur_age, pur_salary))

#정답 준비
purchase_target = [purchased for purchased in purchase_target['Purchased']]



#훈련 세트 / 테스트 세트 분리
train_input, test_input, train_target, test_target = train_test_split(
    purchase_data, purchase_target, random_state=100
)
#print(train_input)
#print("================================================================")
#print(test_input)
#print(train_input.shape, test_input.shape)
#print(test_target)




#ml
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
#print(kn.predict([[22, 10000]]))




#matplot
distances, indexes = kn.kneighbors([['Female', 50, 80000]])

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std

new = (['Female', 50, 80000] - mean) / std

kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)

#print(kn.predict([new]))
if kn.predict([new])==1:
    print("사용자는 물품을 구매할 것이다.")
else :
    print("사용자는 물품을 구매하지 않을 것이다.")

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('age')
plt.ylabel('salary')
plt.show()
