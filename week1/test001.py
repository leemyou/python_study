import numpy
import csv

#매 시간 센서로부터 temp와  hum을 입력 받아야한다.
#temp = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20]
temp = []
#hum = [55, 60, 65, 70, 75, 80, 90, 85, 60, 60, 60]
hum = []

with open('week1/hum.csv') as hum :
    hum = list(csv.reader(hum))
with open('week1/temp.csv') as temp:
    temp = list(csv.reader(temp))

hum = list(map(int, hum[0]))
temp = list(map(int, temp[0]))

print(hum)
print(temp)


#에어컨(히터) 동작
for idx, hour in enumerate(range(8,18)):
    if temp[idx]>=26 or hum[idx]>=80:
        print(f"{hour}시, 온도: {temp[idx]}, 습도: {hum[idx]}, ON")
    else:
        print(f"{hour}시, 온도: {temp[idx]}, 습도: {hum[idx]}, OFF")


#h_temp_hour과 h_temp는 최고 온도가 달성 될 때마다 새로 지정을 해줘야함
h_temp = max(temp)
h_hum = max(hum)
h_temp_hour = temp.index(h_temp)+8
h_hum_hour = hum.index(h_hum)+8


#평균 온도 & 습도
avg_temp = numpy.mean(temp[idx])
avg_hum = numpy.mean(hum)

#하루가 끝나면 하루평균 온도와 습도 출력
print(f"하루 평균 온도는 {avg_temp} 이고, 하루 평균 습도는 {avg_hum} 입니다")
print(f"온도가 가장 높은 시간: {h_temp_hour} / 온도 :  {h_temp}")
print(f"습도가 가장 높은 시간: {h_hum_hour} / 습도 : {h_hum}")
