import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[1,4,9,16], label='price')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-10,10,100)
y=x**3
plt.plot(x,y)
plt.xscale('symlog')
plt.show()

x=np.linspace(0,5,100)
y=np.exp(x)

plt.plot(x,y)
plt.yscale('log')
plt.show()

plt.plot([1,2,3,4],[1,4,9,16], 'bo--')
#마커+점선 실선은 bo- 이 때 b는 blue
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x=np.arange(0,2,0.2)
plt.plot(x,x,'r--',x,x**2,'bo',x,x**3,'g-.')
plt.show()


x=np.logspace(-1,2,100)
y=x**2
plt.xscale('log')
plt.yscale('log')
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('loglog')
plt.show()


x=np.linspace(0,10,30)
y1=np.logspace(1,0,30)
y2=np.logspace(1,0.8,30)
y3=np.logspace(1,0.5,30)
plt.yscale('log')
plt.scatter(x, y1, marker='v', s=80, facecolor='green',
edgecolor='black')
plt.scatter(x, y2, marker='o', s=60, facecolor='red',
edgecolor='black')
plt.scatter(x, y3, marker='D', s=30, facecolor='blue',
edgecolor='black')
plt.title('CMS')
plt.show()

x=np.linspace(0,2*np.pi,300)
y1=np.sin(x)
y2=np.cos(x)
plt.xlabel('x_rad')
plt.ylabel('value')
plt.title('sin vs cos')
plt.plot(x,y1,label='sin')
plt.plot(x,y2,label='cos')
plt.legend(['sin','cos'])
plt.show()

#산점도 색상매핑
x=np.random.rand(200)
y=np.random.rand(200)
sc=plt.scatter(x,y, c=y,cmap='plasma',s=30)
plt.colorbar(sc,label='y_value')
plt.title('Scatter Plot')
plt.show()

#막대그래프
months=['jan','feb','mar','apr','may','jun']
sales=[23,19,31,29,38,45]
plt.bar(months,sales)
plt.grid(axis='y',linestyle=':')
for m,s in zip(months,sales):
    plt.text(m,s+1,str(s),ha='center',va='bottom')
    plt.ylabel('sales')
plt.show()

#원형차트
labels=['a','b','c','d']
sizes=[48,25,20,15]
explode=[0.05,0,0,0]
plt.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%', startangle=90)
plt.title('Market Share')
plt.axis('equal')
plt.show()

#히스토그램
data=np.random.normal(size=20000)
plt.hist(data,bins=200,alpha=0.7)
plt.title('Histogram')
plt.xlim(-4,4)
plt.savefig('hist_norm.png',dpi=120)
plt.show()

#데이터의 다항식근사
x=[1,2,3,1.5,4,2.5,6,4,3,5.5,5,2,]
y=[3,4,8,4.5,10,5,15,9,5,16,13,3]
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')

linear_model=np.polyfit(x,y,2) #2차 곡선에 맞춤
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,7)
plt.plot(x_s,linear_model_fn(x_s),color="green")
plt.show()


x=np.array([1,2,3,4])
y=np.array([2,3,5,4])
plt.scatter(x,y)
plt.grid(axis='y',linestyle=':')

linear_model=np.polyfit(x,y,1) #1차 곡선에 맞춤
linear_model_fn=np.poly1d(linear_model)
x_s=np.linspace(x.min(),x.max(),10)
plt.plot(x_s,linear_model_fn(x_s),color="green")
plt.show()

# 1) 데이터
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 5, 4])

# 2) 1차 직선 적합
coef = np.polyfit(x, y, deg=1)        # [기울기, 절편]
line = np.poly1d(coef)                # 함수 형태로 래핑
print("계수:", coef)
print("x = 2.5 에서 예측 y :", line(2.5))

# 3) 그래프 (실제 점 + 피팅 직선)
plt.figure()
plt.scatter(x, y, color='red', label='실측 데이터')

# 직선을 부드럽게 그리기 위한 x‑grid
x_fit = np.linspace(x.min(), x.max(), 10)
plt.plot(x_fit, line(x_fit), color='green', label='1')


#예제2

rng = np.random.default_rng(0) #난수생성
x = np.linspace(-3, 3, 20)
y = x**2 + rng.normal(scale=1.0, size=x.size)
plt.figure()
plt.scatter(x, y, color='red', label='실측 데이터')


coef = np.polyfit(x, y, deg=3)        # [기울기, 절편]
line = np.poly1d(coef)                # 함수 형태로 래핑
print("계수:", coef)
print("x = 2.5 에서 예측 y :", line(2.5))

x_fit = np.linspace(x.min(), x.max(), 100)
plt.plot(x_fit, line(x_fit), color='green', label='1')
plt.show()

#pandas 데이터 분석 라이브러리
import pandas as pd
list1=[1,2,3]
df_list=pd.DataFrame(list1,columns=['col'])
print(df_list)

array1=np.array([1,2,3])
df_array1=pd.DataFrame(array1,columns=['col'])
print(df_array1)
#2차원데이터
cols=['col1','col2','col3']
list2=[[1,2,3],[11,12,13]]
df_list2=pd.DataFrame(list2,columns=cols)
print(df_list2)

cols=['국어','수학','영어','과학','사회']
indexes=['태현','준수','가은']
lists=[[83,68,92,55,85],[40,95,64,87,77],[65,87,58,92,72]]
dfs=pd.DataFrame(lists,columns=cols,index=indexes)
print(dfs)
dfs.to_csv('./results.csv')


#3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 5, 4])

# 1차 직선 적합
coef = np.polyfit(x, y, deg=1)
line = np.poly1d(coef)

# 예측값과 오차 계산
y_fit = line(x)
residual = y - y_fit

# DataFrame 생성
df = pd.DataFrame({
    'x': x,
    'y': y,
    'y_fit': y_fit,
    'residual (y - y_fit)': residual
})

# 출력
print(df)
