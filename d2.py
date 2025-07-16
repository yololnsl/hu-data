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

#데이터의 다항식 근사
x=[1,2,3,1.5,4,2.5,6,4,3,5.5,2,4]
y=[3,4,8,4.5,10,5,15,9,5,16,13,3]
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')

# 근사모듈
linear_model=np.polyfit(x,y,1)
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,7)
plt.plot(x_s,linear_model_fn(x_s)),color=import matplotlib.pyplot as plt
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
print(linear_model)

#데이터의 다항식근사
x=[1,2,3,1.5,4,2.5,6,4,3,5.5,5,2,]
y=[3,4,8,4.5,10,5,15,9,5,16,13,3]
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')

linear_model=np.polyfit(x,y,1)
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,7)
plt.plot(x_s,linear_model_fn(x_s),color="green")
plt.show()
