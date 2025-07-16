print("hello world!")
myInt=4
myReal=2.5
myChar='a'
myString='hello'
print(myInt)
print(myString)
a=2
b=3
print('구구단 {0}*{1}={2}'.format(a,b,a*b))
number=input("숫자를 입력하세요:")
print(number)
a=input()

str_input= input("숫자 입력:")
num_input=float(str_input)
print(num_input, "inch")
print((num_input*2.54), "cm")

a=input("숫자입력:")
b=input("두번째 숫자 입력:")
print(float(a)+float(b))

y=input("년도입력:")
print(2025-float(y))

#이름 첫글자 또는 마지막글자 출력하기
name=input("이름을 입력:")
print(str(name)[0])
print(str(name)[-1])

#전화번호 분리하기
num=input("전화번호입력:")
print(int(num[0:2]),int(num[3:7]),int(num[8:11]))

number=input("정수 입력:")
last_character=number[-1]
last_number=int(last_character)

if last_number==0 \
    or last_number==2\
    or last_number==4\
    or last_number==6\
    or last_number==8:
    print("짝수입니다5")


nu=input("정수 입력:")
if nu>10:
    print("10보다 큰 수 입니다")

x=input("입력:")
y=input("입력:")
if int(x)>int(y):
   print("x가 더큼")
k=input("입력:")
if int(k)//3 == 0:
   print("3의배수")

pw=input("비번입력:")
if pw==1234 :
   print("비밀번호 일치")


#반복문

fruits=['사과','바나나','포도']
list=[]
for fruit in fruits:
   list.append(fruit)
print(list)
if x%2 == 0:
    for x in range(5):
        print('x')


total = 0
for i in range(1, 101):
    if i % 3 == 0:
        total += i

print("1부터 100까지 3의 배수의 합:", total)

text = "Hello, world!"

for char in text:
    print(char)

total = 0
for i in range(1, 101):
    if i % 3 == 0:
        total += i

print("1부터 100까지 3의 배수의 합:", total)

#리스트의 생성과 추출
a=[10,20,30,40,50,60,70,80,90,100]
x=a.index(30)
print(x)

a.remove(90)
print(a)

a.clear()
print(a)

Fruits={'apple','banana']
Fruits.append('orange')

Colors=['red','blue']
Colors.insert(0,'green')

Colors.extend(['yellow','purple'])

numbers = []

for i in range(5):
    numbers.append(i)

print(numbers)

numbers = []

for i in range(5):
    i=i*2
    numbers.append(i)

    print(numbers)