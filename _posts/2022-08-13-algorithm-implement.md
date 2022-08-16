---
layout: post
title: Implement Algorithm
category: Algorithm
tag: Algorithm
---

해당 게시물은 [이것이 코딩 테스트다 with 파이썬](https://www.hanbit.co.kr/store/books/look.php?p_code=B8945183661)을 바탕으로 언제든지 블로그에 들어와서 보기위해 작성되었습니다.





# 구현(Implement)

코딩 테스트에서 **<span style="background-color: #fff5b1">구현(Implementation)</span>** 이란 **<span style="color:red">"머릿속에 있는 알고리즘을 소스코드로 바꾸는 과정"</span>** 입니다.

* 흔히 알고리즘 대회에서 구현 유형의 문제란?
    * 풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제를 지칭합니다.
* 구현 유형의 예시는 다음과 같습니다.
    * 알고리즘은 간단한데 코드가 지나칠 만큼 길어지는 문제
    * 실수 연산으 다루고, 특정 소수점 자리까지 출력해야 하는 문제
    * 문자열이 입력으로 주어졌을 때 한 문자 단위로 끊어서 리스트에 넣어야 하는(파싱을 해야 하는 문제)
    * 적절한 라이브러리를 찾아서 사용해야 하는 문제
* 구현 파트의 대표적인 유형
    * **<span style="color:red">완전 탐색(Brute-Force)</span>** : **<span style="background-color: #fff5b1">모든 경우의 수를 주저 없이 다 계산하는 해결 방법을 의미</span>**
    * **<span style="color:red">시뮬레이션(Simulation)</span>** : **<span style="background-color: #fff5b1">문제에서 제시한 알고리즘을 한 단계씩 차례대로 직접 수행해야 하는 문제 유형을 의미</span>**

<br>

모든 알고리즘 문제에 적용되는 이야기지만, 특히 구현에서는 입력 변수의 표현 범위와 제한된 메모리, 제한된 채점시간을 주의 해야한다.
* **<span style="background-color: #fff5b1">입력 변수의 표현 범위</span>**
    * 파이썬은 직접 자료형을 지정할 필요가 없으며 매우 큰 수의 연산 또한 기본으로 지원한다.
    * 하지만 다른 언어의 경우 int형을 벗어나는 경우 long long형 등 자료형을 고려해야 한다.
* **<span style="background-color: #fff5b1">제한된 메모리</span>**
    * 보통의 문제의 경우 128MB를 제한으로 둡니다.
        * 데이터의 개수(리스트의 길이):1,000 = 메모리 사용량: 약 4KB
        * 데이터의 개수(리스트의 길이):1,000,000 = 메모리 사용량: 약 4MB
        * 데이터의 개수(리스트의 길이):10,000,000 = 메모리 사용량: 약 40KB
    * 즉, 128MB의 메모리 제한이 있는 곳에선 32,000,000개의 데이터까지 쓸 수 있다는 말이 됩니다.
* **<span style="background-color: #fff5b1">제한된 채점시간</span>**
    * 보통의 문제의 경우 시간 제한은 1초로 지정됩니다.
    * 파이썬 3.7을 기준으로 1초에 2000만번(20,000,000)의 연산을 수행한다고 가정하고 문제를 풀면 실행 시간 제한에 안정적입니다.
    * 시간 제한이 1초이고, 데이터의 개수가 100만 개인 문제가 있다면 일반적으로 시간 복잡도 O(NlogN) 이내의 알고리즘을 이용하여 문제를 풀어야 합니다.
        * 실제로 N = 1,000,000일 때 NlogN은 약 20,000,000 입니다.
    
<br>

정리하면 알고리즘 문제를 풀 때, **<span style="color:red">시간 제한과 데이터의 개수를 먼저 확인한 뒤에 이 문제를 어느 정도의 시간복잡도의 알고리즘으로 작성해야 풀 수 있을 것인지 예측할 수 있어야 합니다.</span>**

<br>





# 상하좌우

## 문제 정의

* 여행가 A는 N × N 크기의 정사각형 공간 위에 서 있다. 이 공간은 1 × 1 크기의 정사각형으로 나누어져 있다. 가장 왼쪽 위 좌표는 (1, 1) 이며, 가장 오른쪽 아래 좌표는 (N, N)에 해당한다. 여행가 A는 상, 하, 좌, 우 방향으로 이동할 수 있으며, 시작 좌표는 항상 (1, 1)이다. 우리 앞에는 여행가 A가 이동할 계획이 적힌 계획서가 놓여 있다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/184776165-bb39a54b-9484-4792-8f35-3c60e2cfb919.png" style="zoom:20%;">
</p>

* 계획서에는 하나의 줄에 띄어쓰기를 기준으로 하여 L, R, U, D 중 하나의 문자가 반복적으로 적혀있다. 각 문자의 의미는 다음과 같다.
    * L: 왼쪽으로 한 칸 이동
    * R: 오른쪽으로 한 칸 이동
    * U : 위로 한 칸 이동
    * D: 아래로 한 칸 이동

이때 여행가 A가 N X N 크기의 정사각형 공간을 벗어나는 움직임은 무시된다. 예를 들어 (1, 1)의 위치에서 L혹은 U를 만나면 무시된다. 다음은 N = 5인 지도와 계획서이다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/184776944-670258a7-de88-4b43-8ea9-fe16173102c9.png" style="zoom:20%;">
</p>

* 이 경우 6개의 명령에 따라서 여행가가 움직이게 되는 위치는 순서대로 (1, 2), (1, 3), (1, 4), (1, 4), (2, 4), (3, 4)이므로, 최종적으로 여행가 A가 도착하게 되는 곳의 좌표는 (3, 4)이다. 다시 말해 3행 4열의 위치에 해당하므로 (3, 4) 라고 적는다. 계획서가 주어졌을 때 여행가 A가 최종적으로 도착할 지점의 좌표를 출력하는 프로그램을 작성하시오.

### 입력 조건
* 첫째 줄에 공간의 크기를 나타내는 N이 주어진다. (1≤ N ≤ 100)
* 둘째 줄에 여행가 A가 이동할 계획서 내용이 주어진다. (1 < 이동 횟수 100)

### 출력 조건
* 첫째 줄에 여행가 A가 최종적으로 도착할 지점의 좌표 (X, Y)를 공백으로 구분하여 출력한다.

### Test Case
```
[input]
5
R R R U D D

[output]
3 4
```



## 문제 해설

* 이 문제는 요구사항대로 구현하면 연산 횟수는 이동 횟수에 비례하게 됩니다. 예를 들어 이동 횟수가 N번인 경우 시간복잡도는 O(N)입니다. 따라서 이 문제의 시간복잡도는 매우 넉넉한 편 입니다.
* 이러한 문제는 **<span style="background-color: #fff5b1">일련의 명령에 따라서 개체를 차례대로 이동시킨다는 점</span>** 에서 **<span style="color:red">시뮬레이션(Simulation)</span>** 유형으로 분류되며 구현이 중요한 대표적인 문제 유형입니다.



## Solution

```py
# N 입력받기
n = int(input())
x, y = 1, 1
plans = input().split()

# L, R, U, D에 따른 이동 방향
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]
move_types = ['L', 'R', 'U', 'D']

# 이동 계획을 하나씩 확인
for plan in plans:
    # 이동 후 좌표 구하기
    for i in range(len(move_types)):
        if plan == move_types[i]:
            nx = x + dx[i]
            ny = y + dy[i]
    # 공간을 벗어나는 경우 무시
    if nx < 1 or ny < 1 or nx > n or ny > n:
        continue
    # 이동 수행
    x, y = nx, ny

print(x, y)
```

<br>

---





# 시각

## 문제 정의

* 정수 N이 입력되면 00시 00분 00초부터 N시 59분 59초까지의 모든 시각 중에서 3이 하나라도 포함되는 모든 경우의 수를 구하는 프로그램을 작성하시오. 예를 들어 1을 입력했을 때 다음은 3이 하나라도 포함되어 있으므로 세어야 하는 시각이다.
    * 00시 00분03초
    * 00시 13분 30초
* 반면에 다음은 3이 하나도 포함되어 있지 않으므로 세면 안 되는 시각이다.
    * 00시 02분 55초
    * 01시 27분 45초


### 입력 조건
* 첫째 줄에 정수 N이 입력된다. (0 N ≤ 23)

### 출력 조건
* 0시 00분 00초부터 N시 59분 59초 까지의 모든 시각 중에서 3이 하나라도 포함되는 모든 경우의 수를 출력한다.

### Test Case
```
[input]
5

[output]
11475
```



## 문제 해설

* 이 문제는 모든 시각의 경우를 하나씩 모두 세서 쉽게 풀 수 있는 문제입니다.
* 하루는 86,400초로, 00시 00분 00초부터 23시 59분 59초까지의 모든 경우는 86,400가지밖에 존재하지 않기 때문입니다.
    * 즉 경우의 수가 100,000개도 되지 않으므로 파이썬에서 문자열 연산을 이용해 3이 시각에 포함되어 있는지 확인해도 시간 제한 2초 안에 문제를 해결할 수 있습니다.
* 따라서 단순히 시각을 1씩 증가시키면서 3이 하나라도 포함되어 있는지 확인하면 됩니다.
* 이러한 유형은 **<span style="color:red">완전 탐색(Brute-Force)</span>** 유형이라고 불립니다.
    * 완전 탐색 알고리즘은 **<span style="background-color: #fff5b1">가능한 경우의 수를 모두 검사해보는 탐색 방법</span>** 입니다.
    * 완전 탐색 문제 또한 구현이 중요한 대표적인 문제 유형인데, 일반적으로 완전 탐색 알고리즘은 비효율적인 시간복잡도를 가지고 있으므로, 일반적으로 알고리즘 문제를 풀 때는 확인(탐색)해야 할 전체 데이터의 개수가 100만 개 이하일 때 완전 탐색을 사용하면 적절합니다.



## Solution

```py
# H를 입력받기
h = int(input())

count = 0
for i in range(h + 1):
    for j in range(60):
        for k in range(60):
            # 매 시각 안에 '3'이 포함되어 있다면 카운트 증가
            if '3' in str(i) + str(j) + str(k):
                count += 1

print(count)
```

<br>

---





# 왕실의 나이트

## 문제 정의

* 행복 왕국의 왕실 정원은 체스판과 같은 8 x8 좌표 평면이다. 왕실 정원의 특정한 한 칸에 나이트가 서 있다. 나이트는 매우 충성스러운 신하로서 매일 무술을 연마한다.
* 나이트는 말을 타고 있기 때문에 이동을 할 때는 L자 형태로만 이동할 수 있으며 정원 밖으로는 나갈 수 없다. 나이트는 특정한 위치에서 다음과 같은 2가지 경우로 이동할 수 있다.
    1. 수평으로 두 칸 이동한 뒤에 수직으로 한 칸 이동하기
    2. 수직으로 두 칸 이동한 뒤에 수평으로 한 칸 이동하기

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/184780719-fda2d9a1-fa1d-4c5a-9053-a9235a1fd6ae.png" style="zoom:20%;">
</p>

* 이처럼 8×8 좌표 평면상에서 나이트의 위치가 주어졌을 때 나이트가 이동할 수 있는 경우의 수를출력하는 프로그램을 작성하시오. 이때 왕실의 정원에서 행 위치를 표현할 때는 1부터 8로 표현하며, 열 위치를 표현할 때는 a부터 로 표현한다.


### 입력 조건
* 입력 조건 첫째 줄에 8×8 좌표 평면상에서 현재 나이트가 위치한 곳의 좌표를 나타내는 두 문자로 구성된 문자열이 입력된다. 입력 문자는 a1 처럼 열과 행으로 이뤄진다.

### 출력 조건
* 첫째 줄에 나이트가 이동할 수 있는 경우의 수를 출력하시오.

### Test Case
```
[input]
a1

[output]
2
```





## My Solution

```py
from sys import stdin

location = stdin.readline()
row = int(location[1])
col = int(ord(location[0])) - int(ord('a')) + 1

movable = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

count = 0
for move in movable:
    move_row = row + move[0]
    move_col = col + move[1]
    
    if move_row < 1 or move_col < 1 or move_row > 8 or move_col > 8:
        continue
    
    count += 1

print(count)
```

<br>





## Solution

```py
# 현재 나이트의 위치 입력받기
input_data = input()
row = int(input_data[1])
column = int(ord(input_data[0])) - int(ord('a')) + 1

# 나이트가 이동할 수 있는 8가지 방향 정의
steps = [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1)]

# 8가지 방향에 대하여 각 위치로 이동이 가능한지 확인
result = 0
for step in steps:
    # 이동하고자 하는 위치 확인
    next_row = row + step[0]
    next_column = column + step[1]
    # 해당 위치로 이동이 가능하다면 카운트 증가
    if next_row >= 1 and next_row <= 8 and next_column >= 1 and next_column <= 8:
        result += 1

print(result)
```

<br>

* 나이트가 이동할 수 있는 경로를 하나씩 확인하여 이동하면 됩니다. 다만, 8×8 좌표 평면을 벗어나지 않도록 꼼꼼하게 검사하는 과정이 필요합니다.
* 나이트의 이동 경로를 steps 변수에 넣는다면, 이 2가지 규칙에 따라 steps [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1)]로 값을 대입할 수 있습니다.
* 나이트의 현재 위치가 주어지면 현재 위치에서 이동 경로를 더한 다음, 8 × 8 좌표 평면에 있는지 확인하면 되는데, 이 과정은 반복문으로 처리할 수 있습니다.

<br>

---





# 게임 개발

## 문제 정의

* 현민이는 게임 캐릭터가 맵 안에서 움직이는 시스템을 개발 중이다. 캐릭터가 있는 장소는 1×1크기의 정사각형으로 이뤄진 N×M 크기의 직사각형으로, 각각의 칸은 육지 또는 바다이다. 캐릭터는 동서남북 중 한 곳을 바라본다.
* 맵의 각 칸은 (A, B)로 나타낼 수 있고, A는 북쪽으로부터 떨어진 칸의 개수, B는 서쪽으로부터 떨어진 칸의 개수이다. 캐릭터는 상하좌우로 움직일 수 있고, 바다로 되어 있는 공간에는 갈 수 없다.캐릭터의 움직임을 설정하기 위해 정해 놓은 매뉴얼은 이러하다.
    1. 현재 위치에서 현재 방향을 기준으로 왼쪽 방향(반시계 방향으로 90도 회전한 방향)부터차례대로 갈 곳을 정한다.
    2. 캐릭터의 바로 왼쪽 방향에 아직 가보지 않은 칸이 존재한다면, 왼쪽 방향으로 회전한 다음왼쪽으로 한 칸을 전진한다. 왼쪽 방향에 가보지 않은 칸이 없다면, 왼쪽 방향으로 회전만수행하고 1단계로 돌아간다.
    3. 만약 네 방향 모두 이미 가본 칸이거나 바다로 되어 있는 칸인 경우에는, 바라보는 방향을유지한 채로 한 칸 뒤로 가고 1단계로 돌아간다. 단, 이때 뒤쪽 방향이 바다인 칸이라 뒤로갈 수 없는 경우에는 움직임을 멈춘다.
* 현민이는 위 과정을 반복적으로 수행하면서 캐릭터의 움직임에 이상이 있는지 테스트하려고 한다.매뉴얼에 따라 캐릭터를 이동시킨 뒤에, 캐릭터가 방문한 칸의 수를 출력하는 프로그램을 만드시오.

### 입력 조건
* 첫째 줄에 맵의 세로 크기 N과 가로크기 M을 공백으로 구분하여 입력한다. (3 ≤ N, M≤ 50)
* 둘째 줄에 게임 캐릭터가 있는 칸의 좌표 (A, B)와 바라보는 방향 d가 각각 서로 공백으로 구분하여주어진다. 방향 의 값으로는 다음과 같이 4가지가 존재한다.
    - 0: 북쪽
    - 1: 동쪽
    - 2: 남쪽
    - 3: 서쪽
* 셋째 줄부터 맵이 육지인지 바다인지에 대한 정보가 주어진다. N개의 줄에 맵의 상태가 북쪽부터 남쪽순서대로 각 줄의 데이터는 서쪽부터 동쪽 순서대로 주어진다. 맵의 외곽은 항상 바다로 되어 있다.
    - 0 : 육지
    - 1 : 바다
* 처음에 게임 캐릭터가 위치한 칸의 상태는 항상 육지이다.

### 출력 조건
* 첫째 줄에 이동을 마친 후 캐릭터가 방문한 칸의 수를 출력한다.

### Test Case
```
[input]
4 4     # 4 by 4 맵 생성
1 1 0   # (1, 1)에 북쪽(0)을 바라보고 서 있는 캐릭터
1 1 1 1 # 첫 출은 모두 바다
1 0 0 1 # 둘째 줄은 바다/육지/육지/바다
1 1 0 1 # 셋째 줄은 바다/바다/육지/바다
1 1 1 1 # 넷째 줄은 모두 바다

[output]
3
```





## My Solution

```py
from sys import stdin

n, m = map(int, stdin.readline().split())
x, y, d = map(int, stdin.readline().split())

map_init = [list(map(int, stdin.readline().split())) for _ in range(n)] # 초기화 지도 생성
visited = [[False] * m for _ in range(n)] # 방문한 곳 인식
visited[x][y] = True # 초기 위치는 1로 처리

movable_check = [(-1, 0), (0, 1), (1, 0), (0, -1)] # 북, 동, 남, 서
move_count = 1 # 이동 횟수
turn_count = 0 # 회전 횟수

while True:
    # 왼쪽으로 회전
    d = 3 if d == 0 else d - 1

    mx = x + movable_check[d][0]
    my = y + movable_check[d][1]
    
    # 왼쪽회전 후 가보지도 않았고, 육지인 칸이 존재한다면
    if visited[mx][my] == False and map_init[mx][my] == 0:
        x, y = mx, my
        visited[mx][my] = True
        move_count += 1 # 이동 횟수
        turn_count = 0 # 회전 횟수
    else:
        turn_count += 1
    
    # 4개 방향으로 모두 갈 수 없다면 뒤로 이동(단, 바다면 안됨)
    if turn_count == 4:
        mx = x - movable_check[d][0]
        my = y - movable_check[d][1]
        
        if map_init[mx][my] == 0:
            x, y = mx, my
        else:
            break
        turn_count = 0
    
print(move_count)
```

<br>





## Solution

```py
# N, M을 공백을 기준으로 구분하여 입력받기
n, m = map(int, input().split())

# 방문한 위치를 저장하기 위한 맵을 생성하여 0으로 초기화
d = [[0] * m for _ in range(n)]
# 현재 캐릭터의 X 좌표, Y 좌표, 방향을 입력받기
x, y, direction = map(int, input().split())
d[x][y] = 1 # 현재 좌표 방문 처리

# 전체 맵 정보를 입력받기
array = []
for i in range(n):
    array.append(list(map(int, input().split())))

# 북, 동, 남, 서 방향 정의
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

# 왼쪽으로 회전
def turn_left():
    global direction
    direction -= 1
    if direction == -1:
        direction = 3

# 시뮬레이션 시작
count = 1
turn_time = 0
while True:
    # 왼쪽으로 회전
    turn_left()
    nx = x + dx[direction]
    ny = y + dy[direction]
    # 회전한 이후 정면에 가보지 않은 칸이 존재하는 경우 이동
    if d[nx][ny] == 0 and array[nx][ny] == 0:
        d[nx][ny] = 1
        x = nx
        y = ny
        count += 1
        turn_time = 0
        continue
    # 회전한 이후 정면에 가보지 않은 칸이 없거나 바다인 경우
    else:
        turn_time += 1
    # 네 방향 모두 갈 수 없는 경우
    if turn_time == 4:
        nx = x - dx[direction]
        ny = y - dy[direction]
        # 뒤로 갈 수 있다면 이동하기
        if array[nx][ny] == 0:
            x = nx
            y = ny
        # 뒤가 바다로 막혀있는 경우
        else:
            break
        turn_time = 0

# 정답 출력
print(count)
```




