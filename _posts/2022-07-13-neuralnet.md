---
1layout: post
title: 신경망(Neural Network)과 활성화 함수(Activation Function)의 등장
category: Deep Learning
tag: Deep-Learning
---




# From Perceptron to Neural Network

신경망에대해 살펴보기전에 퍼셉트론의 장단점을 살펴본다면, 퍼셉트론으로 복잡한 함수도 표현할 수 있다는 장점을 가진 반면 원하는 결과를 출력하도록 가중치 값을 적절히 정하는 작업을 여전히 인간이 수동으로 한다는 것이 단점이었습니다. AND, NAND, OR 게이트의 진리표를 보면서 우리 인간이 적절한 가중치 값을 정했습니다

**신경망(Neural Net)** 은 이 단점을 해결해 주는데, **가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력이 신경망의 중요한 성질입니다.**


<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178497156-1fe0f13f-8dba-48f4-8add-328bbbc88aea.png" alt="activation f" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.tibco.com/reference-center/what-is-a-neural-network">tibco.com</a></p>


신경망을 그림으로 나타내면 위 그림처럼 됩니다. 여기에서 가장 왼쪽 줄을 **입력층**, 맨 오른쪽 줄을 **출력층**, 중간 줄을 **은닉층**(입력층이나 출력층과 달리 사람 눈에는 보이지 않아서)이라고 합니다. 위 그림은 앞 장에서 본 퍼셉트론과 특별히 달라 보이지 않고, 실제로 뉴런이 연결되는 방식은 퍼셉트론에서 달라진 것이 없습니다.


## 퍼셉트론 돌아보기

신경망에서의 신호 전달 방법을 살펴보기 전에 다음과 같은 구조의 네트워크를 생각해봅시다.

<p align="center"><img width="232" alt="fig 3-2" src="https://user-images.githubusercontent.com/77891754/178498354-f80dcefd-840e-4020-9aa5-fb4113ec5161.png"></p>

<p align="center" style="font-size:80%">이미지 출처 : 밑바닥부터 시작하는 딥러닝</a></p>

위 그림은 $x_1$과 $x_2$라는 두 신호를 입력받아 $y$를 출력하는 퍼셉트론입니다. 이 퍼셉트론을 수식으로 나타내면 다음과 같이 됩니다.

$$
y = \begin{cases} 0 \qquad (b + w_1x_1 + w_2x_2 \leq 0) \\
1 \qquad (b + w_1x_1 + w_2x_2 > 0) \end{cases}
$$

여기서 $b$는 **편향**을 나타내는 매개변수로, **뉴런이 얼마나 쉽게 활성화되느냐를 제어**합니다. 한편, $w_1$과 $w_2$는 각 신호의 **가중치**를 나타내는 매개변수로, **각 신호의 영향력을 제어합니다.** 그런데 위 그림의 네트워크에는 편향 $b$가 보이지 않는데, 여기에 편향을 명시한다면 다음과 같이 나타낼 수 있습니다.


<p align="center"><img width="235" alt="fig 3-3" src="https://user-images.githubusercontent.com/77891754/178618368-94ab728e-dbbd-455f-b759-9bea5c060313.png"></p>

<p align="center" style="font-size:80%">이미지 출처 : 밑바닥부터 시작하는 딥러닝</a></p>


위 그림에서는 가중치가 $b$이고 입력이 1인 뉴런이 추가되었습니다. 이 퍼셉트론의 동작은 $x_1, x_2, 1$이라는 3개의 신호가 뉴런에 입력되어 각 신호에 가중치를 곱한 후 다음 뉴런에 전달되며, 다음 뉴런에서는 이 신호들의 값을 더하여 그 합이 0을 넘으면 1을 출력하고 그렇지 않으면 0을 출력합니다.

위 식을 더 간결한 형태로 다시 작성해보면, 조건 분기의 동작(0을 넘으면 1을 출력하고 그렇지 않으면 0을 출력)을 하나의 함수로 나타낼 수 있으며, 이 함수를 $h(x)$라 하면 다음과 같이 표현할 수 있습니다.


$$
y = h(b + w_1x_1 + w_2x_2) \\
h(x) = \begin{cases} 0 \qquad (x \leq 0) \\
1 \qquad (x > 0) \end{cases}
$$


위 식은 입력 신호의 총합이 $h(x)$라는 함수를 거쳐 변환되어, 그 변환된 값이 $y$의 출력이 됨을 보여주는데, $h(x)$함수는 입력이 0을 넘으면 1을 돌려주고 그렇지 않으면 0을 돌려줍니다.



## The emergence of Activation functions

조금 전 $h(x)$라는 함수가 등장했는데, 이처럼 **입력 신호의 총합을 출력 신호로 변환하는 함수**를 일반적으로 **활성화 함수(activation function)**라 합니다. 활성화라는 이름이 말해주듯 **활성화 함수는 신호의 총합이 활성화를 일으키는지를 정하는 역할**을 합니다.

위에서 본 수식에서는 가중치가 곱해진 입력 신호의 총합을 계산하고, 그 합을 활성화 함수에 입력해 결과를 내는 2단계로 처리됩니다. 그래서 위 식은 다음과 같은 2개의 식으로 나눌 수 있습니다.

$$
a = b + w_1x_1 + w_2x_2 \\
y = h(a) \\
$$

위 식은 가중치가 달린 입력 신호와 편향의 총합을 계산하고, 이를 $a$라 하며, 그리고 $a$를 함수 $h()$에 넣어 $y$를 출력하는 흐름입니다. 지금까지와 같이 뉴런을 큰 원 다음 그림처럼 나타낼 수 있습니다.


<p align="center"><img width="289" alt="fig 3-4" src="https://user-images.githubusercontent.com/77891754/178621732-e67d5dbd-37c5-4fac-92b7-69e46b60707c.png"></p>

<p align="center" style="font-size:80%">이미지 출처 : 밑바닥부터 시작하는 딥러닝</a></p>


위 그림에서는 기존 뉴런의 원을 키우고, 그 안에 활성화 함수의 처리 과정을 명시적으로 그려 넣었습니다. 즉, 가중치 신호를 조합한 결과가 $a$라는 노드가 되고, 활성화 함수 $h()$를 통과하여 $y$라는 노드로 변환되는 과정이 분명하게 나타나 있습니다. **즉 활성화 함수가 퍼셉트론에서 신경망으로 가기 위한 길잡이** 입니다.

일반적으로 **단순 퍼셉트론**은 단층 네트워크에서 계단 함수(임계값을 경계로 출력이 바뀌는 함수)를 활성화 함수로 사용한 모델을 가리키고, **다층 퍼셉트론**은 신경망(여러 층으로 구성되고 시그모이드 함수 등의 매끈한 활성화 함수를 사용하는 네트워크)을 가리킵니다.




<br>

# Activation function

## Step function

$$
h(x) = \begin{cases} 0 \qquad (x \leq 0) \\
1 \qquad (x > 0) \end{cases}
$$

위 식과 같은 **활성화 함수**는 **임계값을 경계로 출력이 바뀌는데**, 이런 함수를 **계단 함수(step function)** 라 합니다. 그래서 “퍼셉트론에서는 활성화 함수로 계단 함수를 이용한다”라 할 수 있습니다. **즉, 활성화 함수로 쓸 수 있는 여러 후보 중에서 퍼셉트론은 계단 함수를 채용하고 있습니다.** 그렇다면 계단 함수 이외의 함수를 사용하면 어떻게 될까요? 우선 계단 함수를 구현해보겠습니다.

계단 함수는 입력이 0을 넘으면 1을 출력하고, 그 외에는 0을 출력하는 함수입니다. 다음은 이러한 계단 함수를 단순하게 구현한 것 입니다.

```python
# 첫번째 구현
# 이 구현은 단순하고 쉽지만, 인수 x는 실수(부동소수점)만 받아들입니다.
# 즉, 넘파이 배열을 인수로 넣을 수 없습니다.
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


# 두번째 구현
# 넘파이 배열도 지원하도록 다음과 같이 구현
def step_function(x):
    y = x > 0
    return y.astype(np.int)
```

앞에서 정의한 계단 함수를 그래프로 출력해보겠습니다.

```python
%matplotlib inline 

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()
```

<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178623076-612dd346-edf6-4058-ac22-537ef7578bec.png"></p>



위 그림에서 보듯 계단 함수는 0을 경계로 출력이 0에서 1(또는 1에서 0)로 바뀝니다. 바로 이 그림처럼 값이 바뀌는 형태가 계단처럼 생겼기 때문입니다.





## Sigmoid function

다음은 신경망에서 자주 이용하는 활성화 함수인 **시그모이드 함수(sigmoid function)** 를 나타낸 식입니다.

$$
h(x) = \frac{1}{1+e^{-x}}
$$

위 식에서 $exp(-x)$는 $e^{-x}$를 뜻하며, $e$는 자연상수로 2.7182...의 값을 갖는 실수입니다. 위 식으로 나타나는 시그모이드 함수 역시 단순한 함수일 뿐이며, 함수는 입력을 주면 출력을 돌려주는 변환기입니다. 예를 들어 시그모이드 함수에 1.0과 2.0을 입력하면 h(1.0) = 0.731.... h(2.0) = 0.880...처럼 특정 값을 출력합니다.

신경망에서는 활성화 함수로 시그모이드 함수를 이용하여 신호를 변환하고, 그 변환된 신호를 다음 뉴런에 전달합니다. **사실 퍼셉트론과 신경망의 주된 차이는 이 활성화 함수 뿐입니다.** 그 외에 뉴런이 여러 층으로 이어지는 구조와 신호를 전달하는 방법은 기본적으로 앞에서 살펴본 퍼셉트론과 같습니다. 그렇다면 시그모이드 함수를 구현해보겠습니다.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

시그모이드 함수를 그래프로 그리면, 그래프를 그리는 코드는 앞 절의 계단 함수 그리기 코드와 거의 같습니다. 유일하게 다른 부분은 y를 출력하는 함수를 sigmoid 함수로 변경한 곳 입니다.**

```python
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()
```

<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178623469-0438d7de-810e-4da9-87db-d831b3496dd5.png"></p>





## ReLU function

활성화 함수로서 계단 함수와 시그모이드 함수를 소개했는데, 시그모이드 함수는 신경망 분야에서 오래전부터 이용해왔으나, 최근에는 ReLU(Rectifted Linear Unit)함수를 주로 이용합니다.

ReLU는 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수로 수식으로는 다음과 같이 쓸 수 있습니다.

$$
h(x) = \begin{cases} x \qquad (x > 0) \\
0 \qquad (x \leq 0) \end{cases}
$$

ReLU 함수에 대하여 간단하게 구현해보고 시각화를 해보겠습니다.

```python
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.1, 5.1) # y축의 범위 지정
plt.show()
```

<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178624839-884749a2-24aa-4d1e-af39-8cf67ce0ccac.png"></p>





## Step function VS Sigmoid function

시그모이드 함수와 계단 함수를 비교해보겠습니다.

```python
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1, label="sigmoid")
plt.plot(x, y2, linestyle="--", label="step_function")
plt.xlabel("X") # x축 이름
plt.ylabel("y") # y축 이름
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()
```

<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178623890-73c09993-23b9-4ce7-8103-0283a0447f60.png"></p>


### 계단 함수

- 계단 함수는 **0을 경계로 출력이 갑자기 바뀝니다.**
- 계단 함수가 0과 1 중 하나의 값만 돌려줍니다.



### 시그모이드 함수

- 시그모이드 함수는 부드러운 곡선이며 **입력에 따라 출력이 연속적으로 변화**합니다. 시그모이드 함수의 이 매끈함이 신경망 학습에서 아주 중요한 역할을 하게 됩니다.
- 시그모이드 함수는 실수(0.731..., 0.880... 등)를 돌려준다는 점도 다릅니다. 다시 말해 퍼셉트론에서는 뉴런 사이에 0 혹은 1이 흘렀다면, **신경망에서는 연속적인 실수가 흐릅니다.**



### 공통점

- 큰 관점에서 보면 둘은 같은 모양을 하고 있습니다. 둘 다 입력이 작을 때의 출력은 0에 가깝고 (혹은 0이고), 입력이 커지면 출력이 1에 가까워지는(혹은 1이 되는) 구조입니다.
- 즉, 계단 함수와 시그모이드 함수는 입력이 중요하면 큰 값을 출력하고 입력이 중요하지 않으면 작은 값을 출력합니다.
- 입력이 아무리 작거나 커도 출력은 0에서 1 사이라는 것도 둘의 공통점입니다.
- 둘 다 비선형 함수입니다.




<br>

# Non-linear function

계단 함수와 시그모이드 함수의 중요한 공통점으로, 둘 모두는 **비선형 함수**입니다. 시그모이드 함수는 곡선, 계단 함수는 계단처럼 구부러진 직선으로 나타나며, 동시에 비선형 함수로 분류됩니다.

## 신경망에서는 활성화 함수로 비선형 함수를 사용해야 합니다.

달리 말하면 선형 함수를 사용해서는 안 됩니다. 왜 선형 함수는 안 되는 걸까? 그 이유는 바로 선형 함수를 이용하면 신경망의 층을 깊게 하는 의미가 없어지기 때문입니다.

## 선형 함수의 문제는 층을 아무리 깊게 해도 '은닉층이 없는 네트워크'로도 똑같은 기능을 할 수 있다는 데 있습니다.

구체적으로 설명해주는 간단한 예를 생각해보면, 선형 함수인 $h(x) = cx$를 활성화 함수로 사용한 3층 네트워크를 떠올려보자. 이를 식으로 나타내면 $y(x) = h(h(h(x)))$가 됩니다. 이 계산은 $y(x) = c * c * c * x$처럼 곱셈을 세 번 수행하지만, 실은 $y(x) = ax$와 똑같은 식입니다. $a = c^3$이라고만 하면 끝이다. 즉, 은닉층이 없는 네트워크로 표현할 수 있습니다.

이 예처럼 선형 함수를 이용해서는 여러 층으로 구성하는 이점을 살릴 수 없습니다. 그래서 층을 쌓는 혜택을 얻고 싶다면 활성화 함수로는 반드시 비선형 함수를 사용해야 합니다.





