---
title: 활성화 함수의 역할과 종류
category: Deep Learning
tag: Deep-Learning
date: 2022-07-14
---




# 활성화 함수의 필요성

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/178894876-39ae23f3-dab5-41c1-8f71-feb55b652e00.gif" alt="activation f" style="zoom:100%;" /></p>

동물의 대뇌피질에 있는 신경세포들은 서로 화학물질을 전달함으로써 신경세포 또는 뉴런(neuron)간 정보를 교환합니다. 각 신경세포들은 이렇게 전달받은 정보들을 취합한 후 신경세포의 세포체(cell body, soma)에서는 화학적으로 전달받은 정보를 활성화(activation)라는 과정을 통해 전기신호로 전환한 후 신경세포 말단으로 전달하고, 전기신호로 신경세포 말단으로 전달된 정보는 다시 화학물질로 전환되어 다음 신경세포로 전달됩니다. 이때 신경세포의 세포체는 여러가지 복잡한 생물학적 규칙에 따라 신호를 보내거나 또는 보내지 않거나 하며 신호의 강도를 조절하고 신경망 연결을 강화하거나 악화시킵니다. 이처럼 생물학적 신경망에서 **활성화는 학습과정에서 매우 중요한 역할**을 합니다.


<p align="center"><img src="https://user-images.githubusercontent.com/77891754/178860852-3de9c4c7-b0df-4d29-9b67-66b9e6b112da.png" alt="activation f" style="zoom:40%;" /></p>

<p align="center" style="font-size:80%">Heaviside step function</a></p>


생물학적 신경망을 모방한 인공신경망은 생물학적 신경망처럼 신호가 전달되는 어떤 규칙을 설명하는 수학적인 모델이 필요합니다. 이러한 **신호전달 역할을 하는 수학적인 모델**이 바로 **활성화 함수(activation function)** 입니다. 인공신경망의 원조인 TLU(Threshold Logic Unit)나 퍼셉트론(Perceptron) 에서는 그림과 같이 0과 1 두가지 강도를 가지는  Heaviside step function을 사용하였는데 신호전달이 같은 강도로 **두가지(binary) 형태로 발생되는 것 보다는 여러가지 강도로 연속적으로 보내는 것이 타당하다고 판단**되어 시그모이드(sigmoid)나 tanh 같은 **연속적이고 미분 가능한 함수가 활성화 함수로 사용되기 시작**되었습니다.

인공신경망은 최초 SLP(Single Layered Perceptron) 형태에서 보다 복잡한 문제를 풀 수 있는 MLP(Multi Layered Perceptron) 구조로 진화하면서 MLP 모델을 학습시킬 수 있는 에러의 역전파(Propagation of Error) 알고리즘이 개발되었고 역전파에 적합한 활성화 함수가 도입되었습니다. 역전파 과정은 사실 생물학적 신경망에서는 발생되는 않는 인공신경망에서 필요한 수학적인 학습 방식이라고 할 수 있습니다.

경사법(Gradient Method)을 이용하여 역전파를 통해 인공신경망 모델을 학습시키는 과정을 한번 살펴보겠습니다. 데이터가 입력되면 학습변수 텐서에 곱해져서 활성화 함수에 적용됩니다. 이런 과정이 여러 단계의 신경층을 통해 순전파하면서 예측값을 만들고, 예측값을 기반으로 목적함수를 계산하고, 이 목적함수를 각 신경층 단계별로 연쇄법칙(Chain rule) 적용하여 역전파 하면서 원하는 학습변수에 대한 기울기(gradient)를 구합니다. 그리고 이 기울기에 학습률을 곱해 각 학습변수를 업데이트 합니다. **이때 활성화 함수가 포함된 목적함수의 기울기가 0이 되면 학습이 되지 않습니다.** 따라서 순전파를 통해 목적함수에 포함될 활성화 함수는 학습 변수가 존재할 대부분의 범위에서 그 미분값이 0이 아닌 것이 좋습니다. 이러한 사항들을 정리하면 인공신경망에서의 활성화 함수는 다음과 같은 기능을 수행해야 합니다.


## 정보의 희소성(sparsity) 강화

인공신경망에서 순전파 할 때 활성화 함수는 의미 있는 정보는 증폭하고 의미 없는 정보는 소멸시킵니다. 즉 이러한 **정보의 양극화**를 **희소성(sparsity)** 라고 하는데 수학모델을 기반으로 하는 인공신경망은 이러한 **희소한 데이터로 분석하는 것이 정확도나 수렴속도 향상에 도움이 됩니다.** 예를 들면 다음 그림과 같이 각 신경망에서 정보가 전달될 때 활성화 함수를 통해 정보가 희소해지면 어떤 노드는 1을 가지고 어떤 노드는 0을 갖는 좀 더 변별력 있는 결과값을 찾을 수 있고 수렴속도도 빠릅니다.


<p align="center"><img width="592" alt="image" src="https://user-images.githubusercontent.com/77891754/178884336-413c769e-9db2-4159-843b-dc9e85d32d1e.png" /></p>

<p align="center" style="font-size:80%">Deep Sparse Rectifier Neural Networks, 2011</a></p>


## 경사소멸(gradient vanishing)의 최소화

인공신경망에서 학습과정은 목적함수의 역전파를 통해 각 학습변수를 갱신하는 것입니다. 학습변수를 갱신하는 방법 중 가장 보편적인 경사법을 사용한다면 각 학습변수는 경사도 또는 기울기를 통해 구해집니다. **따라서 목적함수에 포함될 활성화 함수의 미분인 도함수 또는 미분함수가 각 학습변수의 모든 영역에서 0이 아닌 것이 좋습니다.** 활성화 함수가 포함된 목적함수를 학습변수로 미분한 값이 0일 때 이런 상황을 경사소멸이라고 하며 갱신할 값이 0이기 때문에 학습이 이루어지지 않습니다.






# 활성화 함수의 종류


## Heaviside Step Function


Heaviside step function 라는 이름이 붙은 이유는 간단합니다. 이 함수로 들어온 입력이 특정 임계점을 넘으면 $1$(혹은 True)를 출력하고 그렇지 않을 때는 $0$을 출력하기 때문입니다.


### 함수 그래프

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/178910428-ef4b35df-17c6-4bac-b5ea-f1aeb1e2ca2e.png" alt="activation f" style="zoom:30%;" /></p>


### 함수 수식

$$
f(x)=
\begin{cases}
0 \; for\ x < 0 \\ 1\; for\ x\ ≥ 0
\end{cases}
$$


### 특징
- 단순한 구조
- 계단 함수는 0을 경계로 출력이 갑자기 바뀐다. 즉 0과 1 중 하나의 값만 돌려준다.
- 입력이 아무리 작거나 커도 출력은 0 또는 1
- 큰 관점에서 입력이 작을 때의 출력은 0이고, 입력이 커지면 출력이 1이되는 구조로, 입력이 중요하면 큰 값을 출력하고 입력이 중요하지 않으면 작은 값을 출력한다.
- 비선형 함수


### 단점
- 계단 함수는 그래프에서 보이는 것 처럼, 굉장히 극적으로 모양이 변하기 때문에 데이터의 손실이 발생할 가능성이 굉장히 높아진다.
- 불연속 함수이기 때문에 미분이 불가능하다.
- 다중 출력이 불가능하다.
- 합산된 값이 0.1이든 1.0이든 모두 무시하고 1로 전달하므로 출력되는 결과값이 너무 희석된다.





## Linear Activation Function

선형 활성화 함수(linear activation function)은 말 그대로 '선형'인 활성화 함수입니다.


### 그래프

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/178911611-76a71980-949a-4431-a00c-2cffdde3b59b.png" alt="activation f" style="zoom:30%;" /></p>


### 함수 수식
$$
f(x) = x
$$


### 특징
- 이 함수는 입력의 가중치 합에 대해 아무 것도 하지 않고 단순히 주어진 값을 내보낸다.
- 선형 활성화 함수를 사용한 모델은 이진 계단 함수를 사용한 모델과 다르게 다중 출력이 가능하다.
- 때문에 이진 분류는 물론이고 간단한 다중 분류 문제까지도 해결할 수 있습니다.
- 또한 미분이 가능해서 역전파 알고리즘 또한 사용할 수 있다.


### 단점
- 함수의 도함수가 상수이고 입력 x와 관련이 없기 때문에 역전파를 사용은 가능할뿐 실제로 사용할 수 없다.
- 모델에 선형 활성화 함수를 사용한다면 비선형적 특성을 지닌 데이터를 예측하지 못한다.
- 선형 활성화 함수를 사용하면 신경망의 모든 레이어가 하나로 축소된다. 신경망의 레이어 수에 관계없이 마지막 레이어는 여전히 첫 번째 레이어의 선형 함수다. 따라서 본질적으로 선형 활성화 함수는 신경망을 단 하나의 계층으로 바꾼다.





## Sigmoid / Logistic Activation Function

시그모이드 함수는 계단함수를 부드럽게 만든 함수라고 볼 수 있다. 여기서 부드럽다는 의미는 모든 영역에서 미분 가능하다는 뜻입니다. 시그모이드 함수는 다음 그래프와 같이 (-∞, +∞)의 값을 입력하면 [0, 1] 의 실수값으로 압축해줍니다. 시그모이드 함수는 계단함수 이후 가장 많이 사용되었던 활성화 함수였으나 경사소멸 문제나 수렴속도의 저하로 최근에는 출력값을 [0, 1]로 맞추는 것이 필요한 경우가 아니라면 잘 사용되지 않습니다.


### 그래프

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/178914026-082d5f1b-83f8-4683-a449-fb7ea6d563d0.png" alt="activation f" style="zoom:30%;"/>
</p>


### 함수 수식

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$


### 특징
- 비선형 함수이다.
- 결과값을 [0, 1] 범위로 압축하므로 이 범위 밖의 값으로 확대되는 경우는 없다.
- 입력이 중요하면 큰 값을 출력하고 입력이 중요하지 않으면 작은 값을 출력한다.
- 부드러운 활성화 함수로 1차 미분 가능 함수이다.
- 시그모이드 함수를 쓰는 가장 주된 이유가 바로 치역이 $0$과 $1$사이이므로, 특히 확률을 예측해야 하는 모델에서 자주 사용된다.
- 시그모이드 함수는 부드러운 곡선이며 입력에 따라 출력이 연속적으로 변화한다. 시그모이드 함수의 이 매끈함이 신경망 학습에서 아주 중요한 역할을 하게 됩니다.


### 단점
- 0을 중심으로 x값이 시그모이드 양 끝으로 멀어지게 되면 0과 1로 수렴하게 되어 변별력이 없어짐.
- x값이 0에서 멀어지면서 경사소멸 현상이 발생한다.
- 결과값이 [0, 1]이므로 중간값이 0이 아니기 때문에 수렴속도가 느리다.




## Tanh Function (Hyperbolic Tangent)

### 그래프
<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/178922610-9f74bd86-89c4-4250-ab72-0367a318dd5a.png" alt="activation f" style="zoom:30%;"/>
</p>


### 함수 수식

$$
\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}
$$


### 특징


### 단점














## ReLU Function


### 그래프

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/178923169-dac1021f-1e7d-4750-9035-30a21b988524.png" alt="activation f" style="zoom:30%;"/>
</p>


### 함수 수식


### 특징


### 단점





## Leaky ReLU Function


### 그래프

<p align="center">
<img src="https://user-images.githubusercontent.com/77891754/178926003-ca4a186d-489c-4740-8f12-6f6462995f71.png" alt="activation f" style="zoom:30%;"/>
</p>


### 함수 수식


### 특징


### 단점





## Parametric ReLU Function


### 그래프

<p align="center">
<img src="" alt="activation f" style="zoom:30%;"/>
</p>


### 함수 수식


### 특징


### 단점













## Softmax Function

### 그래프

<p align="center">
<img src="" alt="activation f" style="zoom:30%;"/>
</p>


### 함수 수식


### 특징


### 단점





## Swish

### 그래프

<p align="center">
<img src="" alt="activation f" style="zoom:30%;"/>
</p>


### 함수 수식


### 특징


### 단점





## Scaled Exponential Linear Unit (SELU)

### 그래프

<p align="center">
<img src="" alt="activation f" style="zoom:30%;"/>
</p>


### 함수 수식


### 특징


### 단점











