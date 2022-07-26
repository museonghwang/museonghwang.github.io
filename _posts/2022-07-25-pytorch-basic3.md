---
title: Linear Regression 학습을 위한 pytorch 기본
category: Pytorch
tag: pytorch
date: 2022-07-25
---     





# 1.선형 회귀(Linear Regression)

## 1.1 가설(Hypothesis) 수립

선형 회귀란 학습 데이터와 가장 잘 맞는 하나의 직선을 찾는 방법으로, 이때 선형 회귀의 가설(직선의 방정식)은 아래와 같은 형식을 가집니다.

$$y=Wx+b$$

머신 러닝에서 식을 세울때 이 식을 가설(Hypothesis)라고 합니다. 보통 머신 러닝에서 가설은 임의로 추측해서 세워보는 식일수도 있고, 경험적으로 알고 있는 식일 수도 있고, 또는 맞는 가설이 아니라고 판단되면 계속 수정해나가게 되는 식이기도 합니다. 가설의 H를 따서 y 대신 다음과 같이 식을 표현하기도 합니다.

$$H(x)=Wx+b$$

이때 $x$와 곱해지는 $W$를 가중치(Weight)라고 하며, $b$를 편향(bias)이라고 합니다.



## 1.2 비용 함수(Cost function)에 대한 이해

**비용 함수(cost function) = 손실 함수(loss function) = 오차 함수(error function) = 목적 함수(objective function)**

손실 함수에 대해 이해하기 위해서 다음과 같은 예가 있다고 가정하겠습니다. 어떤 4개의 훈련 데이터가 있고, 이를 2차원 그래프에 4개의 점으로 표현한 상태라고 하겠습니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/180999219-a55e2523-af0e-441c-8cf1-bacea51035b9.png">
</p>

지금 목표는 4개의 점을 가장 잘 표현하는 직선을 그리는 일입니다. 임의로 3개의 직선을 그립니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/181124551-1fb51f8d-a6e1-48ff-ad52-439ba9b1ad73.png">
</p>

위의 그림은 서로 다른 $W$와 $b$의 값에 따라서 천차만별로 그려진 3개의 직선의 모습을 보여줍니다. 이 3개의 직선 중에서 4개의 점을 가장 잘 반영한 직선은 4개의 점에 가깝게 지나가는 느낌의 검은색 직선 같습니다. 하지만 수학에서 어떤 직선이 가장 적절한 직선인지를 수학적인 근거를 대서 표현할 수 있어야 합니다. 그래서 **오차(error)**라는 개념을 도입합니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/181124560-98e9a256-0ccb-4538-a6d0-4338702b095f.png">
</p>

위 그림은 임의로 그려진 주황색 선에 대해서 각 실제값(4개의 점)과 직선의 예측값(동일한 $x$값에서의 직선의 $y$값)에 대한 값의 차이를 빨간색 화살표 ↕로 표현한 것입니다. 각 실제값과 각 예측값과의 차이고, 이를 각 실제값에서의 오차라고 말할 수 있습니다. 이 직선의 예측값들과 실제값들과의 총 오차(total error)는 직관적으로 생각하기에 모든 오차를 다 더하면 될 것 같습니다. 위 주황색 직선의 식은 $y=13x+1$이며, 각 오차는 다음과 같습니다.

|$hours(x)$|2|3|4|5|
|:-|:-|:-|:-|:-|
|실제값|25|50|42|61|
|예측값|27|40|53|66|
|오차|-2|10|-11|-5|

각 오차를 계산해봤습니다. 그런데 수식적으로 단순히 '오차 = 실제값 - 예측값'으로 정의하면 오차값이 음수가 나오는 경우가 생깁니다. 예를 들어 위의 표에서만 봐도 오차가 음수인 경우가 3번이나 됩니다. 이 경우, 오차를 모두 더하면 덧셈 과정에서 오차값이 +가 되었다가 -되었다가 하므로 제대로 된 오차의 크기를 측정할 수 없습니다. 그래서 오차를 그냥 전부 더하는 것이 아니라, 각 오차들을 제곱해준 뒤에 전부 더하겠습니다.


이를 수식으로 표현하면 아래와 같습니다. 단, 여기서 n은 갖고 있는 데이터의 개수를 의미합니다.

$$\sum_{i=1}^{n}[y^{(i)} - H(x^{(i)})]^{2} = (-2)^{2}+(10)^{2}+(-11)^{2}+(-5)^{2} = 250$$

이때 데이터의 개수인 n으로 나누면, 오차의 제곱합에 대한 평균을 구할 수 있는데 이를 평균 제곱 오차(Mean Squered Error, MSE)라고 합니다. 수식은 아래와 같습니다.

$$\frac{1}{n}\sum_{i=1}^{n}[y^{(i)} - H(x^{(i)})]^{2} = 250/4 = 62.5$$

이를 실제로 계산하면 44.5가 됩니다. 이는 $y=13x+1$의 예측값과 실제값의 평균 제곱 오차의 값이 62.5임을 의미합니다. 평균 제곱 오차는 이번 회귀 문제에서 적절한 $W$와 $b$를 찾기위해서 최적화된 식입니다. 그 이유는 평균 제곱 오차의 값을 최소값으로 만드는 $W$와 $b$를 찾아내는 것이 가장 훈련 데이터를 잘 반영한 직선을 찾아내는 일이기 때문입니다.

평균 제곱 오차를 $W$와 $b$에 의한 비용 함수(Cost function)로 재정의해보면 다음과 같습니다.

$cost(W,b) = \frac{1}{n}\sum_{i=1}^{n}[y^{(i)} - H(x^{(i)})]^{2}$

**즉 $Cost(W,b)$를 최소가 되게 만드는 $W$와 $b$를 구하면 훈련 데이터를 가장 잘 나타내는 직선을 구할 수 있습니다.**



## 1.3 옵티마이저 - 경사 하강법(Gradient Descent)

앞서 정의한 비용 함수(Cost Function)의 값을 최소로 하는 W와 b를 찾는 방법으로 사용되는 것이 *옵티마이저(Optimizer) 알고리즘* 또는 *최적화 알고리즘*이라고도 부릅니다. 그리고 이 옵티마이저 알고리즘을 통해 적절한 $W$와 $b$를 찾아내는 과정을 머신 러닝에서 *학습(training)*이라고 부릅니다. 여기서는 가장 기본적인 옵티마이저 알고리즘인 *경사 하강법(Gradient Descent)*에 대해서 살펴보겠습니다.

설명에서 편향 $b$는 고려하지 않겠습니다. 즉, $b$가 0이라고 가정한 $y=Wx$라고 가정합니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/181125444-5128f4a3-d15d-4aa7-98c3-5e7203c6d99b.png">
</p>

위의 그림에서 주황색선은 기울기 $W$가 20일 때, 초록색선은 기울기 $W$가 1일 때를 보여줍니다. 다시 말하면 각각 $y=20x, y=x$에 해당되는 직선입니다. ↕는 각 점에서의 실제값과 두 직선의 예측값과의 오차를 보여줍니다. 이는 앞서 예측에 사용했던 $y=13x+1$ 직선보다 확연히 큰 오차값들입니다. 즉, 기울기가 지나치게 크면 실제값과 예측값의 오차가 커지고, 기울기가 지나치게 작아도 실제값과 예측값의 오차가 커집니다. 사실 $b$ 또한 마찬가지인데 $b$가 지나치게 크거나 작으면 오차가 커집니다.

$W$와 cost의 관계를 그래프로 표현하면 다음과 같습니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/181126068-54640ecd-0d33-40f3-9d52-a6c34ad5d97a.png">
</p>

기울기 $W$가 무한대로 커지면 커질 수록 cost의 값 또한 무한대로 커지고, 반대로 기울기 $W$가 무한대로 작아져도 cost의 값은 무한대로 커집니다. 위의 그래프에서 cost가 가장 작을 때는 맨 아래의 볼록한 부분입니다. 기계가 해야할 일은 cost가 가장 최소값을 가지게 하는 $W$를 찾는 일이므로, 맨 아래의 볼록한 부분의 $W$의 값을 찾아야 합니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/181126068-54640ecd-0d33-40f3-9d52-a6c34ad5d97a.png">
</p>

기계는 임의의 초기값 $W$값을 정한 뒤에, 맨 아래의 볼록한 부분을 향해 점차 $W$의 값을 수정해나갑니다. 위의 그림은 $W$값이 점차 수정되는 과정을 보여줍니다. 그리고 이를 가능하게 하는 것이 경사 하강법(Gradient Descent)입니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/181126068-54640ecd-0d33-40f3-9d52-a6c34ad5d97a.png">
</p>

위의 그림에서 초록색 선은 $W$가 임의의 값을 가지게 되는 네 가지의 경우에 대해서 그래프 상으로 접선의 기울기를 보여주며, 맨 아래의 볼록한 부분으로 갈수록 접선의 기울기가 점차 작아진다는 점입니다. 그리고 맨 아래의 볼록한 부분에서는 결국 접선의 기울기가 0이 됩니다. 그래프 상으로는 초록색 화살표가 수평이 되는 지점입니다.

즉, cost가 최소화가 되는 지점은 접선의 기울기가 0이 되는 지점이며, 또한 미분값이 0이 되는 지점입니다. 경사 하강법의 아이디어는 비용 함수(Cost function)를 미분하여 현재 $W$에서의 접선의 기울기를 구하고, 접선의 기울기가 낮은 방향으로 $W$의 값을 변경하는 작업을 반복하는 것에 있습니다. 이 반복 작업에는 현재 $W$에 접선의 기울기를 구해 특정 숫자 $\alpha$를 곱한 값을 빼서 새로운 $W$로 사용하는 식이 사용됩니다.

여기서의 $\alpha$는 학습률(learning rate)입니다. 학습률 $\alpha$는 $W$의 값을 변경할 때, 얼마나 크게 변경할지를 결정합니다. 또는 $W$를 그래프의 한 점으로보고 접선의 기울기가 0일 때까지 경사를 따라 내려간다는 관점에서는 얼마나 큰 폭으로 이동할지를 결정합니다.

$$기울기 = \frac{\partial cost(W)}{\partial W}$$

기울기가 음수일 때와 양수일 때 어떻게 $W$값이 조정되는지 보겠습니다.

* **기울기가 음수일 때 : $W$의 값이 증가**

$$W:=W−α×(음수기울기)=W+α×(양수기울기)$$

기울기가 음수면 $W$의 값이 증가하는데 이는 결과적으로 접선의 기울기가 0인 방향으로 $W$의 값이 조정됩니다.

* **기울기가 양수일 때 : W의 값이 감소**
$$W:=W−α×(양수기울기)$$

기울기가 양수면 $W$의 값이 감소하게 되는데 이는 결과적으로 기울기가 0인 방향으로 $W$의 값이 조정됩니다. 즉, 아래의 수식은 접선의 기울기가 음수거나, 양수일 때 모두 접선의 기울기가 0인 방향으로 W의 값을 조정합니다.

$$W:=W−α\frac{\partial }{\partial W}cost(W)$$

지금까지는 $b$는 배제시키고 최적의 $W$를 찾아내는 것에만 초점을 맞추어 경사 하강법의 원리에 대해서 배웠는데, 실제 경사 하강법은 $W$와 $b$에 대해서 동시에 경사 하강법을 수행하면서 최적의 $W$와 $b$의 값을 찾아갑니다.





# 2. 파이토치로 단순 선형 회귀 구현하기

## 2.2 기본 셋팅

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

```py
# 랜덤 시드(random seed)
torch.manual_seed(1)
```



## 2.2 변수 선언

훈련 데이터인 x_train과 y_train을 선언합니다.

```py
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
```

x_train과 y_train의 값과 크기(shape)를 출력해보겠습니다.
```py
print(x_train)
print(x_train.shape)
```
```
tensor([[1.],
        [2.],
        [3.]])
torch.Size([3, 1])
```

```py
print(y_train)
print(y_train.shape)
```
```
tensor([[2.],
        [4.],
        [6.]])
torch.Size([3, 1])
```



## 2.3 가중치와 편향의 초기화

선형 회귀의 목표는 가장 잘 맞는 직선을 정의하는 $W$와 $b$의 값을 찾는 것입니다. 우선 가중치 $W$를 0으로 초기화하고, 이 값을 출력해보겠습니다.

```py
# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
W = torch.zeros(1, requires_grad=True) 
# 가중치 W를 출력
print(W) 
```
```
tensor([0.], requires_grad=True)
```

가중치 $W$가 0으로 초기화되어있으므로 0이 출력된 것을 확인할 수 있습니다. 위에서 `requires_grad=True`가 인자로 주어진 것을 확인할 수 있습니다. 이는 이 변수는 학습을 통해 계속 값이 변경되는 변수임을 의미합니다. 즉 이것을 True로 설정하면 자동 미분 기능이 적용됩니다. 선형 회귀부터 신경망과 같은 복잡한 구조에서 파라미터들이 모두 이 기능이 적용됩니다. `requires_grad = True`가 적용된 텐서에 연산을 하면, 계산 그래프가 생성되며 backward 함수를 호출하면 그래프로부터 자동으로 미분이 계산됩니다.

마찬가지로 편향 $b$도 0으로 초기화하고, 학습을 통해 값이 변경되는 변수임을 명시합니다.
```
b = torch.zeros(1, requires_grad=True)
print(b)
```
```
tensor([0.], requires_grad=True)
```

현재 가중치 $W$와 $b$ 둘 다 0이므로 현 직선의 방정식은 다음과 같습니다.

$$y=0*x+0$$

지금 상태에선 x에 어떤 값이 들어가도 가설은 0을 예측하게 됩니다. 즉, 아직 적절한 $W$와 $b$의 값이 아닙니다.



## 2.4 가설 세우기

파이토치 코드 상으로 직선의 방정식에 해당되는 가설을 선언합니다.

$$H(x) = Wx+b$$

```py
hypothesis = x_train * W + b
print(hypothesis)
```
```
tensor([[0.],
        [0.],
        [0.]], grad_fn=<AddBackward0>)
```



## 2.5 비용 함수 선언하기

파이토치 코드 상으로 선형 회귀의 비용 함수에 해당되는 평균 제곱 오차를 선언합니다.

$$cost(W,b) = \frac{1}{n}\sum_{i=1}^{n}[y^{(i)} - H(x^{(i)})]^{2}$$

```py
# 앞서 배운 torch.mean으로 평균을 구한다.
cost = torch.mean((hypothesis - y_train) ** 2) 
print(cost)
```
```
tensor(18.6667, grad_fn=<MeanBackward0>)
```



## 2.6 경사 하강법 구현하기

이제 경사 하강법을 구현합니다. 아래의 'SGD'는 경사 하강법의 일종입니다. lr은 학습률(learning rate)를 의미합니다. 학습 대상인 $W$와 $b$가 SGD의 입력이 됩니다.
```py
optimizer = optim.SGD([W, b], lr=0.01)
```

`optimizer.zero_grad()`를 실행하므로서 미분을 통해 얻은 기울기를 0으로 초기화합니다. 기울기를 초기화해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있습니다. 그 다음 `cost.backward()` 함수를 호출하면 가중치 W와 편향 b에 대한 기울기가 계산됩니다. 그 다음 경사 하강법 최적화 함수 opimizer의 `optimizer.step()` 함수를 호출하여 인수로 들어갔던 W와 b에서 리턴되는 변수들의 기울기에 학습률(learining rate) 0.01을 곱하여 빼줌으로서 업데이트합니다.
```py
# gradient를 0으로 초기화
optimizer.zero_grad() 
# 비용 함수를 미분하여 gradient 계산
cost.backward() 
# W와 b를 업데이트
optimizer.step()
```

### optimizer.zero_grad()가 필요한 이유

파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있습니다.
```py
import torch
w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    z = 2*w

    z.backward()
    print(f'수식을 w로 미분한 값 : {w.grad}')
```
```
수식을 w로 미분한 값 : 2.0
수식을 w로 미분한 값 : 4.0
수식을 w로 미분한 값 : 6.0
수식을 w로 미분한 값 : 8.0
수식을 w로 미분한 값 : 10.0
수식을 w로 미분한 값 : 12.0
수식을 w로 미분한 값 : 14.0
수식을 w로 미분한 값 : 16.0
수식을 w로 미분한 값 : 18.0
수식을 w로 미분한 값 : 20.0
수식을 w로 미분한 값 : 22.0
수식을 w로 미분한 값 : 24.0
수식을 w로 미분한 값 : 26.0
수식을 w로 미분한 값 : 28.0
수식을 w로 미분한 값 : 30.0
수식을 w로 미분한 값 : 32.0
수식을 w로 미분한 값 : 34.0
수식을 w로 미분한 값 : 36.0
수식을 w로 미분한 값 : 38.0
수식을 w로 미분한 값 : 40.0
수식을 w로 미분한 값 : 42.0
```

계속해서 미분값인 2가 누적되는 것을 볼 수 있습니다. 그렇기 때문에 `optimizer.zero_grad()`를 통해 미분값을 계속 0으로 초기화시켜줘야 합니다.



## 2.7 전체 코드

결과적으로 훈련 과정에서 `W`와 `b`는 훈련 데이터와 잘 맞는 직선을 표현하기 위한 적절한 값으로 변화해갑니다.
```py
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
```
```
Epoch    0/2000 W: 0.187, b: 0.080 Cost: 18.666666
Epoch  100/2000 W: 1.746, b: 0.578 Cost: 0.048171
Epoch  200/2000 W: 1.800, b: 0.454 Cost: 0.029767
Epoch  300/2000 W: 1.843, b: 0.357 Cost: 0.018394
Epoch  400/2000 W: 1.876, b: 0.281 Cost: 0.011366
Epoch  500/2000 W: 1.903, b: 0.221 Cost: 0.007024
Epoch  600/2000 W: 1.924, b: 0.174 Cost: 0.004340
Epoch  700/2000 W: 1.940, b: 0.136 Cost: 0.002682
Epoch  800/2000 W: 1.953, b: 0.107 Cost: 0.001657
Epoch  900/2000 W: 1.963, b: 0.084 Cost: 0.001024
Epoch 1000/2000 W: 1.971, b: 0.066 Cost: 0.000633
Epoch 1100/2000 W: 1.977, b: 0.052 Cost: 0.000391
Epoch 1200/2000 W: 1.982, b: 0.041 Cost: 0.000242
Epoch 1300/2000 W: 1.986, b: 0.032 Cost: 0.000149
Epoch 1400/2000 W: 1.989, b: 0.025 Cost: 0.000092
Epoch 1500/2000 W: 1.991, b: 0.020 Cost: 0.000057
Epoch 1600/2000 W: 1.993, b: 0.016 Cost: 0.000035
Epoch 1700/2000 W: 1.995, b: 0.012 Cost: 0.000022
Epoch 1800/2000 W: 1.996, b: 0.010 Cost: 0.000013
Epoch 1900/2000 W: 1.997, b: 0.008 Cost: 0.000008
Epoch 2000/2000 W: 1.997, b: 0.006 Cost: 0.000005
```

에포크(Epoch)는 전체 훈련 데이터가 학습에 한 번 사용된 주기를 말합니다. 현재의 경우 2,000번을 수행했습니다.

최종 훈련 결과를 보면 최적의 기울기 $W$는 2에 가깝고, $b$는 0에 가까운 것을 볼 수 있습니다. 현재 훈련 데이터가 x_train은 [[1], [2], [3]]이고 y_train은 [[2], [4], [6]]인 것을 감안하면 실제 정답은 $W$가 2이고, $b$가 0인 H(x)=2x이므로 거의 정답을 찾은 셈입니다.





# 3. 자동 미분(Autograd)

경사 하강법 코드를 보고있으면 `requires_grad=True`, `backward()` 등이 나옵니다. 이는 파이토치에서 제공하고 있는 자동 미분(Autograd) 기능을 수행하고 있는 것입니다. 모델이 복잡해질수록 경사 하강법을 넘파이 등으로 직접 코딩하는 것은 까다로운 일이지만, 파이토치에서는 이런 수고를 하지 않도록 자동 미분(Autograd)을 지원합니다. 자동 미분을 사용하면 미분 계산을 자동화하여 경사 하강법을 손쉽게 사용할 수 있게 해줍니다.

## 자동 미분(Autograd) 실습

자동 미분에 대해서 실습을 통해 이해해봅시다. 임의로 $2w^{2} + 5$라는 식을 세워보고, $w$에 대해 미분해보겠습니다.

값이 2인 임의의 스칼라 텐서 w를 선언합니다. 이때 `required_grad`를 `True`로 설정합니다. 이는 이 텐서에 대한 기울기를 저장하겠다는 의미입니다. 이렇게 하면 w.grad에 w에 대한 미분값이 저장됩니다.
```py
w = torch.tensor(2.0, requires_grad=True)
```

이제 수식을 정의합니다.
```py
y = w**2
z = 2*y + 5
```

이제 해당 수식을 w에 대해서 미분해야합니다. `.backward()`를 호출하면 해당 수식의 w에 대한 기울기를 계산합니다.
```py
z.backward()
```

이제 w.grad를 출력하면 w가 속한 수식을 w로 미분한 값이 저장된 것을 확인할 수 있습니다.
```py
print(f'수식을 w로 미분한 값 : {w.grad}')
```
```
수식을 w로 미분한 값 : 8.0
```





# 4. 다중 선형 회귀(Multivariable Linear regression)

앞서 다룬 $x$가 1개인 선형 회귀를 단순 선형 회귀(Simple Linear Regression)이라고 하며, 다수의 $x$로부터 $y$를 예측하는 것을 다중 선형 회귀(Multivariable Linear Regression)라고 합니다.

단순 선형 회귀와 다른 점은 독립 변수 x의 개수가 이제 1개가 아니라는 점입니다. 3개의 퀴즈 점수로부터 최종 점수를 예측하는 모델을 만들어보겠습니다. 독립 변수 $x$의 개수가 3라 가정했을때 이를 수식으로 표현하면 아래와 같습니다.

$$H(x)=w_1x_1+w_2x_2+w_3x_3+b$$



## 4.1 파이토치로 다중 선형 회귀 구현하기

위의 식을 보면 이번에는 단순 선형 회귀와 다르게 $x$의 개수가 3개입니다. 그러니까 $x$를 3개 선언합니다.

```py
# 훈련 데이터
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

이제 가중치 $w$와 편향 $b$를 선언합니다. 가중치 $w$도 3개 선언해주어야 합니다.
```py
# 가중치 w와 편향 b 초기화
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

이제 가설, 비용 함수, 옵티마이저를 선언한 후에 경사 하강법을 1,000회 반복합니다.
```py
# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
```
```
Epoch    0/1000 w1: 0.294 w2: 0.294 w3: 0.297 b: 0.003 Cost: 29661.800781
Epoch  100/1000 w1: 0.674 w2: 0.661 w3: 0.676 b: 0.008 Cost: 1.563628
Epoch  200/1000 w1: 0.679 w2: 0.655 w3: 0.677 b: 0.008 Cost: 1.497595
Epoch  300/1000 w1: 0.684 w2: 0.649 w3: 0.677 b: 0.008 Cost: 1.435044
Epoch  400/1000 w1: 0.689 w2: 0.643 w3: 0.678 b: 0.008 Cost: 1.375726
Epoch  500/1000 w1: 0.694 w2: 0.638 w3: 0.678 b: 0.009 Cost: 1.319507
Epoch  600/1000 w1: 0.699 w2: 0.633 w3: 0.679 b: 0.009 Cost: 1.266222
Epoch  700/1000 w1: 0.704 w2: 0.627 w3: 0.679 b: 0.009 Cost: 1.215703
Epoch  800/1000 w1: 0.709 w2: 0.622 w3: 0.679 b: 0.009 Cost: 1.167810
Epoch  900/1000 w1: 0.713 w2: 0.617 w3: 0.680 b: 0.009 Cost: 1.122429
Epoch 1000/1000 w1: 0.718 w2: 0.613 w3: 0.680 b: 0.009 Cost: 1.079390
```

위의 경우 가설을 선언하는 부분인 hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b에서도 x_train의 개수만큼 w와 곱해주도록 작성해준 것을 확인할 수 있습니다.



## 4.2 행렬 연산을 고려하여 파이토치로 구현하기

$x$의 개수가 3개였으니까 x1_train, x2_train, x3_train와 w1, w2, w3를 일일히 선언해주었습니다. 그런데 $x$의 개수가 1,000개라고 가정한다면, 위와 같은 방식을 고수할때 x_train1 ~ x_train1000을 전부 선언하고, w1 ~ w1000을 전부 선언해야 합니다. 다시 말해 $x$와 $w$ 변수 선언만 총 합 2,000개를 해야합니다. 또한 가설을 선언하는 부분에서도 마찬가지로 x_train과 w의 곱셈이 이루어지는 항을 1,000개를 작성해야 합니다. 이는 굉장히 비효율적입니다.

이를 해결하기 위해 행렬 곱셈 연산(또는 벡터의 내적)을 사용합니다. 즉 행렬 연산을 고려하여 파이토치로 재구현해보겠습니다. 이번에는 훈련 데이터 또한 행렬로 선언해야 합니다.
```py
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
```

이전에 x_train을 3개나 구현했던 것과 다르게 이번에는 x_train 하나에 모든 샘플을 전부 선언하였습니다. 다시 말해 (5 x 3) 행렬 $X$을 선언한 것입니다.

x_train과 y_train의 크기(shape)를 출력해보겠습니다.
```py
print(x_train.shape)
print(y_train.shape)
```
```
torch.Size([5, 3])
torch.Size([5, 1])
```

각각 (5 × 3) 행렬과 (5 × 1) 행렬(또는 벡터)의 크기를 가집니다. 이제 가중치 $W$와 편향 $b$를 선언합니다.
```py
# 가중치와 편향 선언
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

여기서 주목할 점은 가중치 $W$의 크기가 (3 × 1) 벡터라는 점입니다. 행렬의 곱셈이 성립되려면 곱셈의 좌측에 있는 행렬의 열의 크기와 우측에 있는 행렬의 행의 크기가 일치해야 합니다. 현재 X_train의 행렬의 크기는 (5 × 3)이며, $W$ 벡터의 크기는 (3 × 1)이므로 두 행렬과 벡터는 행렬곱이 가능합니다. 행렬곱으로 가설을 선언하면 아래와 같습니다.
```py
hypothesis = x_train.matmul(W) + b
```

가설을 행렬곱으로 간단히 정의하였습니다. 이는 앞서 x_train과 w의 곱셈이 이루어지는 각 항을 전부 기재하여 가설을 선언했던 것과 대비됩니다. 이 경우, 사용자가 독립 변수 x의 수를 후에 추가적으로 늘리거나 줄이더라도 위의 가설 선언 코드를 수정할 필요가 없습니다. 이제 해야할 일은 비용 함수와 옵티마이저를 정의하고, 정해진 에포크만큼 훈련을 진행하는 일입니다. 이를 반영한 전체 코드는 다음과 같습니다.
```py
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
```




