---
title: Linear Regression 학습을 위한 pytorch 기본2
category: Pytorch
tag: pytorch
date: 2022-07-25
---     





# 1. nn.Module로 구현하는 선형 회귀

파이토치에서 이미 구현되어져 제공되고 있는 함수들을 불러오는 것으로 더 쉽게 선형 회귀 모델을 구현해보겠습니다.



## 1.1 단순 선형 회귀 구현하기

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
```

데이터를 선언합니다. 아래 데이터는 $y=2x$를 가정된 상태에서 만들어진 데이터로 정답이 `W=2`, `b=0`임을 알고 있는 상황입니다. 모델이 이 두 W와 b의 값을 제대로 찾아내도록 하는 것이 목표입니다.
```py
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
```

데이터를 정의하였으니 이제 선형 회귀 모델을 구현할 차례입니다. `nn.Linear()`는 입력의 차원, 출력의 차원을 인수로 받습니다.
```py
# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
model = nn.Linear(1,1)
```

위 `torch.nn.Linear` 인자로 1, 1을 사용하였습니다. 하나의 입력 $x$에 대해서 하나의 출력 $y$을 가지므로, 입력 차원과 출력 차원 모두 1을 인수로 사용하였습니다. model에는 가중치 `W`와 편향 `b`가 저장되어져 있습니다. 이 값은 `model.parameters()`라는 함수를 사용하여 불러올 수 있는데, 한 번 출력해보겠습니다.
```py
print(list(model.parameters()))
```
```
[Parameter containing:
tensor([[-0.1119,  0.2710, -0.5435]], requires_grad=True), Parameter containing:
tensor([0.3462], requires_grad=True)]
```
2개의 값이 출력되는데 첫번째 값이 `W`고, 두번째 값이 `b`에 해당됩니다. 두 값 모두 현재는 랜덤 초기화가 되어져 있습니다. 그리고 두 값 모두 학습의 대상이므로 `requires_grad=True`가 되어져 있는 것을 볼 수 있습니다.

이제 옵티마이저를 정의합니다. `model.parameters()`를 사용하여 W와 b를 전달합니다. 학습률(learning rate)은 0.01로 정합니다.
```py
# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
```
```py
# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```
```
Epoch    0/2000 Cost: 13.103540
... 중략 ...
Epoch 2000/2000 Cost: 0.000000
```

학습이 완료되었습니다. Cost의 값이 매우 작습니다. $W$와 $b$의 값도 최적화가 되었는지 확인해봅시다. $x$에 임의의 값 4를 넣어 모델이 예측하는 $y$의 값을 확인해보겠습니다.
```py
# 임의의 입력 4를 선언
new_var =  torch.FloatTensor([[4.0]]) 
# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) # forward 연산
# y = 2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
print("훈련 후 입력이 4일 때의 예측값 :", pred_y) 
```
```
훈련 후 입력이 4일 때의 예측값 : tensor([[7.9989]], grad_fn=<AddmmBackward0>)
```

이 문제의 정답은 $y=2x$가 정답이므로 $y$값이 8에 가까우면 $W$와 $b$의 값이 어느정도 최적화가 된 것으로 볼 수 있습니다. 실제로 예측된 $y$값은 7.9989로 8에 매우 가깝습니다.

이제 학습 후의 $W$와 $$의 값을 출력해보겠습니다.
```py
print(list(model.parameters()))
```
```
[Parameter containing:
tensor([[1.9994]], requires_grad=True), Parameter containing:
tensor([0.0014], requires_grad=True)]
```

$W$의 값이 2에 가깝고, $b$의 값이 0에 가까운 것을 볼 수 있습니다.

*   $H(x)$  식에 입력 $x$로부터 예측된 $y$를 얻는 것을 forward 연산이라고 합니다.
*   학습 전, prediction = model(x_train)은 x_train으로부터 예측값을 리턴하므로 forward 연산입니다.
*   학습 후, pred_y = model(new_var)는 임의의 값 new_var로부터 예측값을 리턴하므로 forward 연산입니다.
*   학습 과정에서 비용 함수를 미분하여 기울기를 구하는 것을 backward 연산이라고 합니다.
*   cost.backward()는 비용 함수로부터 기울기를 구하라는 의미이며 backward 연산입니다.



## 1.2 다중 선형 회귀 구현하기

`nn.Linear()`와 `nn.functional.mse_loss()`로 다중 선형 회귀를 구현해봅니다. 사실 코드 자체는 달라지는 건 거의 없는데, `nn.Linear()`의 인자값과 학습률(learning rate)만 조절해주었습니다.

데이터를 선언해줍니다. 여기서는 3개의 $x$로부터 하나의 $y$를 예측하는 문제입니다. 즉, 가설 수식은 $H(x) = w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + b$입니다.
```py
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

데이터를 정의하였으니 이제 선형 회귀 모델을 구현할 차례입니다. `nn.Linear()`는 입력의 차원, 출력의 차원을 인수로 받습니다.
```py
# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3,1)
```

위 `torch.nn.Linear` 인자로 3, 1을 사용하였습니다. 3개의 입력 x에 대해서 하나의 출력 y을 가지므로, 입력 차원은 3, 출력 차원은 1을 인수로 사용하였습니다. model에는 3개의 가중치 w와 편향 b가 저장되어져 있습니다. 이 값은 `model.parameters()`라는 함수를 사용하여 불러올 수 있는데, 한 번 출력해보겠습니다.
```py
print(list(model.parameters()))
```
```
[Parameter containing:
tensor([[-0.1119,  0.2710, -0.5435]], requires_grad=True), Parameter containing:
tensor([0.3462], requires_grad=True)]
```

첫번째 출력되는 것이 3개의 $w$고, 두번째 출력되는 것이 $b$에 해당됩니다. 두 값 모두 현재는 랜덤 초기화가 되어져 있습니다. 그리고 두 출력 결과 모두 학습의 대상이므로 `requires_grad=True`가 되어져 있는 것을 볼 수 있습니다.

이제 옵티마이저를 정의합니다. `model.parameters()`를 사용하여 3개의 w와 b를 전달합니다. 학습률(learning rate)은 0.00001로 정합니다. 파이썬 코드로는 1e-5로도 표기합니다.
```py
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
```
```py
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```
```
Epoch    0/2000 Cost: 31667.597656
... 중략 ...
Epoch 2000/2000 Cost: 0.199777
```

학습이 완료되었습니다. Cost의 값이 매우 작습니다. 3개의 w와 b의 값도 최적화가 되었는지 확인해봅시다. $x$에 임의의 입력 [73, 80, 75]를 넣어 모델이 예측하는 $y$의 값을 확인해보겠습니다.
```py
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
```
```
훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[153.7184]], grad_fn=<AddmmBackward0>)
```

사실 3개의 값 73, 80, 75는 훈련 데이터로 사용되었던 값입니다. 당시 y의 값은 152였는데, 현재 예측값이 151이 나온 것으로 보아 어느정도는 3개의 w와 b의 값이 최적화 된것으로 보입니다. 이제 학습 후의 3개의 w와 b의 값을 출력해보겠습니다.
```py
print(list(model.parameters()))
```
```
[Parameter containing:
tensor([[0.8541, 0.8475, 0.3096]], requires_grad=True), Parameter containing:
tensor([0.3568], requires_grad=True)]
```





# 2. 클래스로 파이토치 모델 구현하기

파이토치의 대부분의 구현체들은 대부분 모델을 생성할 때 클래스(Class)를 사용하고 있습니다. 앞의 선형 회귀를 클래스로 구현해보겠습니다. 앞서 구현한 코드와 다른 점은 오직 클래스로 모델을 구현했다는 점입니다.



## 2.1 모델을 클래스로 구현하기

앞서 단순 선형 회귀 모델은 다음과 같이 구현했었습니다.

```py
# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
model = nn.Linear(1,1)
```

이를 클래스로 구현하면 다음과 같습니다.
```py
class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속받는 파이썬 클래스
    def __init__(self): #
        super().__init__()
        self.linear = nn.Linear(1, 1) # 단순 선형 회귀이므로 input_dim=1, output_dim=1.

    def forward(self, x):
        return self.linear(x)
```
```py
model = LinearRegressionModel()
```

위와 같은 클래스를 사용한 모델 구현 형식은 대부분의 파이토치 구현체에서 사용하고 있는 방식으로 반드시 숙지할 필요가 있습니다.

클래스(class) 형태의 모델은 `nn.Module` 을 상속받습니다. 그리고 `__init__()`에서 모델의 구조와 동적을 정의하는 생성자를 정의합니다. 이는 파이썬에서 객체가 갖는 속성값을 초기화하는 역할로, 객체가 생성될 때 자동으호 호출됩니다. `super()` 함수를 부르면 여기서 만든 클래스는 `nn.Module` 클래스의 속성들을 가지고 초기화 됩니다. `foward()` 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수입니다. 이 `forward()` 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행이됩니다. 예를 들어 model이란 이름의 객체를 생성 후, model(입력 데이터)와 같은 형식으로 객체를 호출하면 자동으로 forward 연산이 수행됩니다.

*   $H(x)$  식에 입력 $x$로부터 예측된 $y$를 얻는 것을 forward 연산이라고 합니다.


앞서 다중 선형 회귀 모델은 다음과 같이 구현했었습니다.
```py
# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3,1)
```

이를 클래스로 구현하면 다음과 같습니다.
```py
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

    def forward(self, x):
        return self.linear(x)
```
```py
model = MultivariateLinearRegressionModel()
```



## 2.2 단순 선형 회귀 클래스로 구현하기

이제 모델을 클래스로 구현한 코드를 보겠습니다. 달라진 점은 모델을 클래스로 구현했다는 점 뿐입니다. 다른 코드는 전부 동일합니다.
```py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
```
```py
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
```
```py
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```
```py
model = LinearRegressionModel()
```
```py
# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
```
```py
# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```
```
Epoch    0/2000 Cost: 4.610449
... 중략 ...
Epoch 2000/2000 Cost: 0.000004
```



## 2.3 다중 선형 회귀 클래스로 구현하기

다중 선형 회귀도 마찬가지입니다.
```py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
```
```py
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```
```py
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

    def forward(self, x):
        return self.linear(x)
```
```py
model = MultivariateLinearRegressionModel()
```
```py
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
```
```py
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```
```
Epoch    0/2000 Cost: 52396.968750
... 중략 ...
Epoch 2000/2000 Cost: 0.635827
```





# 3. 미니 배치와 데이터 로드(Mini Batch and Data Load)



### 3.1 미니 배치와 배치 크기(Mini Batch and Batch Size)

데이터가 수십만개 이상이라면 전체 데이터에 대해서 경사 하강법을 수행하는 것은 매우 느릴 뿐만 아니라 많은 계산량이 필요합니다. 정말 어쩌면 메모리의 한계로 계산이 불가능한 경우도 있을 수 있습니다. 그렇기 때문에 전체 데이터를 더 작은 단위로 나누어서 해당 단위로 학습하는 개념이 나오게 되었습니다. 이 단위를 미니 배치(Mini Batch)라고 합니다.

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/181143158-183dde9b-c026-40e6-b89f-e3317569e221.png">
</p>

위의 그림은 전체 데이터를 미니 배치 단위로 나누는 것을 보여줍니다. 미니 배치 학습을 하게되면 미니 배치만큼만 가져가서 미니 배치에 대한 대한 비용(cost)를 계산하고, 경사 하강법을 수행합니다. 그리고 다음 미니 배치를 가져가서 경사 하강법을 수행하고 마지막 미니 배치까지 이를 반복합니다. 이렇게 전체 데이터에 대한 학습이 1회 끝나면 1 에포크(Epoch)(전체 훈련 데이터가 학습에 한 번 사용된 주기)가 끝나게 됩니다.

미니 배치 학습에서는 미니 배치의 개수만큼 경사 하강법을 수행해야 전체 데이터가 한 번 전부 사용되어 1 에포크(Epoch)가 됩니다. 미니 배치의 개수는 결국 미니 배치의 크기를 몇으로 하느냐에 따라서 달라지는데 미니 배치의 크기를 배치 크기(batch size)라고 합니다.

* 전체 데이터에 대해서 한 번에 경사 하강법을 수행하는 방법을 '배치 경사 하강법'이라고 부릅니다. 반면, 미니 배치 단위로 경사 하강법을 수행하는 방법을 '미니 배치 경사 하강법'이라고 부릅니다.
* 배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터를 사용하므로 가중치 값이 최적값에 수렴하는 과정이 매우 안정적이지만, 계산량이 너무 많이 듭니다. 미니 배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터의 일부만을 보고 수행하므로 최적값으로 수렴하는 과정에서 값이 조금 헤매기도 하지만 훈련 속도가 빠릅니다.
* 배치 크기는 보통 2의 제곱수를 사용합니다. ex) 2, 4, 8, 16, 32, 64... 그 이유는 CPU와 GPU의 메모리가 2의 배수이므로 배치크기가 2의 제곱수일 경우에 데이터 송수신의 효율을 높일 수 있다고 합니다.



### 3.2 이터레이션(Iteration)

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/181143448-2945763e-fa66-4e25-93a3-a00929e4351a.png">
</p>

위의 그림은 에포크와 배치 크기와 이터레이션(iteration)의 관계를 보여줍니다. 이터레이션은 한 번의 에포크 내에서 이루어지는 매개변수인 가중치 $W$와 $b$의 업데이트 횟수입니다. 전체 데이터가 2,000일 때 배치 크기를 200으로 한다면 이터레이션의 수는 총 10개입니다. 이는 한 번의 에포크 당 매개변수 업데이트가 10번 이루어짐을 의미합니다.



### 3.3 데이터 로드하기(Data Load)

파이토치에서는 데이터를 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 `데이터셋(Dataset)`과 `데이터로더(DataLoader)`를 제공합니다. 이를 사용하면 미니 배치 학습, 데이터 셔플(shuffle), 병렬 처리까지 간단히 수행할 수 있습니다. 기본적인 사용 방법은 `Dataset`을 정의하고, 이를 `DataLoader`에 전달하는 것입니다.

여기서는 텐서를 입력받아 Dataset의 형태로 변환해주는 `TensorDataset`을 사용해보겠습니다. TensorDataset과 DataLoader를 임포트합니다.
```py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
```

TensorDataset은 기본적으로 텐서를 입력으로 받습니다. 텐서 형태로 데이터를 정의합니다.
```py
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
```

이제 이를 TensorDataset의 입력으로 사용하고 dataset으로 저장합니다.
```py
dataset = TensorDataset(x_train, y_train)
```

파이토치의 데이터셋을 만들었다면 데이터로더를 사용 가능합니다. 데이터로더는 기본적으로 2개의 인자를 입력받는데 하나는 데이터셋, 미니 배치의 크기입니다. 이때 미니 배치의 크기는 통상적으로 2의 배수를 사용합니다. (ex) 64, 128, 256...) 그리고 추가적으로 많이 사용되는 인자로 shuffle이 있습니다. shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿉니다.
```py
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

이제 모델과 옵티마이저를 설계합니다.
```py
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
```

이제 훈련을 진행합니다.
```py
nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
            ))
```
```
Epoch    0/20 Batch 1/3 Cost: 26085.919922
Epoch    0/20 Batch 2/3 Cost: 3660.022949
Epoch    0/20 Batch 3/3 Cost: 2922.390869
... 중략 ...
Epoch   20/20 Batch 1/3 Cost: 6.315856
Epoch   20/20 Batch 2/3 Cost: 13.519956
Epoch   20/20 Batch 3/3 Cost: 4.262849
```

Cost의 값이 점차 작아집니다. 이제 모델의 입력으로 임의의 값을 넣어 예측값을 확인합니다.
```py
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
```
```
훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[153.2754]], grad_fn=<AddmmBackward0>)
```





# 4. 커스텀 데이터셋(Custom Dataset)

파이토치에서는 데이터셋을 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 `torch.utils.data.Dataset`과 `torch.utils.data.DataLoader`를 제공하며, 이를 사용하면 미니 배치 학습, 데이터 셔플(shuffle), 병렬 처리까지 간단히 수행할 수 있습니다. 기본적인 사용 방법은 `Dataset`을 정의하고, 이를 `DataLoader`에 전달하는 것입니다.



### 4.1 커스텀 데이터셋(Custom Dataset)

그런데 `torch.utils.data.Dataset`을 상속받아 직접 커스텀 데이터셋(Custom Dataset)을 만드는 경우도 있습니다. `torch.utils.data.Dataset`은 파이토치에서 데이터셋을 제공하는 추상 클래스입니다. Dataset을 상속받아 다음 메소드들을 오버라이드 하여 커스텀 데이터셋을 만들어보겠습니다.

커스텀 데이터셋을 만들 때, 일단 가장 기본적인 뼈대는 아래와 같습니다. 여기서 필요한 기본적인 define은 3개입니다.
```py
class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self):return

    def __len__(self):return

    def __getitem__(self, idx):return
```

이를 좀 더 자세히 봅시다.
```py
class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self):return
    #  데이터셋의 전처리를 해주는 부분

    def __len__(self):return
    #  데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분

    def __getitem__(self, idx):return
    #  데이터셋에서 특정 1개의 샘플을 가져오는 함수
```

*   len(dataset)을 했을 때 데이터셋의 크기를 리턴할 **len**
*   dataset[i]을 했을 때 i번째 샘플을 가져오도록 하는 인덱싱을 위한 **get_item**



### 4.2 커스텀 데이터셋(Custom Dataset)으로 선형 회귀 구현하기

```py
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
```
```py
# Dataset 상속
class CustomDataset(Dataset): 
    def __init__(self):
        self.x_data = [[73, 80, 75],
                        [93, 88, 93],
                        [89, 91, 90],
                        [96, 98, 100],
                        [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
```
```py
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```
```py
model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
```
```py
nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
            ))
```
```
Epoch    0/20 Batch 1/3 Cost: 27697.917969
Epoch    0/20 Batch 2/3 Cost: 4941.327148
Epoch    0/20 Batch 3/3 Cost: 1006.548645
... 중략 ...
Epoch   20/20 Batch 1/3 Cost: 1.181388
Epoch   20/20 Batch 2/3 Cost: 4.077807
Epoch   20/20 Batch 3/3 Cost: 0.055477
```
```py
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
```
```
훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[149.6803]], grad_fn=<AddmmBackward0>)
```




