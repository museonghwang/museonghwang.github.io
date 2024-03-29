---
title: 자주 사용되는 pytorch 텐서 조작 함수(Tensor Manipulation Functions)
category: Pytorch
tag: pytorch
date: 2022-07-25
---     





# 텐서 연산 및 조작 함수

텐서를 다룰때 주로 사용하는 함수에 대해 익히고 다루는 방법을 살펴보겠습니다.

<br>





# 텐서 연산 관련 함수(Tensor Operations Functions)



## 1. 행렬 곱셈과 곱셈의 차이(Matrix Multiplication Vs. Multiplication)

행렬로 곱셈을 하는 방법은 크게 두 가지가 있습니다.

* 행렬 곱셈(`.matmul`)
* 원소 별 곱셈(`.mul`)

파이토치 텐서의 행렬 곱셈을 보겠습니다. 이는 `matmul()`을 통해 수행합니다.

```py
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

[output]
Shape of Matrix 1:  torch.Size([2, 2])
Shape of Matrix 2:  torch.Size([2, 1])
tensor([[ 5.],
        [11.]])
```

위의 결과는 2 × 2 행렬과 2 × 1 행렬(벡터)의 행렬 곱셈의 결과를 보여줍니다.

행렬 곱셈이 아니라 **element-wise 곱셈**이라는 것이 존재하는데, 이는 **동일한 크기의 행렬이 동일한 위치에 있는 원소끼리 곱하는 것**을 말합니다. 아래는 서로 다른 크기의 행렬이 브로드캐스팅이 된 후에 element-wise 곱셈이 수행되는 것을 보여줍니다. 이는 `*` 또는 `mul()`을 통해 수행합니다.

```py
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

[output]
Shape of Matrix 1:  torch.Size([2, 2])
Shape of Matrix 2:  torch.Size([2, 1])
tensor([[1., 2.],
        [6., 8.]])
tensor([[1., 2.],
        [6., 8.]])
```

m1 행렬의 크기는 `(2, 2)` 이었습니다. m2 행렬의 크기는 `(2, 1)` 였습니다. 이때 element-wise 곱셈을 수행하면, 두 행렬의 크기는 브로드캐스팅이 된 후에 곱셈이 수행됩니다. 더 정확히는 여기서 m2의 크기가 변환됩니다.

```
# 브로드캐스팅 과정에서 m2 텐서가 어떻게 변경되는지 보겠습니다.
[1]
[2]
==> [[1, 1],
     [2, 2]]
```

<br>



## 2. 평균(Mean)

다음은 평균을 구하는 방법으로 Numpy에서의 사용법과 매우 유사합니다. 우선 1차원인 벡터를 선언하여 `.mean()`을 사용하여 원소의 평균을 구합니다.

```py
t = torch.FloatTensor([1, 2])
print(t.mean())

[output]
tensor(1.5000)
```

이번에는 2차원인 행렬을 선언하여 `.mean()`을 사용해봅시다.

```py
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean())

[output]
tensor([[1., 2.],
        [3., 4.]])
tensor(2.5000)
```

이번에는 `dim` 즉, 차원(dimension)을 인자로 주는 경우를 보겠습니다.

```py
print(t.mean(dim=0))

[output]
tensor([2., 3.])
```

`dim=0` 이라는 것은 첫번째 차원을 의미합니다. 행렬에서 첫번째 차원은 '행'을 의미하므로, **인자로 dim을 준다면 해당 차원을 제거한다는 의미가 됩니다.** 다시 말해 행렬에서 '열'만을 남기겠다는 의미가 됩니다. 기존 행렬의 크기는 `(2, 2)` 였지만 이를 수행하면 열의 차원만 보존되면서 `(1, 2)` 가 되며, 이는 `(2,)` 와 같으며 벡터입니다. 열의 차원을 보존하면서 평균을 구하면 아래와 같이 연산합니다.

```
# 실제 연산 과정
t.mean(dim=0)은 입력에서 첫번째 차원을 제거한다.

[[1., 2.],
 [3., 4.]]

1과 3의 평균을 구하고, 2와 4의 평균을 구한다.
결과 ==> [2., 3.]
```

이번에는 인자로 `dim=1` 을 주겠습니다. 이번에는 두번째 차원을 제거합니다. 즉, 열이 제거된 텐서가 되어야 합니다.

```py
print(t.mean(dim=1))

[output]
tensor([1.5000, 3.5000])
```

열의 차원이 제거되어야 하므로 `(2, 2)` 의 크기에서 `(2, 1)` 의 크기가 됩니다. 이번에는 1과 3의 평균을 구하고 3과 4의 평균을 구하게 됩니다. 그렇다면 결과는 아래와 같습니다.

```
# 실제 연산 결과는 (2 × 1)
[[ 1.5 ]
 [ 3.5 ]]
```

하지만 `(2 × 1)` 은 결국 1차원이므로 `(1 × 2)` 와 같이 표현되면서 위와 같이 [1.5, 3.5]로 출력됩니다. 이번에는 `dim=-1` 를 주는 경우를 보겠습니다. 이는 **마지막 차원을 제거한다는 의미이고, 결국 열의 차원을 제거한다는 의미**와 같습니다. 그러므로 위와 출력 결과가 같습니다.

```py
print(t.mean(dim=-1))

[output]
tensor([1.5000, 3.5000])
```

<br>



## 3. 덧셈(Sum)

덧셈(Sum)은 평균(Mean)과 연산 방법이나 인자가 의미하는 바는 정확히 동일합니다.

```py
t = torch.FloatTensor([[3, 2], [1, 4]])
print(t)

[output]
tensor([[3., 2.],
        [1., 4.]])
```

```py
print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거

[output]
tensor(10.)
tensor([4., 6.])
tensor([5., 5.])
tensor([5., 5.])
```

<br>



## 4. 최대(Max)와 아그맥스(ArgMax)

`Max`는 원소의 최대값을 리턴하고, `ArgMax`는 최대값을 가진 인덱스를 리턴합니다.

```py
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

[output]
tensor([[1., 2.],
        [3., 4.]])
```

우선 `(2, 2)` 행렬을 선언하였습니다. 이제 `.max()`를 사용합니다.

```py
print(t.max()) # Returns one value: max

[output]
tensor(4.)
```

이번에는 인자로 `dim=0` 을 주겠습니다. 첫번째 차원을 제거한다는 의미입니다.

```py
print(t.max(dim=0)) # Returns two values: max and argmax

[output]
torch.return_types.max(
values=tensor([3., 4.]),
indices=tensor([1, 1]))
```

행의 차원을 제거한다는 의미이므로 `(1, 2)` 텐서를 만들며, 결과는 [3, 4]입니다.

그런데 [1, 1]이라는 값도 함께 리턴되었습니다. max에 `dim` 인자를 주면 `argmax`도 함께 리턴하는 특징 때문입니다. 첫번째 열에서 3의 인덱스는 1이었습니다. 두번째 열에서 4의 인덱스는 1이었습니다. 그러므로 [1, 1]이 리턴됩니다. 어떤 의미인지 다음과 같습니다.

```
# [1, 1]가 무슨 의미인지 봅시다. 기존 행렬을 다시 상기해봅시다.
[[1, 2],
 [3, 4]]
첫번째 열에서 0번 인덱스는 1, 1번 인덱스는 3입니다.
두번째 열에서 0번 인덱스는 2, 1번 인덱스는 4입니다.
다시 말해 3과 4의 인덱스는 [1, 1]입니다.
```

만약 두 개를 함께 리턴받는 것이 아니라 max 또는 argmax만 리턴받고 싶다면 다음과 같이 리턴값에도 인덱스를 부여하면 됩니다. 0번 인덱스를 사용하면 max 값만 받아올 수 있고, 1번 인덱스를 사용하면 argmax 값만 받아올 수 있습니다.
```py
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])

[output]
Max:  tensor([3., 4.])
Argmax:  tensor([1, 1])
```

이번에는 `dim=1` 로 인자를 주었을 때와 `dim=-1` 로 인자를 주었을 때를 보겠습니다.
```py
print(t.max(dim=1))
print(t.max(dim=-1))

[output]
torch.return_types.max(
values=tensor([2., 4.]),
indices=tensor([1, 1]))

torch.return_types.max(
values=tensor([2., 4.]),
indices=tensor([1, 1]))
```

<br>



## 5. 기타 연산

```py
import math

a = torch.rand(1, 2) * 2 - 1
print(a)
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

[output]
tensor([[-0.8096, -0.5772]])
tensor([[0.8096, 0.5772]])
tensor([[-0., -0.]])
tensor([[-1., -1.]])
tensor([[-0.5000, -0.5000]])
```

```py
print(a)
print(torch.min(a))
print(torch.max(a))
print(torch.mean(a))
print(torch.std(a))
print(torch.prod(a))
print(torch.unique(torch.tensor([1, 2, 3, 1, 2, 2])))

[output]
tensor([[-0.8096, -0.5772]])
tensor(-0.8096)
tensor(-0.5772)
tensor(-0.6934)
tensor(0.1643)
tensor(0.4673)
tensor([1, 2, 3])
```

<br>





# 텐서 조작 관련 함수(Tensor Manipulation Functions)



## 1. 인덱싱(Indexing) - NumPy처럼 인덱싱 형태로 사용가능

```py
x = torch.Tensor([[1, 2],
                  [3, 4]])
print(x)

print(x[0, 0])
print(x[0, 1])
print(x[1, 0])
print(x[1, 1])

print(x[:, 0])
print(x[:, 1])

print(x[0, :])
print(x[1, :])
```
```
tensor([[1., 2.],
        [3., 4.]])
tensor(1.)
tensor(2.)
tensor(3.)
tensor(4.)
tensor([1., 3.])
tensor([2., 4.])
tensor([1., 2.])
tensor([3., 4.])
```

<br>



## 2. 뷰(View) - 원소의 수를 유지하면서 텐서의 크기 변경

파이토치 텐서의 **뷰(View)**는 numpy에서의 Reshape와 같은 역할을 합니다. **텐서의 크기(Shape)를 변경해주는 역할**을 합니다.

```py
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
```

ft라는 이름의 3차원 텐서를 만들었습니다. 크기(shape)를 확인해보겠습니다.

```py
print(ft.shape)

[output]
torch.Size([2, 2, 3])
```

```py
x = torch.randn(4, 5)
print(x)
y = x.view(20)
print(y)
z = x.view(5, -1)
print(z)

[output]
tensor([[ 0.5298,  0.8756, -0.6373,  0.5298,  0.6077],
        [ 0.3980, -0.3816,  2.0473,  0.9791,  0.3316],
        [-0.1470,  0.5306,  0.1874,  1.3221, -1.3989],
        [ 1.0273,  0.2375,  0.4490, -1.6272, -0.8203]])
tensor([ 0.5298,  0.8756, -0.6373,  0.5298,  0.6077,  0.3980, -0.3816,  2.0473,
         0.9791,  0.3316, -0.1470,  0.5306,  0.1874,  1.3221, -1.3989,  1.0273,
         0.2375,  0.4490, -1.6272, -0.8203])
tensor([[ 0.5298,  0.8756, -0.6373,  0.5298],
        [ 0.6077,  0.3980, -0.3816,  2.0473],
        [ 0.9791,  0.3316, -0.1470,  0.5306],
        [ 0.1874,  1.3221, -1.3989,  1.0273],
        [ 0.2375,  0.4490, -1.6272, -0.8203]])
```

<br>



### 2.1 3차원 텐서에서 2차원 텐서로 변경

ft 텐서를 `view`를 사용하여 크기(shape)를 2차원 텐서로 변경해봅시다.

```py
print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경
print(ft.view([-1, 3]).shape)

[output]
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])
torch.Size([4, 3])
```

`view([-1, 3])` 이 가지는 의미는 다음과 같습니다. -1은 첫번째 차원은 사용자가 잘 모르겠으니 파이토치에 맡기겠다는 의미이고, 3은 두번째 차원의 길이는 3을 가지도록 하라는 의미입니다. 다시 말해 현재 3차원 텐서를 2차원 텐서로 변경하되 `(?, 3)` 의 크기로 변경하라는 의미입니다. 결과적으로 `(4, 3)` 의 크기를 가지는 텐서를 얻었습니다.

즉 내부적으로 크기 변환은 다음과 같이 이루어졌습니다. `(2, 2, 3)` -> `(2 × 2, 3)` -> `(4, 3)`

* view는 기본적으로 변경 전과 변경 후의 텐서 안의 원소의 개수가 유지되어야 합니다.
* 파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추합니다.

변경 전 텐서의 원소의 수는 (2 × 2 × 3) = 12개였습니다. 그리고 변경 후 텐서의 원소의 개수 또한 (4 × 3) = 12개였습니다.

<br>



### 2.2 3차원 텐서의 크기 변경

이번에는 3차원 텐서에서 3차원 텐서로 차원은 유지하되, 크기(shape)를 바꾸는 작업을 해보겠습니다. view로 텐서의 크기를 변경하더라도 원소의 수는 유지되어야 합니다.

```py
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

[output]
tensor([[[ 0.,  1.,  2.]],

        [[ 3.,  4.,  5.]],

        [[ 6.,  7.,  8.]],

        [[ 9., 10., 11.]]])
torch.Size([4, 1, 3])
```

<br>



## 3. item

텐서에 값이 단 하나라도 존재하면 숫자값을 얻을 수 있습니다.

```py
x = torch.randn(1)
print(x)
print(x.item())
print(x.dtype)

[output]
tensor([1.3015])
1.3014956712722778
torch.float32
```

**스칼라값 하나만 존재해야 `item()` 사용이 가능합니다.**

```py
x = torch.randn(2)
print(x)
print(x.item())
print(x.dtype)

[output]
tensor([ 0.8509, -0.5549])
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-34-7c023f92a1c8> in <module>()
      1 x = torch.randn(2)
      2 print(x)
----> 3 print(x.item())
      4 print(x.dtype)

ValueError: only one element tensors can be converted to Python scalars
```

<br>



## 4. 스퀴즈(Squeeze) - 1인 차원을 축소(제거)

**스퀴즈는 차원이 1인 경우에는 해당 차원을 제거합니다.** 우선 2차원 텐서를 만들겠습니다.

```py
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

[output]
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1])
```

해당 텐서는 `(3 × 1)` 의 크기를 가집니다. 두번째 차원이 1이므로 squeeze를 사용하면 `(3,)` 의 크기를 가지는 텐서로 변경됩니다.

```py
print(ft.squeeze())
print(ft.squeeze().shape)

[output]
tensor([0., 1., 2.])
torch.Size([3])
```

위의 결과는 1이었던 두번째 차원이 제거되면서 `(3,)` 의 크기를 가지는 텐서로 변경되어 1차원 벡터가 된 것을 보여줍니다.

<br>



## 5. 언스퀴즈(Unsqueeze) - 특정 위치에 1인 차원을 추가한다. 즉 차원을 증가(생성)

**언스퀴즈는 스퀴즈와 정반대입니다. 특정 위치에 1인 차원을 추가할 수 있습니다.**

```py
ft = torch.Tensor([0, 1, 2])
print(ft.shape)

[output]
torch.Size([3])
```

현재는 차원이 1개인 1차원 벡터입니다. 여기에 첫번째 차원에 1인 차원을 추가해보겠습니다. 첫번째 차원의 인덱스를 의미하는 숫자 0을 인자로 넣으면 첫번째 차원에 1인 차원이 추가됩니다.

```py
print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
print(ft.unsqueeze(0).shape)

[output]
tensor([[0., 1., 2.]])
torch.Size([1, 3])
```

위 결과는 `(3,)` 의 크기를 가졌던 1차원 벡터가 `(1, 3)` 의 2차원 텐서로 변경된 것을 보여줍니다. 방금 한 연산을 view로도 구현 가능합니다. 2차원으로 바꾸고 싶으면서 첫번째 차원은 1이기를 원한다면 view에서 `(1, -1)` 을 인자로 사용하면 됩니다.

```py
print(ft.view(1, -1))
print(ft.view(1, -1).shape)

[output]
tensor([[0., 1., 2.]])
torch.Size([1, 3])
```

위의 결과는 unsqueeze와 view가 동일한 결과를 만든 것을 보여줍니다. 이번에는 unsqueeze의 인자로 1을 넣어보겠습니다. 인덱스는 0부터 시작하므로 이는 두번째 차원에 1을 추가하겠다는 것을 의미합니다. 현재 크기는 `(3,)` 이었으므로 두번째 차원에 1인 차원을 추가하면 `(3, 1)` 의 크기를 가지게 됩니다.

```py
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

[output]
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1])
```

이번에는 unsqueeze의 인자로 -1을 넣어보겠습니다. -1은 인덱스 상으로 마지막 차원을 의미합니다. 현재 크기는 `(3,)` 이었으므로 마지막 차원에 1인 차원을 추가하면 `(3, 1)` 의 크기를 가지게 됩니다. 다시 말해 현재 텐서의 경우에는 1을 넣은 경우와 -1을 넣은 경우가 결과가 동일합니다.

```py
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

[output]
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1])
```

맨 뒤에 1인 차원이 추가되면서 1차원 벡터가 `(3, 1)` 의 크기를 가지는 2차원 텐서로 변경되었습니다. 즉, `view()`, `squeeze()`, `unsqueeze()`는 **텐서의 원소 수를 그대로 유지하면서 모양과 차원을 조절합니다.**

다음은 dim을 추가해보겠습니다.

```py
t = torch.rand(3, 3)
print(t)
print(t.shape)

[output]
tensor([[0.6907, 0.8602, 0.4158],
        [0.4249, 0.5767, 0.8503],
        [0.5024, 0.4996, 0.1251]])
torch.Size([3, 3])
```

```py
tensor = t.unsqueeze(dim=0)
print(tensor)
print(tensor.shape)

[output]
tensor([[[0.6907, 0.8602, 0.4158],
         [0.4249, 0.5767, 0.8503],
         [0.5024, 0.4996, 0.1251]]])
torch.Size([1, 3, 3])
```

```py
tensor = t.unsqueeze(dim=1)
print(tensor)
print(tensor.shape)

[output]
tensor([[[0.6907, 0.8602, 0.4158]],

        [[0.4249, 0.5767, 0.8503]],

        [[0.5024, 0.4996, 0.1251]]])
torch.Size([3, 1, 3])
```

```py
tensor = t.unsqueeze(dim=2)
print(tensor)
print(tensor.shape)

[output]
tensor([[[0.6907],
         [0.8602],
         [0.4158]],

        [[0.4249],
         [0.5767],
         [0.8503]],

        [[0.5024],
         [0.4996],
         [0.1251]]])
torch.Size([3, 3, 1])
```

<br>



## 6. 연결하기(concatenate)

**텐서를 결합하는 메소드로, 쌓을 dim이 존재해야 합니다. 해당하는 차원을 늘려준 후 결합합니다.**

```py
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
```

이제 두 텐서를 `torch.cat([ ])`를 통해 연결해보겠습니다. 그런데 연결 방법은 한 가지만 있는 것이 아닙니다. `torch.cat` 은 어느 차원을 늘릴 것인지를 인자로 줄 수 있습니다. 예를 들어 `dim=0` 은 첫번째 차원을 늘리라는 의미를 담고있습니다.

```py
print(torch.cat([x, y], dim=0))

[output]
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
```

`dim=0` 을 인자로 했더니 두 개의 `(2 × 2)` 텐서가 `(4 × 2)` 텐서가 된 것을 볼 수 있습니다. 이번에는 `dim=1` 을 인자로 주겠습니다.

```py
print(torch.cat([x, y], dim=1))

[output]
tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
```

`dim=1` 을 인자로 했더니 두 개의 (2 × 2) 텐서가 (2 × 4) 텐서가 된 것을 볼 수 있습니다.

딥러닝에서는 주로 모델의 입력 또는 중간 연산에서 두 개의 텐서를 연결하는 경우가 많습니다. **두 텐서를 연결해서 입력으로 사용하는 것은 두 가지의 정보를 모두 사용한다는 의미를 가지고 있습니다.**

<br>



## 7. 스택킹(Stacking)

**연결(concatenate)을 하는 또 다른 방법**입니다. 때로는 연결을 하는 것보다 스택킹이 더 편리할 때가 있는데, 이는 스택킹이 많은 연산을 포함하고 있기 때문입니다. 크기가 `(2,)` 로 모두 동일한 3개의 벡터를 만듭니다.

```py
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
```

이제 `torch.stack`을 통해서 3개의 벡터를 모두 Stacking 해보겠습니다.

```py
print(torch.stack([x, y, z]))

[output]
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
```

위 결과는 3개의 벡터가 순차적으로 쌓여 `(3 × 2)` 텐서가 된 것을 보여줍니다. 스택킹은 사실 **많은 연산을 한 번에 축약**하고 있습니다. 예를 들어 위 작업은 아래의 코드와 동일한 작업입니다.

```py
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
```

x, y, z는 기존에는 전부 `(2,)` 의 크기를 가졌습니다. 그런데 `.unsqueeze(0)` 을 하므로서 3개의 벡터는 전부 `(1, 2)` 의 크기의 2차원 텐서로 변경됩니다. 여기에 연결(concatenate)를 의미하는 `cat` 을 사용하면 `(3 x 2)` 텐서가 됩니다.

```
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
```

위에서는 `torch.stack([x, y, z])` 라는 한 번의 커맨드로 수행했지만, 연결(concatenate)로 이를 구현하려고 했더니 꽤 복잡해졌습니다.

스택킹에 추가적으로 dim을 인자로 줄 수도 있습니다. 이번에는 `dim=1` 인자를 주겠습니다. 이는 두번째 차원이 증가하도록 쌓으라는 의미로 해석할 수 있습니다.

```py
print(torch.stack([x, y, z], dim=1))

[output]
tensor([[1., 2., 3.],
        [4., 5., 6.]])
```

위의 결과는 두번째 차원이 증가하도록 스택킹이 된 결과를 보여줍니다. 결과적으로 `(2 × 3)` 텐서가 됩니다.

<br>



## 8. chunk

텐서를 여러 개로 나눌 때 사용합니다. (몇 개로 나눌 것인가?)

```py
tensor = torch.rand(3, 6)
print(tensor)

t1, t2, t3 = torch.chunk(tensor, 3, dim=0)
print(t1)
print(t2)
print(t3)

[output]
tensor([[0.7229, 0.1978, 0.0996, 0.0198, 0.9782, 0.4380],
        [0.7233, 0.8010, 0.0060, 0.4960, 0.1566, 0.5789],
        [0.0613, 0.8835, 0.7048, 0.8353, 0.4018, 0.3844]])
tensor([[0.7229, 0.1978, 0.0996, 0.0198, 0.9782, 0.4380]])
tensor([[0.7233, 0.8010, 0.0060, 0.4960, 0.1566, 0.5789]])
tensor([[0.0613, 0.8835, 0.7048, 0.8353, 0.4018, 0.3844]])
```

```py
tensor = torch.rand(3, 6)
print(tensor)

t1, t2, t3 = torch.chunk(tensor, 3, dim=1)
print(t1)
print(t2)
print(t3)

[output]
tensor([[0.2103, 0.3330, 0.2791, 0.9946, 0.2185, 0.7475],
        [0.7396, 0.6518, 0.1193, 0.9112, 0.9514, 0.7630],
        [0.3139, 0.5222, 0.6987, 0.8860, 0.0796, 0.5894]])
tensor([[0.2103, 0.3330],
        [0.7396, 0.6518],
        [0.3139, 0.5222]])
tensor([[0.2791, 0.9946],
        [0.1193, 0.9112],
        [0.6987, 0.8860]])
tensor([[0.2185, 0.7475],
        [0.9514, 0.7630],
        [0.0796, 0.5894]])
```

<br>



## 9. split

`chunk`와 동일한 기능이지만 조금 다름 (텐서의 크기는 몇인가?)

```py
tensor = torch.rand(3, 6)
t1, t2 = torch.split(tensor, 3, dim=1)

print(tensor)
print(t1)
print(t2)

[output]
tensor([[0.7732, 0.0393, 0.7892, 0.9389, 0.0273, 0.1751],
        [0.0814, 0.2443, 0.5015, 0.0702, 0.0171, 0.1885],
        [0.3454, 0.2807, 0.1119, 0.1323, 0.3292, 0.7515]])
tensor([[0.7732, 0.0393, 0.7892],
        [0.0814, 0.2443, 0.5015],
        [0.3454, 0.2807, 0.1119]])
tensor([[0.9389, 0.0273, 0.1751],
        [0.0702, 0.0171, 0.1885],
        [0.1323, 0.3292, 0.7515]])
```

<br>



## 10. ones_like와 zeros_like - 0으로 채워진 텐서와 1로 채워진 텐서

`(2 × 3)` 텐서를 만듭니다.

```py
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

[output]
tensor([[0., 1., 2.],
        [2., 1., 0.]])
```

위 텐서에 `ones_like`를 하면 동일한 크기(shape)지만 1으로만 값이 채워진 텐서를 생성합니다.

```py
print(torch.ones_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채우기

[output]
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

위 텐서에 `zeros_like`를 하면 동일한 크기(shape)지만 0으로만 값이 채워진 텐서를 생성합니다.

```py
print(torch.zeros_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기

[output]
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

<br>



## 11. In-place Operation (덮어쓰기 연산)

`(2 × 2)` 텐서를 만들고 x에 저장합니다.

```py
x = torch.FloatTensor([[1, 2], [3, 4]])
```

곱하기 연산을 한 값과 기존의 값을 출력해보겠습니다.

```py
print(x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
print(x) # 기존의 값 출력

[output]
tensor([[2., 4.],
        [6., 8.]])
tensor([[1., 2.],
        [3., 4.]])
```

첫번째 출력은 곱하기 2가 수행된 결과를 보여주고, 두번째 출력은 기존의 값이 그대로 출력된 것을 확인할 수 있습니다. 곱하기 2를 수행했지만 이를 x에다가 다시 저장하지 않았으니, 곱하기 연산을 하더라도 기존의 값 x는 변하지 않는 것이 당연합니다.

그런데 연산 뒤에 `_`를 붙이면 기존의 값을 덮어쓰기 합니다.

```py
print(x.mul_(2.))  # 곱하기 2를 수행한 결과를 변수 x에 값을 저장하면서 결과를 출력
print(x)  # 변경된 값 출력

[output]
tensor([[2., 4.],
        [6., 8.]])
tensor([[2., 4.],
        [6., 8.]])
```

이번에는 x의 값이 덮어쓰기 되어 2 곱하기 연산이 된 결과가 출력됩니다.

<br>



## 12. torch ↔ numpy

- Torch Tensor(텐서)를 NumPy array(배열)로 변환 가능
    - `numpy()`
    - `from_numpy()`
- Tensor가 CPU상에 있다면 NumPy 배열은 메모리 공간을 공유하므로 하나가 변하면, 다른 하나도 변함

```py
a = torch.ones(7)
print(a)

[output]
tensor([1., 1., 1., 1., 1., 1., 1.])
```

```py
b = a.numpy()
print(b)

[output]
[1. 1. 1. 1. 1. 1. 1.]
```

```py
a.add_(1)
print(a)
print(b)

[output]
tensor([2., 2., 2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2. 2. 2.]
```

```py
import numpy as np

a = np.ones(7)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

[output]
[2. 2. 2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2., 2., 2.], dtype=torch.float64)
```




