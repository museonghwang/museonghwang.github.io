---
title: [Pytorch] 텐서 조작하기(Tensor Manipulation)
category: Pytorch
tag: pytorch
date: 2022-07-26
---     





# 텐서 조작하기(Tensor Manipulation)

벡터, 행렬, 텐서의 개념에 대해서 이해하고, Numpy와 파이토치로 벡터, 행렬, 텐서를 다루는 방법에 대해서 이해합니다.


## 1. 벡터, 행렬 그리고 텐서(Vector, Matrix and Tensor)


<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/180890436-bd4170e9-0c3e-4568-a314-c1f0ed2e7d8f.png">
</p>

딥 러닝을 하게 되면 다루게 되는 가장 기본적인 단위는 벡터, 행렬, 텐서입니다. 차원이 없는 값을 스칼라, 1차원으로 구성된 값을 벡터라고 합니다. 2차원으로 구성된 값을 행렬(Matrix)라고 합니다. 그리고 3차원이 되면 우리는 텐서(Tensor)라고 부릅니다.





## 2. 넘파이로 텐서 만들기(벡터와 행렬 만들기)

PyTorch로 텐서를 만들어보기 전에 우선 Numpy로 텐서를 만들어보겠습니다. 우선 numpy를 임포트합니다.
```py
import numpy as np
```

Numpy로 텐서를 만드는 방법은 간단한데 [숫자, 숫자, 숫자]와 같은 형식으로 만들고 이를 np.array()로 감싸주면 됩니다.



### 1D with Numpy

Numpy로 1차원 텐서인 벡터를 만들어보겠습니다.
```py
t = np.array([0., 1., 2., 3., 4., 5., 6.])
# 파이썬으로 설명하면 list를 생성해서 np.array로 1차원 array로 변환함
print(t)
```
```
[0. 1. 2. 3. 4. 5. 6.]
```

이제 1차원 벡터의 차원과 크기를 출력해보겠습니다.
```py
print('Rank of t: ', t.ndim) #1차원 벡터
print('Shape of t: ', t.shape)
```
```
Rank of t:  1
Shape of t:  (7,)
```

`.ndim`은 몇 차원인지를 출력합니다. 1차원은 벡터, 2차원은 행렬, 3차원은 3차원 텐서였습니다. 현재는 벡터이므로 1차원이 출력됩니다.
`.shape`는 크기를 출력합니다. (7, )는 (1, 7)을 의미합니다. 다시 말해 (1 × 7)의 크기를 가지는 벡터입니다.



### 2D with Numpy

Numpy로 2차원 행렬을 만들어보겠습니다.
```py
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
```
```
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]
 [10. 11. 12.]]
```
```py
print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)
```
```
Rank  of t:  2
Shape of t:  (4, 3)
```
현재는 행렬이므로 2차원이 출력되며, shape은(4, 3)입니다. 다른 표현으로는 (4 × 3)입니다. 이는 행렬이 4행 3열임을 의미합니다.





## 파이토치 텐서 선언하기(PyTorch Tensor Allocation)

파이토치는 Numpy와 매우 유사합니다. 하지만 더 낫습니다(better). 우선 torch를 임포트합니다.
```py
import torch
```

Numpy를 사용하여 진행했던 실습을 파이토치로 똑같이 해봅시다.



### 0D with Numpy: Scalar(스칼라)

* 스칼라는 흔히 알고 있는 상수값.
* 즉, 하나의 값을 표현할 때 1개의 수치로 표현한 것.

파이토치로 0차원 텐서인 스칼라(사실은 1차원 텐서인 벡터)를 만들어봅시다.
```py
# 스칼라 값 정의
scalar1 = torch.tensor([1.])
print(scalar1)

scalar2 = torch.tensor([3.])
print(scalar2)
```
```
tensor([1.])
tensor([3.])
```

`dim()`을 사용하면 현재 텐서의 차원을 보여줍니다. `shape`나 `size()`를 사용하면 크기를 확인할 수 있습니다.
```py
print(scalar1.dim())  # rank. 즉, 차원
print(scalar1.shape)  # shape
print(scalar1.size()) # shape
```
```
1
torch.Size([1])
torch.Size([1])
```

스칼라라는 의미로 숫자 한개만을 선언을 했지만 내부적으로 크기가 1인 벡터로 인식합니다. 즉 현재 1차원 텐서이며, 원소는 1개 입니다.

다음은 스칼라 값 간의 사칙연산을 해보겠습니다.
```py
# 스칼라 값 간의 사칙연산: +, -, *, /
add_scalar = scalar1 + scalar2
print(add_scalar)

sub_scalar = scalar1 - scalar2
print(sub_scalar)

mul_scalar = scalar1 * scalar2
print(mul_scalar)

div_scalar = scalar1 / scalar2
print(div_scalar)
```
```
tensor([4.])
tensor([-2.])
tensor([3.])
tensor([0.3333])
```
```py
# 스칼라 값 간의 사칙연산: torch 모듈에 내장된 메서드 이용해 계산
print(torch.add(scalar1, scalar2))
print(torch.sub(scalar1, scalar2))
print(torch.mul(scalar1, scalar2))
print(torch.div(scalar1, scalar2))
```
```
tensor([4.])
tensor([-2.])
tensor([3.])
tensor([0.3333])
```



### 1D with Numpy: Vector(벡터)

* 벡터는 하나의 값을 표현할 때 2개 이상의 수치로 표현한 것.
* 스칼라의 형태와 동일한 속성을 갖고 있지만, 여러 수치 값을 이용해 표현하는 방식

파이토치로 1차원 텐서인 벡터를 만들어봅시다.
```py
# 벡터 값 정의
vector1 = torch.tensor([1., 2., 3.])
print(vector1)

vector2 = torch.tensor([4., 5., 6.])
print(vector2)
```
```
tensor([1.])
tensor([3.])
```

마찬가지로 현재 텐서의 차원과 크기를 확인할 수 있습니다.
```py
print(vector1.dim())  # rank. 즉, 차원
print(vector1.shape)  # shape
print(vector1.size()) # shape
```
```
1
torch.Size([3])
torch.Size([3])
```

현재 1차원 텐서이며, 원소는 3개입니다.

다음은 벡터 값 간의 사칙연산을 해보겠습니다.
```py
# 벡터 값 간의 사칙연산: +, -, *, /
# 여기서 곱셈과 나눗셈은 각 요소별로(element-wise) 연산된다.
add_vector = vector1 + vector2
print(add_vector)

sub_vector = vector1 - vector2
print(sub_vector)

mul_vector = vector1 * vector2
print(mul_vector)

div_vector = vector1 / vector2
print(div_vector)
```
```
tensor([5., 7., 9.])
tensor([-3., -3., -3.])
tensor([ 4., 10., 18.])
tensor([0.2500, 0.4000, 0.5000])
```
```py
# 벡터 값 간의 사칙연산: torch 모듈에 내장된 메서드 이용해 계산
print(torch.add(vector1, vector2))
print(torch.sub(vector1, vector2))
print(torch.mul(vector1, vector2))
print(torch.div(vector1, vector2))
print(torch.dot(vector1, vector2))# 벡터의 내적
```
```
tensor([5., 7., 9.])
tensor([-3., -3., -3.])
tensor([ 4., 10., 18.])
tensor([0.2500, 0.4000, 0.5000])
tensor(32.)
```



### 2D with Numpy: Matrix(행렬)

* 행렬은(Matrix)은 2개 이상의 벡터 값을 통합해 구성된 값
* 벡터 값 간의 연산 속도를 빠르게 진행할 수 있는 선형 대수의 기본 단위

파이토치로 2차원 텐서인 행렬을 만들어봅시다.
```py
# 행렬 값 정의
matrix1 = torch.tensor([[1., 2.],
                        [3., 4.]])
print(matrix1)

matrix2 = torch.tensor([[5., 6.],
                        [7., 8.]])
print(matrix2)
```
```
tensor([[1., 2.],
        [3., 4.]])
tensor([[5., 6.],
        [7., 8.]])
```

마찬가지로 현재 텐서의 차원과 크기를 확인할 수 있습니다.
```py
print(matrix1.dim())  # rank. 즉, 차원
print(matrix1.shape)  # shape
print(matrix1.size()) # shape
```
```
2
torch.Size([2, 2])
torch.Size([2, 2])
```

현재 2차원 텐서입니다.

다음은 행렬 값 간의 사칙연산을 해보겠습니다.
```py
# 행렬 값 간의 사칙연산: +, -, *, /
sum_matrix = matrix1 + matrix2
print(sum_matrix)

sub_matrix = matrix1 - matrix2
print(sub_matrix)

mul_matrix = matrix1 * matrix2
print(mul_matrix)

div_matrix = matrix1 / matrix2
print(div_matrix)
```
```
tensor([[ 6.,  8.],
        [10., 12.]])
tensor([[-4., -4.],
        [-4., -4.]])
tensor([[ 5., 12.],
        [21., 32.]])
tensor([[0.2000, 0.3333],
        [0.4286, 0.5000]])
```
```py
# 행렬 값 간의 사칙연산: torch 모듈에 내장된 메서드 이용해 계산
print(torch.add(matrix1, matrix2))
print(torch.sub(matrix1, matrix2))
print(torch.mul(matrix1, matrix2))
print(torch.div(matrix1, matrix2))
print(torch.matmul(matrix1, matrix2))# 행렬 곱 연산
```
```
tensor([[ 6.,  8.],
        [10., 12.]])
tensor([[-4., -4.],
        [-4., -4.]])
tensor([[ 5., 12.],
        [21., 32.]])
tensor([[0.2000, 0.3333],
        [0.4286, 0.5000]])
tensor([[19., 22.],
        [43., 50.]])
```



### 3D with Numpy: Tensor(텐서)

* 행렬을 2차원 배열이라 표현할 수 있다면, 텐서는 2차원 이상의 배열이라 표현할 수 있다.
* 텐서 내 행렬 단위의 인덱스 간, 행렬 내 인덱스 간 원소끼리 계산되며 행렬 곱은 텐서 내 같은 행렬 단위의 인덱스 간에 계산된다.

파이토치로 3차원인 텐서를 만들어봅시다.
```py
# 텐서 값 정의
tensor1 = torch.tensor([ [ [1., 2.],
                           [3., 4.] ],
                        
                         [ [5., 6.],
                           [7., 8.] ] ])
print(tensor1)

tensor2 = torch.tensor([ [ [9., 10.],
                           [11., 12.] ],
                        
                         [ [13., 14.],
                           [15., 16.] ] ])
print(tensor2)
```
```
tensor([[[1., 2.],
         [3., 4.]],

        [[5., 6.],
         [7., 8.]]])
tensor([[[ 9., 10.],
         [11., 12.]],

        [[13., 14.],
         [15., 16.]]])
```

마찬가지로 현재 텐서의 차원과 크기를 확인할 수 있습니다.
```py
print(tensor1.dim())  # rank. 즉, 차원
print(tensor1.shape)  # shape
print(tensor1.size()) # shape
```
```
3
torch.Size([2, 2, 2])
torch.Size([2, 2, 2])
```

현재 3차원 텐서입니다.

다음은 텐서 값 간의 사칙연산을 해보겠습니다.
```py
# 텐서 값 간의 사칙연산: +, -, *, /
sum_tensor = tensor1 + tensor2
print(sum_tensor)

sub_tensor = tensor1 - tensor2
print(sub_tensor)

mul_tensor = tensor1 * tensor2
print(mul_tensor)

div_tensor = tensor1 / tensor2
print(div_tensor)
```
```
tensor([[[10., 12.],
         [14., 16.]],

        [[18., 20.],
         [22., 24.]]])
tensor([[[-8., -8.],
         [-8., -8.]],

        [[-8., -8.],
         [-8., -8.]]])
tensor([[[  9.,  20.],
         [ 33.,  48.]],

        [[ 65.,  84.],
         [105., 128.]]])
tensor([[[0.1111, 0.2000],
         [0.2727, 0.3333]],

        [[0.3846, 0.4286],
         [0.4667, 0.5000]]])
```
```py
# 텐서 값 간의 사칙연산: torch 모듈에 내장된 메서드 이용해 계산
print(torch.add(tensor1, tensor2))
print(torch.sub(tensor1, tensor2))
print(torch.mul(tensor1, tensor2))
print(torch.div(tensor1, tensor2))
print(torch.matmul(tensor1, tensor2))# 텐서 간 텐서곱
```
```
tensor([[[10., 12.],
         [14., 16.]],

        [[18., 20.],
         [22., 24.]]])
tensor([[[-8., -8.],
         [-8., -8.]],

        [[-8., -8.],
         [-8., -8.]]])
tensor([[[  9.,  20.],
         [ 33.,  48.]],

        [[ 65.,  84.],
         [105., 128.]]])
tensor([[[0.1111, 0.2000],
         [0.2727, 0.3333]],

        [[0.3846, 0.4286],
         [0.4667, 0.5000]]])
tensor([[[ 31.,  34.],
         [ 71.,  78.]],

        [[155., 166.],
         [211., 226.]]])
```


