---
title: pytorch와 텐서 조작하기(Tensor Manipulation)
category: Pytorch
tag: pytorch
date: 2022-07-25
---     





<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/180914328-59cb3c89-8d9a-4bbf-821c-e7c2202d9817.png">
</p>

# 파이토치(PyTorch)

* 페이스북이 초기 루아(Lua) 언어로 개발된 토치(Torch)를 파이썬 버전으로 개발하여 2017년도에 공개
* 초기에 토치(Torch)는 넘파이(NumPy) 라이브러리처럼 과학 연산을 위한 라이브러리로 공개
* 이후 GPU를 이용한 텐서 조작 및 동적 신경망 구축이 가능하도록 딥러닝 프레임워크로 발전시킴
* 파이썬답게 만들어졌고, 유연하면서도 가속화된 계산 속도를 제공



## 파이토치 모듈 구조

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/180914532-fcf64536-a88a-4890-a274-afd224a65565.png">
</p>



## 파이토치의 구성요소

- `torch`: 메인 네임스페이스, 텐서 등의 다양한 수학 함수가 포함
- `torch.autograd`: 자동 미분 기능을 제공하는 라이브러리
- `torch.nn`: 신경망 구축을 위한 데이터 구조나 레이어 등의 라이브러리
- `torch.multiprocessing`: 병럴처리 기능을 제공하는 라이브러리
- `torch.optim`: SGD(Stochastic Gradient Descent)를 중심으로 한 파라미터 최적화 알고리즘 제공
- `torch.utils`: 데이터 조작 등 유틸리티 기능 제공
- `torch.onnx`: ONNX(Open Neural Network Exchange), 서로 다른 프레임워크 간의 모델을 공유할 때 사용

<br>





# 텐서 조작하기(Tensor Manipulation)

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/77891754/180914663-277155ce-b839-4a71-96fc-5ab7254314c0.png">
</p>

* 데이터 표현을 위한 기본 구조로 텐서(tensor)를 사용
* 텐서는 데이터를 담기위한 컨테이너(container)로서 일반적으로 수치형 데이터를 저장
* 넘파이(NumPy)의 ndarray와 유사
* GPU를 사용한 연산 가속 가능



## 1. 넘파이로 텐서 만들기(벡터와 행렬 만들기)

딥러닝을 하게 되면 다루게 되는 가장 기본적인 단위는 벡터, 행렬, 텐서입니다. 차원이 없는 값을 스칼라, 1차원으로 구성된 값을 **벡터**라고 합니다. 2차원으로 구성된 값을 **행렬(Matrix)**라고 합니다. 그리고 3차원이 되면 **텐서(Tensor)**라고 부릅니다. PyTorch로 텐서를 만들어보기 전에 우선 Numpy로 텐서를 만들어보겠습니다.

```py
import numpy as np
```

Numpy로 텐서를 만드는 방법은 간단한데 [숫자, 숫자, 숫자]와 같은 형식으로 만들고 이를 `np.array()`로 감싸주면 됩니다.



## 1.1 1D with Numpy

Numpy로 1차원 텐서인 벡터를 만들어보겠습니다.

```py
# 파이썬으로 설명하면 list를 생성해서 np.array로 1차원 array로 변환함
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

[output]
[0. 1. 2. 3. 4. 5. 6.]
```

이제 1차원 벡터의 차원과 크기를 출력해보겠습니다.

```py
print('Rank of t: ', t.ndim) #1차원 벡터
print('Shape of t: ', t.shape)

[output]
Rank of t:  1
Shape of t:  (7,)
```

* `.ndim`은 몇 차원인지를 출력합니다.
    * 1차원은 벡터, 2차원은 행렬, 3차원은 3차원 텐서였습니다. 현재는 벡터이므로 1차원이 출력됩니다.
* `.shape`는 크기를 출력합니다.
    * `(7, )`는 `(1, 7)`을 의미합니다. 다시 말해 `(1 × 7)`의 크기를 가지는 벡터입니다.



## 1.2 2D with Numpy

Numpy로 2차원 행렬을 만들어보겠습니다.

```py
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)

[output]
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]
 [10. 11. 12.]]
```

```py
print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)

[output]
Rank  of t:  2
Shape of t:  (4, 3)
```

현재는 행렬이므로 2차원이 출력되며, shape은 `(4, 3)` 입니다. 다른 표현으로는 `(4 × 3)` 입니다. 이는 행렬이 4행 3열임을 의미합니다.

<br>





## 2. 파이토치 텐서(PyTorch Tensor)

파이토치는 Numpy와 매우 유사합니다. 우선 torch를 임포트합니다.

```py
import torch
```





### 2.1 텐서 초기화

우선 텐서를 초기화 하는 여러가지 방법을 살펴보겠습니다.

#### 초기화 되지 않은 텐서
```py
x = torch.empty(4, 2)
print(x)
```
```
tensor([[7.2747e-35, 0.0000e+00],
        [3.3631e-44, 0.0000e+00],
        [       nan, 0.0000e+00],
        [1.1578e+27, 1.1362e+30]])
```

#### 무작위로 초기화된 텐서
```py
x = torch.rand(4, 2)
print(x)
```
```
tensor([[0.7464, 0.7540],
        [0.5432, 0.0055],
        [0.4031, 0.0854],
        [0.6742, 0.8194]])
```

#### 데이터 타입(dtype)이 long이고, 0으로 채워진 텐서
```py
x = torch.zeros(4, 2, dtype=torch.long)
print(x)
```
```
tensor([[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]])
```

#### 사용자가 입력한 값으로 텐서 초기화
```py
x = torch.tensor([3, 2.3])
print(x)
```
```
tensor([3.0000, 2.3000])
```

#### 2 x 4 크기, double 타입, 1로 채워진 텐서
```py
x = x.new_ones(2, 4, dtype=torch.double)
print(x)
```
```
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype=torch.float64)
```

#### x와 같은 크기, float 타입, 무작위로 채워진 텐서
```py
x = torch.randn_like(x, dtype=torch.float)
print(x)
```
```
tensor([[ 0.4575, -0.9619,  1.2463, -0.5515],
        [-1.5581, -0.6273,  0.0430,  0.5415]])
```

#### 텐서의 크기 계산
```py
print(x.size())
```
```
torch.Size([2, 4])
```





### 2.2 데이터 타입(Data Type)

다음은 텐서의 데이터 타입 방법입니다.

| Data type | dtype | CPU tensor | GPU tensor |
| ------ | ------ | ------ | ------ |
| 32-bit floating point | `torch.float32` or `torch.float` |`torch.FloatTensor` | `torch.cuda.FloatTensor` |
| 64-bit floating point | `torch.float64` or `torch.double` |`torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
| 16-bit floating point | `torch.float16` or `torch.half` |`torch.HalfTensor` | `torch.cuda.HalfTensor` |
| 8-bit integer(unsinged) | `torch.uint8` |`torch.ByteTensor` | `torch.cuda.ByteTensor` |
| 8-bit integer(singed) | `torch.int8` |`torch.CharTensor` | `torch.cuda.CharTensor` |
| 16-bit integer(signed) | `torch.int16` or `torch.short` |`torch.ShortTensor` | `torch.cuda.ShortTensor` |
| 32-bit integer(signed) | `torch.int32` or `torch.int` |`torch.IntTensor` | `torch.cuda.IntTensor` |
| 64-bit integer(signed) | `torch.int64` or `torch.long` |`torch.LongTensor` | `torch.cuda.LongTensor` |

```py
ft = torch.FloatTensor([1, 2, 3])
print(ft)
print(ft.dtype)
```
```
tensor([1., 2., 3.])
torch.float32
```

```py
print(ft.short())
print(ft.int())
print(ft.long())
```
```
tensor([1, 2, 3], dtype=torch.int16)
tensor([1, 2, 3], dtype=torch.int32)
tensor([1, 2, 3])
```

```py
it = torch.IntTensor([1, 2, 3])
print(it)
print(it.dtype)
```
```
tensor([1, 2, 3], dtype=torch.int32)
torch.int32
```

```py
print(it.float())
print(it.double())
print(it.half())
```
```
tensor([1., 2., 3.])
tensor([1., 2., 3.], dtype=torch.float64)
tensor([1., 2., 3.], dtype=torch.float16)
```





### 2.3 CUDA Tensors

- `.to` 메소드를 사용하여 텐서를 어떠한 장치(cpu, gpu)로도 옮길 수 있음

```py
x = torch.randn(1)
print(x)
print(x.item())
print(x.dtype)
```
```
tensor([-0.9480])
-0.9479643106460571
torch.float32
```

다음은 cuda와 cpu간 변화를 볼 수 있습니다.
```py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.device('cuda')
#torch.device('cpu')
print(device)
```
```
cuda
```

```py
y = torch.ones_like(x, device=device)
print(y)
```
```
tensor([1.], device='cuda:0')
```

```py
x = x.to(device)
print(x)
```
```
tensor([-0.9480], device='cuda:0')
```

```py
z = x + y
print(z)
```
```
tensor([0.0520], device='cuda:0')
```

```py
print(z.to('cpu', torch.double))
```
```
tensor([0.0520], dtype=torch.float64)
```





### 2.4 다차원 텐서 표현



#### 0D Tensor: Scalar(스칼라)

* 하나의 숫자를 담고 있는 텐서(tensor)
* 축과 형상이 없음

```py
# 스칼라 값 정의
scalar1 = torch.tensor(1)
print(scalar1)

scalar2 = torch.tensor(3)
print(scalar2)
```
```
tensor(1)
tensor(3)
```

`dim()`을 사용하면 현재 텐서의 차원을 보여줍니다. `shape`나 `size()`를 사용하면 크기를 확인할 수 있습니다.
```py
print(scalar1.dim())  # rank. 즉, 차원
print(scalar1.shape)  # shape
print(scalar1.size()) # shape
```
```
0
torch.Size([])
torch.Size([])
```

스칼라라는 의미로 숫자 한개만을 선언을 했지만 내부적으로 크기가 1인 벡터로 인식합니다. 즉 현재 1차원 텐서이며, 원소는 1개 입니다. 다음은 스칼라 값 간의 사칙연산을 해보겠습니다.
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
tensor(4)
tensor(-2)
tensor(3)
tensor(0.3333)
```

```py
# 스칼라 값 간의 사칙연산: torch 모듈에 내장된 메서드 이용해 계산
print(torch.add(scalar1, scalar2))
print(torch.sub(scalar1, scalar2))
print(torch.mul(scalar1, scalar2))
print(torch.div(scalar1, scalar2))
```
```
tensor(4)
tensor(-2)
tensor(3)
tensor(0.3333)
```



### 1D Tensor: Vector(벡터)

* 벡터는 하나의 값을 표현할 때 2개 이상의 수치로 표현한 것.
* 스칼라의 형태와 동일한 속성을 갖고 있지만, 여러 수치 값을 이용해 표현하는 방식
* 값들을 저장한 리스트와 유사한 텐서
* 하나의 축이 존재

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

현재 1차원 텐서이며, 원소는 3개입니다. 다음은 벡터 값 간의 사칙연산을 해보겠습니다.
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



### 2D Tensor: Matrix(행렬)

* 행렬은(Matrix)은 2개 이상의 벡터 값을 통합해 구성된 값
* 벡터 값 간의 연산 속도를 빠르게 진행할 수 있는 선형 대수의 기본 단위
* 일반적인 수치, 통계 데이터셋이 해당
* 주로 샘플(samples)과 특성(features)을 가진 구조로 사용

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

현재 2차원 텐서입니다. 다음은 행렬 값 간의 사칙연산을 해보겠습니다.
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



#### 3D Tensor(텐서)

* 행렬을 2차원 배열이라 표현할 수 있다면, 텐서는 2차원 이상의 배열이라 표현할 수 있다.
* 텐서 내 행렬 단위의 인덱스 간, 행렬 내 인덱스 간 원소끼리 계산되며 행렬 곱은 텐서 내 같은 행렬 단위의 인덱스 간에 계산된다.
* 큐브(cube)와 같은 모양으로 세개의 축이 존재
* 데이터가 연속된 시퀀스 데이터나 시간 축이 포함된 시계열 데이터에 해당
* 주식 가격 데이터셋, 시간에 따른 질병 발병 데이터 등이 존재
* 주로 샘플(samples), 타임스텝(timesteps), 특성(features)을 가진 구조로 사용

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

현재 3차원 텐서입니다. 다음은 텐서 값 간의 사칙연산을 해보겠습니다.
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



#### 4D Tensor

* 4개의 축
* 컬러 이미지 데이터가 대표적인 사례 (흑백 이미지 데이터는 3D Tensor로 가능)
* 주로 샘플(samples), 높이(height), 너비(width), 컬러 채널(channel)을 가진 구조로 사용



#### 5D Tensor

* 5개의 축
* 비디오 데이터가 대표적인 사례
* 주로 샘플(samples), 프레임(frames), 높이(height), 너비(width), 컬러 채널(channel)을 가진 구조로 사용




