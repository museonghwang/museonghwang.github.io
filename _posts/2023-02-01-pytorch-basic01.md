---
layout: post
title: pytorch tensor의 이해
category: Pytorch
tag: Pytorch
---




**<span style="color:red">텐서(tensor)</span>** 는 **<span style="color:red">딥러닝에서 가장 기본이 되는 단위 중 하나</span>** 입니다. **스칼라(scalar)**, **벡터(vector)**, **행렬(matrix)**, 그리고 **텐서** 를 통해 **<u>딥러닝 연산을 수행</u>** 할 수 있습니다. 다음 그림은 스칼라, 벡터, 행렬, 텐서의 관계를 나타냅니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/d1253a71-12cd-434f-9e18-77607262b3a0">
</p>

<br>



각 값을 메모리에 저장할 때 **<span style="background-color: #fff5b1">스칼라(scalar)</span>** 는 **<u>하나의 변수</u>** 로 나타낼 수 있고, **<span style="background-color: #fff5b1">벡터(vector)</span>** 는 **<u>1차원의 배열</u>** 로 나타낼수 있으며, **<span style="background-color: #fff5b1">행렬(matrix)</span>** 은 **<u>2차원의 배열</u>** 로 나타내며 **<span style="background-color: #fff5b1">텐서(tensor)</span>** 부터는 **<u>3차원 이상의 배열</u>** 로 나타냅니다. 일반적으로 3차원부터는 모두 텐서라고 묶어서 부릅니다.

<br>




# 행렬의 표현

우리가 다룰 대부분의 값은 보통 **float** 타입이나 **double** 타입으로 표현되는 **실수(real number)** 입니다. 실수들로 채워진 **<span style="color:red">행렬</span>** 은 다음 그림과 같이 표현할 수 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/a1eb24dc-e7a3-46fb-8410-08a0da6a1f8d">
</p>

<br>


행렬 $x$ 는 **<u>$k$ 개의 행(row)</u>** 과 **<u>$n$ 개의 열(column)</u>** 로 이루어져 있으며 값들은 모두 실수로 이루어져 있습니다. 이것을 수식으로 표현하면 다음과 같습니다.

$$
x∈R^{k×n} -> |x|=(k,n)
$$


<br>


이는 텐서에 대한 **size()** 함수를 호출한 것과 같습니다. 앞의 그림에서 **<span style="color:red">첫 번째 차원(dimension) $k$ 가 세로축의 크기를 나타내고</span>** 있고, **<span style="color:red">두 번째 차원 $n$ 이 가로축의 크기를 나타내고</span>** 있습니다.

<br>




# 텐서의 표현

이번에는 행렬에 이어 **<span style="color:red">텐서의 표현</span>** 에 대해서 살펴보도록 하겠습니다. 다음 그림은 실수로 이루어진 **$k×n×m$ 차원의 텐서** 를 나타냅니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/ded0d377-8ced-4b20-8144-53ad442e9e57">
</p>

<br>



중요한 점은 **<span style="color:red">첫 번째 차원 $k$ 가 세로축의 크기를 나타내고</span>**, **<span style="color:red">$n$ 이 가로축의 차원을 나타내고</span>**, **<span style="color:red">$m$ 이 마지막 남은 축의 차원을 나타냅니다.</span>** 

<br>




# 다양한 행렬/텐서의 모양들

딥러닝을 통해 다양한 분야의 많은 문제를 풀어나갈때 데이터의 도메인에 따라 문제들을 나눠 볼 수 있습니다. 각 도메인에 따라서 자주 다루게 될 텐서의 형태 또한 상이합니다. 이번에는 각 도메인 별로 자주 만날 행렬/텐서의 형태에 대해서 알아보겠습니다.

<br>



## 데이터 사이언스: 테이블 형태의 데이터셋

데이터 사이언스 또는 데이터 분석을 수행할때에는 주로 **<span style="background-color: #fff5b1">테이블 형태(tabular)의 데이터셋</span>** 을 다루게 되는데 쉽게 말해, 여러 개의 **열(column)** 이 존재하고 각 샘플들은 각 열에 대해서 값을 가지며 하나의 **행(row)** 을 이루게 됩니다. 테이블 형태의 데이터를 텐서로 나타내면 다음과 같습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/8ebdfcd3-ae71-478d-8539-c788b81d2bf4">
</p>

<br>

- $N$
    - **<span style="background-color: #fff5b1">행의 개수(샘플의 수)</span>** 를 나타내고, **<span style="color:red">세로축의 크기</span>** 를 나타냅니다.
- $n$
    - **<span style="background-color: #fff5b1">열의 개수</span>** 를 나타내고 **<span style="color:red">가로축의 크기</span>** 를 나타냅니다.
    - 열은 피처(feature)라고 부르며 각 샘플의 고유한 속성을 설명하는 값을 담고 있습니다.
    - 만약 피처의 값이 비슷한 샘플끼리는 비슷한 속성을 가진다고 볼 수 있습니다.


<br>


위 그림에서 빨간색 점선으로 둘러싸인 부분은 하나의 샘플을 나타냅니다. 전체 데이터가 $N×n$ 행렬이므로, 하나의 샘플은 $n$ 개의 요소를 갖는 $n$ 차원의 벡터가 됩니다. $n$ 차원의 벡터가 $N$ 개 모이므로, $N×n$ 차원의 행렬이 되겠죠.


딥러닝은 병렬(parallel) 연산을 수행합니다. 만약에 $N$ 개의 샘플을 신경망에 통과시킨다면 $N$ 번의 연산을 각각 수행하는 것이 아니라 메모리의 크기가 허용되는 범위 안에서 덩어리로 통과시킵니다. 예를 들어, $k$ 개의 샘플 벡터를 통과시킨다면 이를 위해 $k$ 의 샘플들은 $k×n$ 의 행렬이 되어 신경망을 통과하게 될 것입니다. 다음 그림은 이런 병렬 연산을 위한 행렬을 빨간 점선으로 나타낸 것입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/61ba8055-1404-41d4-b5f0-0d02de2a4198">
</p>

<br>



빨간색 점선 부분의 $k×n$ 행렬은 전체 데이터셋 $N×n$ 행렬에서 슬라이싱(slicing)을 통해 얻을 수 있습니다.


<br>




## 자연어 처리: 문장 데이터셋

자연어 처리가 주로 다루는 데이터의 대상은 **<span style="background-color: #fff5b1">문장</span>** 입니다. 
- **<span style="color:red">문장</span>** : 단어 또는 토큰(token)들이 모여서 이루어진 **시퀀셜 데이터**
- **<span style="color:red">시퀀셜 데이터</span>** : 내부 토큰들의 출현과 순서 관계에 의해서 속성이 정의

<br>


**<u>단어(토큰)는 각각이 의미를 지니기 때문에 의미를 나타내기 위한 벡터로 표현</u>** 되는데, 이를 **<span style="color:red">단어 임베딩 벡터(word embedding vector)</span>** 라고 부릅니다. 그리고 단어들이 모여서 문장이 되기 때문에 **<u>단어 임베딩 벡터가 모여</u>** **<span style="color:red">문장을 표현하는 행렬</span>** 이 됩니다. 또한 **<u>문장 행렬은 병렬 처리를 위해 덩어리로 묶어</u>** 야 하니 **<span style="color:red">3차원의 텐서</span>** 가 됩니다. 이것을 그림으로 나타내면 다음과 같습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/03c88858-7480-4266-9711-9ed494588ee6">
</p>

<br>




**<span style="color:red">$N$ 개의 문장을 갖는 텐서 $x$</span>** 는 다음과 같이 나타낼 수 있습니다.

$$
|x|=(N,ℓ,d)
$$

<br>



**<span style="color:red">각 문장은 최대 $ℓ$ 개의 단어를 갖고 있을 것이고 이것은 문장의 길이</span>** 를 나타냅니다. 그리고 **<span style="color:red">각 단어는 $d$ 차원의 벡터로 표현</span>** 될 것입니다. 이와 같이 **<span style="background-color: #fff5b1">자연어 처리를 위한 데이터는 3차원의 텐서</span>** 로 나타낼 수 있습니다.


**<span style="color:red">이 데이터의 가장 큰 특징은 문장의 길이에 따라 텐서의 크기가 변할 수 있다는 것</span>** 입니다. 데이터셋 또는 코퍼스(corpus) **<u>내부의 문장의 길이가 전부 제각각일 것이므로 어떻게 문장을 선택하여 덩어리로 구성하느냐에 따라서 $ℓ$ 의 크기가 바뀌게 됩니다.</u>** 즉, 프로그램이 실행되는 와중에 덩어리 텐서의 크기가 가변적이게 되므로 일반적인 신경망 계층(e.g.선형 계층)을 활용하여 처리하기 어렵고, 주로 순환신경망(recurrent neural networks)을 사용하거나 트랜스포머(Transformer)를 사용합니다.






## 컴퓨터비전: 이미지 데이터셋

자연어 처리가 주로 다루는 데이터의 대상은
컴퓨터비전(computer vision) 분야는 주로 **<span style="background-color: #fff5b1">이미지 데이터</span>** 를 다룹니다. 다음 그림은 흑백(gray scale) 이미지의 텐서를 그림으로 나타낸 것입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/d637713a-2515-476e-8b71-28c2b8387a48">
</p>

<br>



**<span style="color:red">흑백 이미지의 각 픽셀</span>** 은 **<u>0부터 255까지의 8비트(bit)(1바이트(1byte)) 값으로 표현</u>** 됩니다. **<span style="background-color: #fff5b1">한 장의 이미지</span>** 는 **<u>세로축×가로축 만큼의 픽셀들로 이루어져</u>** 있으며 이것은 **<span style="background-color: #fff5b1">행렬로 표현</span>** 가능합니다. 그리고 여러장의 이미지 행렬이 합쳐지면 3차원의 텐서가 됩니다.



다음 그림은 컬러 이미지의 텐서를 그림으로 나타낸 것입니다. **<span style="color:red">컬러 이미지의 각 픽셀</span>** 은 **<u>RGB 값으로 표현</u>** 됩니다. RGB 값은 빨강(0~255), 초록(0~255), 파랑(0~255) 값이 모여 **<span style="background-color: #fff5b1">8×3 비트로 표현</span>**됩니다. 여기에서 각 색깔을 나타내는 값은 채널이라고 부릅니다. 즉, **<span style="background-color: #fff5b1">RGB에는 3개의 채널이 존재</span>** 합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/08388581-90f3-4020-afba-52fc4c915588">
</p>

<br>




따라서 **<u>한 장의 흑백 이미지를 표현하기 위해서는 행렬이 필요</u>** 했던 반면, **<u>컬러 이미지 한 장을 표현하기 위해서는 3차원의 텐서가 필요</u>** 합니다. 정확하게는 빨강 값을 나타내는 행렬, 초록값을 나타내는 행렬, 파랑 값을 나타내는 행렬이 합쳐져서 텐서가 됩니다. **<u>결과적으로 이미지 덩어리를 표현하기 위해서 4차원의 텐서가 필요</u>** 합니다.


테이블 형태의 데이터는 각 열(피처)의 값이 굉장히 중요하지만, 이미지의 경우에는 한 픽셀씩 그림이 평행 이동하더라도 그림의 속성이 바뀌지는 않습니다. 따라서 이러한 이미지의 속성을 반영하기 위해 일반적인 계층(e.g. 선형 계층)을 사용하기보다 합성곱신경망(convolution neural network)을 주로 사용합니다.

<br>





# 파이토치 실습

## 텐서 생성

먼저 파이토치를 불러옵니다.
```py
import torch
```


<br>



파이토치 텐서는 다양한 타입의 텐서를 지원합니다. 다음 코드는 실수형 **Float 텐서** 를 선언하는 모습입니다.
```py
ft = torch.FloatTensor([[1, 2],
                        [3, 4]])
ft
```
```
[output]
tensor([[1., 2.],
        [3., 4.]])
```


<br>

출력 결과를 보면 실수형 값들로 요소가 채워진 것을 확인할 수 있습니다. 해당 텐서를 실제 행렬로 나타내면 다음과 같습니다.

$$ ft=\left[
\begin{array}{cc}
   1.0 & 2.0 \\
   3.0 & 4.0 \\
\end{array}
\right]
$$


<br>

이처럼 다차원 배열 값(또는 배열 값이 담겨있는 변수)을 넣어 원하는 요소 값을 갖는 텐서를 직접 생성할 수 있습니다. 같은 방법으로 **Long 타입** 과 **Byte 타입** 을 선언할 수 있습니다.
```py
lt = torch.LongTensor([[1, 2],
                       [3, 4]])
lt
```
```
[output]
tensor([[1, 2],
        [3, 4]])
```


<br>

```py
bt = torch.ByteTensor([[1, 0],
                       [0, 1]])
bt
```
```
[output]
tensor([[1, 0],
        [0, 1]], dtype=torch.uint8)
```


<br>


만약 임의의 값으로 채워진 원하는 크기의 텐서를 만들고자 한다면 다음과 같이 간단하게 만들 수 있습니다.
```py
x = torch.FloatTensor(3, 2)
x
```
```
[output]
tensor([[-1.2503e+16,  4.5586e-41],
        [-1.2503e+16,  4.5586e-41],
        [ 3.1360e+27,  7.0800e+31]])
```


<br>




## 넘파이 호환


파이토치는 **넘파이와 높은 호환성** 을 자랑하며, 실제로 대부분의 함수들은 넘파이와 비슷한 사용법을 가지고 있습니다. 다음과 같이 넘파이를 불러온 후 넘파이의 배열을 선언하고 출력하면 **numpy.ndarray** 가 할당되어 있는 것을 확인할 수 있습니다.
```py
import numpy as np

# Define numpy array.
x = np.array([[1, 2],
              [3, 4]])
print(x, type(x))
```
```
[output]
[[1 2]
 [3 4]] <class 'numpy.ndarray'>
```


<br>


이렇게 선언한 **ndarray** 를 파이토치 텐서로 변환할 수 있습니다.
```py
x = torch.from_numpy(x)
print(x, type(x))
```
```
[output]
tensor([[1, 2],
        [3, 4]]) <class 'torch.Tensor'>
```


<br>


출력 결과를 보면 파이토치 텐서로 변환된 것을 볼 수 있습니다. 반대로 파이토치 텐서를 넘파이 ndarray로 변환할 수도 있습니다.
```py
x = x.numpy()
print(x, type(x))
```
```
[output]
[[1 2]
 [3 4]] <class 'numpy.ndarray'>
```


<br>



## 텐서 타입 변환

파이토치 **텐서의 타입 변환** 도 굉장히 간단합니다. 단순히 원하는 타입을 함수로 호출하면 됩니다. 다음 코드는 **Float 타입 텐서** 를 **Long 타입 텐서** 로 변환하는 코드입니다.
```py
ft.long()
```
```
[output]
tensor([[1, 2],
        [3, 4]])
```


<br>

```py
lt.float()
```
```
[output]
tensor([[1., 2.],
        [3., 4.]])
```


<br>




## 텐서 크기 구하기

딥러닝 계산을 수행하다 보면 텐서의 크기를 구해야 할 때가 많습니다. 텐서 크기를 구하는 방법을 알아보겠습니다. 다음과 같이 3×2×2 텐서 x를 선언합니다.
```py
x = torch.FloatTensor([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]]])
```


<br>


**텐서 크기** 를 구하려면 **size()** 함수를 호출하거나 **shape 속성** 에 접근합니다. **size()** 함수의 결과값이 **shape 속성** 에 담겨있다고 보면 됩니다.
```py
print(x.size())
print(x.shape)
```
```
[output]
torch.Size([3, 2, 2])
torch.Size([3, 2, 2])
```


<br>

이 크기 정보는 배열(list)에 담겨있다고 생각하면 됩니다. 따라서 **특정 차원의 크기를 알기 위해** 서는 **shape** 속성의 해당 차원 인덱스에 접근하거나 **size()** 함수의 인자에 원하는 차원의 인덱스를 넣어주면 됩니다.
```py
print(x.size(1))
print(x.shape[1])
```
```
[output]
2
2
```


<br>

**텐서 차원의 개수** 를 알기 위해서는 **dim()** 함수를 활용합니다. 이것은 **shape** 속성의 배열 크기와 같습니다.
```py
print(x.dim())
print(len(x.size()))
```
```
[output]
3
3
```




