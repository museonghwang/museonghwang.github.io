---
layout: post
title: pytorch tensor의 기본 연산
category: Pytorch
tag: Pytorch
---




파이토치 텐서들을 활용한 기본 연산에 대해서 살펴보겠습니다.
```py
import torch
```

<br>



# 요소별 산술연산

다음과 같이 두 개의 텐서(행렬) a와 b가 있다고 가정하겠습니다.

$$ a=\left[
\begin{array}{cc}
   1 & 2 \\
   3 & 4 \\
\end{array}
\right], 
b=\left[
\begin{array}{cc}
   2 & 2 \\
   3 & 3 \\
\end{array}
\right]
$$

<br>


```py
a = torch.FloatTensor([[1, 2],
                       [3, 4]])
b = torch.FloatTensor([[2, 2],
                       [3, 3]])
```


<br>

이제 두 행렬 사이의 덧셈을 수행할 수 있습니다.

$$ a+b=\left[
\begin{array}{cc}
   1 & 2 \\
   3 & 4 \\
\end{array}
\right] + 
\left[
\begin{array}{cc}
   2 & 2 \\
   3 & 3 \\
\end{array}
\right] = 
\left[
\begin{array}{cc}
   1 + 2 & 2 + 2 \\
   3 + 3 & 4 + 3 \\
\end{array}
\right] = 
\left[
\begin{array}{cc}
   3 & 4 \\
   6 & 7 \\
\end{array}
\right]
$$

<br>


파이토치에서 구현하면 다음과 같습니다.
```py
a + b
```
```
[output]
tensor([[3., 4.],
        [6., 7.]])
```


<br>



마찬가지로 뺄셈, 곱셈, 나눗셈 연산을 파이토치 코드로 구현하면 다음과 같습니다.
```py
print(a - b) # 뺄셈
print(a * b) # 곱셈
print(a / b) # 나눗셈
```
```
[output]
tensor([[-1.,  0.],
        [ 0.,  1.]])
tensor([[ 2.,  4.],
        [ 9., 12.]])
tensor([[0.5000, 1.0000],
        [1.0000, 1.3333]])
```


<br>


제곱 연산도 비슷하게 취해볼 수 있습니다. 이 연산을 파이토치 코드로 구현하면 다음과 같습니다.
```py
a ** b
```
```
[output]
tensor([[ 1.,  4.],
        [27., 64.]])
```


<br>

논리 연산자도 마찬가지로 쉽게 구현할 수 있습니다. 아래 코드는 행렬의 각 위치의 요소가 같은 값일 경우 True, 다른 값일 경우 False를 갖도록 하는 연산입니다.
```py
a == b
```
```
[output]
tensor([[False,  True],
        [ True, False]])
```


<br>


마찬가지로 **!=** 연산자를 사용하게 되면 다른 값일 경우 True, 같은 값일 경우 False를 갖게 됩니다.
```py
a != b
```
```
[output]
tensor([[ True, False],
        [False,  True]])
```


<br>





# 인플레이스 연산

앞에서 수행한 연산들의 결과 텐서는 빈 메모리에 결과 텐서가 새롭게 할당됩니다. 하지만 **인플레이스(in-place)** 연산은 같은 산술 연산을 수행하지만 기존 텐서에 결과가 저장된다는 차이점이 있습니다. 다음 코드를 확인하겠습니다.
```py
print(a)
print(a.mul(b))
print(a)
```
```
[output]
tensor([[1., 2.],
        [3., 4.]])
tensor([[ 2.,  4.],
        [ 9., 12.]])
tensor([[1., 2.],
        [3., 4.]])
```


<br>


곱셈 연산 함수인 **a.mul(b)** 의 연산 결과 텐서는 새로운 메모리에 할당됩니다. 따라서 다시 텐서 a를 출력하면 a의 값은 그대로인 것을 볼 수 있습니다.

인플레이스 연산들은 밑줄(underscore)이 함수명 뒤에 붙어있는 것이 특징입니다. 따라서 곱셈 함수의 인플레이스 연산 함수는 **mul_()** 으로 대응됩니다.
```py
print(a.mul_(b))
print(a)
```
```
[output]
tensor([[ 2.,  4.],
        [ 9., 12.]])
tensor([[ 2.,  4.],
        [ 9., 12.]])
```


<br>


즉, 메모리의 새로운 공간에 계산 결과가 저장되는 것이 아니라 기존 a의 공간에 계산결과가 저장되는 것입니다. 얼핏 생각하면 새로운 메모리의 공간을 할당하는 작업이 생략되기 때문에 속도나 공간 사용 측면에서 훨씬 효율적일 것 같지만 파이토치 측은 가비지 컬렉터가 효율적으로 작동하기 때문에 굳이 인플레이스 연산을 사용할 필요는 없다고 합니다.

<br>




# 차원 축소 연산: 합과 평균

다음과 같은 텐서 x가 있다고 가정해보겠습니다.

$$ x=\left[
\begin{array}{cc}
   1 & 2 \\
   3 & 4 \\
\end{array}
\right]
$$

<br>

```py
x = torch.FloatTensor([[1, 2],
                       [3, 4]])
```

<br>


다음과 같이 **sum()** 함수 또는 **mean()** 함수를 통해 행렬 전체 요소의 합이나 평균을 구할 수 있습니다. 행렬 요소 전체의 합이나 평균은 텐서나 행렬이 아닌 스칼라(scalar) 값으로 저장되므로 차원이 축소된다고 볼 수 있습니다.
```py
print(x.sum())
print(x.mean())
```
```
[output]
tensor(10.)
tensor(2.5000)
```


<br>


여기에서 함수의 **dim** 인자에 원하는 연산의 차원을 넣어줄 수 있습니다. **dim** 인자의 값은 없어지는 차원이라고 생각하면 쉽습니다.
```py
print(x.sum(dim=0))
```
```
[output]
tensor([4., 6.])
```


<br>

**dim=0** 이면 첫 번째 차원을 이야기하는 것이므로 행렬의 세로축에 대해서 합(sum) 연산을 수행합니다. 수식으로 표현하면 다음과 같이 표현될 수 있습니다. 2차원인 행렬의 차원이 축소되어 벡터가 되었으므로 세로로 표현되는 것이 맞지만, 이해를 돕기 위해 전치 연산을 통해 가로 벡터로 표현했습니다.

$$ sum(x, \ dim=0)=\left[
\begin{array}{cc}
   1 & 2 \\
   {+} & {+} \\
   3 & 4 \\
\end{array}
\right] = [4 \ 6]^T
$$

<br>


행렬의 세로 축인 첫 번째 차원에 대해서 축소 연산이 수행되는 것을 확인할 수 있습니다. dim 인자의 값으로 -1도 줄 수 있는데 -1을 차원의 값으로 넣어주게 되면 뒤에서 첫 번째 차원을 의미합니다. 여기에서는 2개의 차원만 존재하므로 dim=1을 넣어준 것과 동일할 것입니다.
```py
print(x.sum(dim=-1))
```
```
[output]
tensor([3., 7.])
```

<br>

$$ sum(x, \ dim=-1)=\left[
\begin{array}{cc}
   1 + 2 \\
   3 + 4 \\
\end{array}
\right] = \left[
\begin{array}{cc}
   3 \\
   7 \\
\end{array}
\right]
$$

<br>




# 브로드캐스트 연산

## 텐서 + 스칼라

가장 먼저 쉽게 생각해볼 수 있는 것은 행렬(또는 텐서)에 스칼라를 더하는 것입니다.
```py
x = torch.FloatTensor([[1, 2],
                       [3, 4]])
y = 1
```



<br>


텐서 x와 스칼라 y를 선언하였습니다. 다음의 코드는 x와 y를 더하여 z에 저장한 후, z의 값과 z의 크기를 출력하도록 하는 코드입니다.
```py
z = x + y
print(z)
print(z.size())
```
```
[output]
tensor([[2., 3.],
        [4., 5.]])
torch.Size([2, 2])
```


<br>

행렬 x의 각 요소에 모두 1이 더해진 것을 볼 수 있습니다.



<br>

## 텐서 + 벡터

```py
x = torch.FloatTensor([[1, 2],
                       [4, 8]])
y = torch.FloatTensor([3,
                       5])

print(x.size())
print(y.size())
```
```
[output]
torch.Size([2, 2])
torch.Size([2])
```


<br>



위의 코드를 실행하면 2×2 행렬 x와 2개의 요소를 갖는 벡터 y를 선언하고, 크기가 다른 두 텐서를 더해보려 합니다. 크기가 다른 두 텐서 사이의 연산을 위해 브로드캐스팅이 적용될 경우 다음과 같이 됩니다. 차원에 맞춰 줄을 세우고 빈칸의 값이 1이라고 가정할 때 다른 한쪽에 똑같이 맞춥니다.
```
[2, 2]     [2, 2]     [2, 2]
[   2] --> [1, 2] --> [2, 2]
```

<br>

이렇게 같은 모양을 맞춘 이후에 덧셈 연산을 수행합니다. 수식으로 나타내면 다음과 같습니다.

$$ a+b=\left[
\begin{array}{cc}
   1 & 2 \\
   4 & 8 \\
\end{array}
\right] + 
\left[
\begin{array}{cc}
   3 \\ 5
\end{array}
\right] = 
\left[
\begin{array}{cc}
   1 & 2 \\
   4 & 8 \\
\end{array}
\right] +
\left[
\begin{array}{cc}
   3 & 5 \\
   3 & 5 \\
\end{array}
\right] = 
\left[
\begin{array}{cc}
   4 & 7 \\
   7 & 13 \\
\end{array}
\right]
$$

<br>


다음 코드를 실행하면 예측한 정답이 나오는 것을 볼 수 있습니다.
```py
z = x + y
print(z)
print(z.size())
```
```
[output]
tensor([[ 4.,  7.],
        [ 7., 13.]])
torch.Size([2, 2])
```


<br>



그러면 텐서들의 덧셈을 살펴보기 위해 텐서를 선언하고 크기를 출력합니다.
```py
x = torch.FloatTensor([[[1, 2]]])
y = torch.FloatTensor([3,
                       5])

print(x.size())
print(y.size())
```
```
[output]
torch.Size([1, 1, 2])
torch.Size([2])
```


<br>


실행 결과를 보면 텐서들의 크기를 확인할 수 있습니다. 그러고 나면 좀 전의 규칙을 똑같이 적용해볼 수 있습니다.
```
[1, 1, 2]     [1, 1, 2]
[      2] --> [1, 1, 2]
```

<br>


다음 코드를 수행하면 결과를 얻을 수 있습니다.
```py
z = x + y
print(z)
print(z.size())
```
```
[output]
tensor([[[4., 7.]]])
torch.Size([1, 1, 2])
```


<br>



## 텐서 + 텐서

이 브로드캐스팅 규칙은 차원의 크기가 1인 차원에 대해서도 비슷하게 적용됩니다. 다음과 같이 두 텐서를 선언하고 크기를 출력합니다.
```py
x = torch.FloatTensor([[1, 2]])
y = torch.FloatTensor([[3],
                       [5]])

print(x.size())
print(y.size())
```
```
[output]
torch.Size([1, 2])
torch.Size([2, 1])
```


<br>



마찬가지로 출력 결과를 통해 텐서들의 크기를 확인할 수 있습니다. 여기에서도 브로드캐스팅 규칙을 적용하면 다음과 같이 크기가 변화하며 덧셈 연산을 수행할 수 있습니다.
```
[1, 2] --> [2, 2]
[2, 1] --> [2, 2]
```


<br>


덧셈 연산을 수행하면 다음과 같은 결과를 얻을 수 있을 것입니다.
```py
z = x + y
print(z)
print(z.size())
```
```
[output]
tensor([[4., 5.],
        [6., 7.]])
torch.Size([2, 2])
```


<br>


이처럼 브로드캐스팅을 지원하는 연산의 경우, 크기가 다른 텐서끼리 연산을 수행할 수있습니다. 다만 앞에서의 예제에서 볼 수 있듯이 브로드캐스팅 규칙 자체가 복잡하기 때문에 잘 적용한다면 편리하겠지만 실수가 발생하면 잘못된 결과를 가져올 수도 있습니다.





