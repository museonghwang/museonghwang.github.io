---
layout: post
title: 장단기 메모리(LSTM) 개념 이해하기
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





1. Vanilla RNN의 한계
2. Vanilla RNN 내부 열어보기
3. LSTM(Long Short‐Term Memory)
    - (1) 입력 게이트 : 현재 정보를 기억하기 위한 게이트
    - (2) 삭제 게이트 : 기억을 삭제하기 위한 게이트
    - (3) 셀 상태
    - (4) 출력 게이트와 은닉 상태 : 현재 시점 $x_t$ 의 은닉 상태를 결정
4. 케라스 SimpleRNN 이해하기
5. 케라스 LSTM 이해하기
6. Bidirectional(LSTM) 이해하기

<br>




# 1. Vanilla RNN의 한계

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/5336d4db-fce9-4163-a7b5-1bb62c26a908">
</p>

<br>


**<span style="color:red">Vanilla RNN</span>** 은 **<u>출력 결과가 이전의 계산 결과에 의존</u>** 하는데, **<span style="background-color: #fff5b1">비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점</span>** 이 있습니다. 즉, **<span style="color:red">Vanilla RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생</span>** 합니다. 위의 그림은 첫번째 입력값인 $x_1$ 의 정보량을 짙은 남색으로 표현했을 때, 색이 점차 얕아지는 것으로 시점이 지날수록 $x_1$ 의 정보량이 손실되어가는 과정을 표현하였습니다. 뒤로 갈수록 $x_1$ 의 정보량은 손실되고, 시점이 충분히 긴 상황에서는 $x_1$ 의 전체 정보에 대한 영향력은 거의 의미가 없을수도 있습니다.


**<u>어쩌면 가장 중요한 정보가 시점의 앞쪽에 위치할 수도 있습니다.</u>** RNN으로 만든 언어 모델이 다음 단어를 예측하는 과정을 생각해봅시다. 예를 들어 다음 문장이 있습니다.
```
모스크바에 여행을 왔는데 건물도 예쁘고 먹을 것도 맛있었어.
그런데 글쎄 직장 상사한테 전화가 왔어. 어디냐고 묻더라구.
그래서 나는 말했지. 저 ___ 여행왔는데요.
```

<br>

다음 단어를 예측하기 위해서는 장소 정보가 필요합니다. 그런데 장소 정보에 해당되는 단어인 '모스크바' 는 앞에 위치하고 있고, **<span style="background-color: #fff5b1">RNN이 충분한 기억력을 가지고 있지 못한다면 다음 단어를 엉뚱하게 예측합니다.</span>** 이를 **<span style="color:red">장기 의존성 문제(the problem of Long‐Term Dependencies)</span>** 라고 합니다.

<br>





# 2. Vanilla RNN 내부 열어보기

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/5358531a-a7d6-43fd-8ff6-141141287eee">
</p>

<br>

위 그림은 **Vanilla RNN** 의 내부 구조를 보여줍니다. 위 그림에 그림에 편향 $b$ 를 그린다면 $x_t$ 옆에 $tanh$ 로 향하는 또 하나의 입력선을 그리면 됩니다.

$$
h_t = tanh(W_x x_t + W_h h_{t-1} + b)
$$

<br>

**Vanilla RNN** 은 $x_t$ 와 $h_{t-1}$ 이라는 두 개의 입력이 각각의 가중치와 곱해져서 메모리 셀의 입력이 됩니다. 그리고 이를 하이퍼볼릭탄젠트 함수의 입력으로 사용하고 이 값은 은닉층의 출력인 은닉 상태가 됩니다.

<br>





# 3. LSTM(Long Short‐Term Memory)

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/69c39718-a9ff-4d1d-a89c-92d124e3584b">
</p>

<br>


위 그림은 RNN의 단점을 보완한 단점을 보완한 RNN의 일종을 **<span style="color:red">장단기 메모리(Long Short‐Term Memory, LSTM)</span>** 라고 하며, **<span style="color:red">LSTM</span>** 의 전체적인 내부의 모습을 보여줍니다. **LSTM** 은 은닉층의 메모리 셀에 **<span style="background-color: #fff5b1">입력 게이트</span>**, **<span style="background-color: #fff5b1">망각 게이트</span>**, **<span style="background-color: #fff5b1">출력 게이트</span>** 를 추가하여 **<span style="color:red">불필요한 기억을 지우고, 기억 해야할 것들을 정합니다.</span>**



요약하면 **LSTM** 은 은닉 상태(hidden state)를 계산하는 식이 전통적인 RNN보다 조금 더 복잡해졌으며 **<span style="background-color: #fff5b1">셀 상태(cell state)</span>** 라는 값을 추가하였습니다. 위의 그림에서는 $t$ 시점의 **<span style="background-color: #fff5b1">셀 상태를 $C_t$ 로 표현</span>** 하고 있습니다. **LSTM** 은 RNN과 비교하여 **긴 시퀀스의 입력을 처리하는데 탁월한 성능** 을 보입니다.

<br>


<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/09a767c9-b62c-4720-9ec2-5e22978b4ea1">
</p>

<br>



**<span style="color:red">셀 상태(cell state)</span>** 는 위 그림에서 **<u>왼쪽에서 오른쪽으로 가는 굵은선</u>** 입니다. 셀 상태 또한 은닉 상태 처럼 **<span style="color:red">이전 시점의 셀 상태가 다음 시점의 셀 상태를 구하기 위한 입력으로서 사용</span>** 됩니다.


**<u>은닉 상태의 값과 셀 상태의 값을 구하기 위해</u>** 서 **<span style="color:red">새로 추가 된 3개의 게이트를 사용</span>** 합니다. 각 게이트는 **<span style="background-color: #fff5b1">삭제 게이트</span>**, **<span style="background-color: #fff5b1">입력 게이트</span>**, **<span style="background-color: #fff5b1">출력 게이트</span>** 라고 부르며 이 3개의 게이트에는 공통적으로 시그모이드 함수가 존재합니다. 시그모이드 함수를 지나면 0과 1사이의 값이 나오게 되는데 이 값들을 가지고 게이트를 조절합니다. 아래의 내용을 참고로 각 게이트에 대해 알아보겠습니다.

- $σ$ : 시그모이드 함수
- $tanh$ : 하이퍼볼릭탄젠트 함수
- $W_{xi}$, $W_{xg}$, $W_{xf}$, $W_{xo}$ : $x_t$ 와 함께 각 게이트에서 사용되는 4개의 가중치
- $W_{hi}$, $W_{hg}$, $W_{hf}$, $W_{ho}$ : $h_{t-1}$ 와 함께 각 게이트에서 사용되는 4개의 가중치
- $b_i$, $b_g$, $b_f$, $b_o$ : 각 게이트에서 사용되는 4개의 편향

<br>




## (1) 입력 게이트 : 현재 정보를 기억하기 위한 게이트

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/003bed8c-4e74-406e-ac46-b23d92c4b01f">
</p>

<br>

$$
i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)
$$


$$
g_t = tanh(W_{xg} x_t + W_{hg} h_{t-1} + b_g)
$$

<br>


- $i_t$
    - $\sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)$
    - 현재시점 $t$ 의 $x_t$ 값과 입력 게이트로 이어지는 가중치 $W_{xi}$ 를 곱한 값과, 이전 시점 $t‐1$ 의 은닉 상태 $h_{t-1}$ 가 입력 게이트로 이어지는 가중치 $W_{hi}$ 를 곱한 값을 더하여 **시그모이드 함수** 를 지납니다.
    - 시그모이드 함수를 지나 **0과 1사이의 값을 가짐.**
- $g_t$
    - $tanh(W_{xg} x_t + W_{hg} h_{t-1} + b_g)$
    - 현재시점 $t$ 의 $x_t$ 값과 입력 게이트로 이어지는 가중치 $W_{xg}$ 를 곱한 값과, 이전 시점 $t‐1$ 의 은닉 상태 $h_{t-1}$ 가 입력 게이트로 이어지는 가중치 $W_{hg}$ 를 곱한 값을 더하여 **하이퍼볼릭탄젠트 함수** 를 지납니다.
    - 하이퍼볼릭탄젠트 함수를 지나 **-1과 1사이의 값을 가짐.**


<br>

0과 1사이의 값을 가지는 **<span style="background-color: #fff5b1">$i_t$</span>** 와 -1과 1사이의 값을 가지는 **<span style="background-color: #fff5b1">$g_t$</span>**, **<span style="color:red">이 두 개의 값을 가지고 이번에 선택된 기억할 정보의 양을 정합니다.</span>**

<br>




## (2) 삭제 게이트 : 기억을 삭제하기 위한 게이트

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/ea8a43ea-ce0e-4a84-b489-89defb7edc20">
</p>

<br>

$$
f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)
$$

<br>

- $f_t$
    - $\sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)$
    - 현재시점 $t$ 의 $x_t$ 값과 삭제 게이트로 이어지는 가중치 $W_{xf}$ 를 곱한 값과, 이전 시점 $t‐1$ 의 은닉 상태 $h_{t-1}$ 가 삭제 게이트로 이어지는 가중치 $W_{hf}$ 를 곱한 값을 더하여 **시그모이드 함수** 를 지나게 됩니다.
    - 시그모이드 함수를 지나 **0과 1사이의 값을 가짐.**

<br>


0과 1사이의 값을 가지는 **<span style="background-color: #fff5b1">$f_t$</span>**, **<span style="color:red">이 값이 곧 삭제 과정을 거친 정보의 양</span>** 입니다. **<span style="background-color: #fff5b1">0에 가까울수록 정보가 많이 삭제된 것이고 1에 가까울수록 정보를 온전히 기억한 것</span>** 입니다. 이를 가지고 셀 상태를 구하게 됩니다.


<br>




## (3) 셀 상태

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/0de659f9-54de-4fac-b7b6-8cc89b1a1346">
</p>

<br>

$$
C_t = f_t ∘ C_{t−1} + i_t ∘ g_t
$$

<br>

- $C_t$
    - 현재 삭제 게이트에서 일부 기억을 잃은 상태
    - 입력 게이트에서 구한 $i_t$, $g_t$ 이 두 개의 값에 대해서 원소별 곱(entrywise product)을 진행하여 같은 위치의 성분끼리 곱합니다. 이것이 이번에 선택된 기억할 값입니다.
    - **입력 게이트에서 선택된 기억을 삭제 게이트의 결과값과 더합니다.** 이 값을 **<span style="color:red">현재 시점 $t$ 의 셀 상태</span>** 라고 하며, **<span style="color:red">이 값은 다음 $t+1$ 시점의 LSTM 셀로 넘겨집니다.</span>**

<br>


삭제 게이트와 입력 게이트의 영향력을 이해해봅시다.

만약 삭제 게이트의 출력값인 $f_t$ 가 0이 된다면, 이전 시점의 셀 상태의 값인 $C_{t-1}$ 은 현재 시점의 셀 상태의 값을 결정하기 위한 영향력이 0이 되면서, 오직 입력 게이트의 결과만이 현재 시점의 셀 상태의 값 $C_t$ 을 결정할 수 있습니다. **<u>이는 삭제 게이트가 완전히 닫히고 입력 게이트를 연 상태를 의미</u>** 합니다. 반대로 입력 게이트의 $i_t$ 값을 0이라고 한다면, 현재 시점의 셀 상태의 값 $C_t$ 는 오직 이전 시점의 셀 상태의 값 $C_{t-1}$ 의 값에만 의존합니다. **<u>이는 입력 게이트를 완전히 닫고 삭제 게이트만을 연 상태를 의미</u>**합니다.


결과적으로 **<span style="color:red">삭제 게이트는 이전 시점의 입력을 얼마나 반영할지를 의미</span>** 하고, **<span style="color:red">입력 게이트는 현재 시점의 입력을 얼마나 반영할지를 결정</span>** 합니다.

<br>




## (4) 출력 게이트와 은닉 상태 : 현재 시점 $x_t$ 의 은닉 상태를 결정

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/32885e99-8151-4fe2-94e5-1ff9b5196420">
</p>

<br>

$$
o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_0)
$$

$$
h_t = o_t ∘ tanh(c_t)
$$

<br>

- $o_t$
    - $\sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_0)$
    - 출력 게이트는 현재 시점 $t$ 의 $x_t$ 값과 이전 시점 $t‐1$ 의 은닉 상태가 시그모이드 함수를 지난 값입니다.
    - 해당 값은 현재 시점 $x_t$ 의 은닉 상태를 결정하는 일에 쓰이게 됩니다.

<br>

셀 상태의 값 $c_t$ 가 하이퍼볼릭탄젠트 함수를 지나 -1과 1사이의 값이 되고, 해당 값은 출력 게이트의 값과 연산되면서, **<u>값이 걸러지는 효과가 발생</u>** 하여 **<span style="color:red">은닉 상태</span>** 가 됩니다. 은닉 상태의 값은 또한 출력층으로도 향합니다.


<br>




# 4. 케라스 SimpleRNN 이해하기

우선 **RNN** 과 **LSTM** 을 테스트하기 위한 임의의 입력을 만듭니다.
```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Bidirectional

train_X = [[0.1, 4.2, 1.5, 1.1, 2.8],
           [1.0, 3.1, 2.5, 0.7, 1.1],
           [0.3, 2.1, 1.5, 2.1, 0.1],
           [2.2, 1.4, 0.5, 0.9, 1.1]]

print(np.shape(train_X))
```
```
[output]
(4, 5)
```


<br>


위 입력은 **단어 벡터의 차원은 5** 이고, **문장의 길이가 4** 인 경우를 가정한 입력입니다. 다시 말해 **4번의 시점(timesteps)이 존재하고, 각 시점마다 5차원의 단어 벡터가 입력으로 사용** 됩니다. 그런데 RNN은 2D 텐서가 아니라 3D 텐서를 입력을 받습니다. 즉, 위에서 만든 2D 텐서를 **3D 텐서** 로 변경하겠습니다. 이는 배치 크기 1을 추가해줌으로서 해결합니다.
```py
train_X = [[[0.1, 4.2, 1.5, 1.1, 2.8],
            [1.0, 3.1, 2.5, 0.7, 1.1],
            [0.3, 2.1, 1.5, 2.1, 0.1],
            [2.2, 1.4, 0.5, 0.9, 1.1]]]

train_X = np.array(train_X, dtype=np.float32)
print(train_X.shape)
```
```
[output]
(1, 4, 5)
```


<br>


**<span style="color:red">(batch_size, timesteps, input_dim)</span>** 에 해당되는 (1, 4, 5)의 크기를 가지는 3D 텐서가 생성되었습니다. **batch_size** 는 한 번에 **RNN** 이 학습하는 데이터의 양을 의미하지만, 여기서는 샘플이 1개 밖에 없으므로 **batch_size** 는 1입니다.

<br>



위에서 생성한 데이터를 **SimpleRNN** 의 입력으로 사용하여 **SimpleRNN** 의 출력값을 이해해보겠습니다. SimpleRNN에는 여러 인자가 있으며 대표적인 인자로 **return_sequences** 와 **return_state** 가 있으며, 기본값으로는 둘 다 **False** 로 지정되어져 있으므로 별도 지정을 하지 않을 경우에는 **False** 로 처리됩니다. 우선, 은닉 상태의 크기를 3으로 지정하고, 두 인자 값이 모두 **False** 일 때의 출력값을 보겠습니다. 출력값 자체보다는 해당 값의 **크기(shape)에 주목** 해야합니다.
```py
rnn = SimpleRNN(3)
# rnn = SimpleRNN(3, return_sequences=False, return_state=False)와 동일.
hidden_state = rnn(train_X)

print('hidden state : {} shape: {}'.format(hidden_state, hidden_state.shape))
```
```
[output]
hidden state : [[-0.9577472  -0.33443117 -0.16784662]] shape: (1, 3)
```


<br>

**(1, 3)** 크기의 텐서가 출력되는데, 이는 마지막 시점의 은닉 상태입니다. 은닉 상태의 크기를 3으로 지정했음을 주목합시다. 기본적으로 **return_sequences** 가 **False** 인 경우에는 **SimpleRNN** 은 마지막 시점의 은닉 상태만 출력합니다. 이번에는 **return_sequences** 를 **True** 로 지정하여 모든 시점의 은닉 상태를 출력해봅시다.
```py
rnn = SimpleRNN(3, return_sequences=True)
hidden_states = rnn(train_X)

print('hidden states : \n{} shape: {}'.format(hidden_states, hidden_states.shape))
```
```
[output]
hidden states : 
[[[-0.07684275 -0.9996449  -0.99920934]
  [-0.6308924  -0.9999172  -0.9968455 ]
  [ 0.6306296  -0.99580896 -0.7375661 ]
  [-0.7447482  -0.9749265  -0.9728286 ]]] shape: (1, 4, 3)
```


<br>


**(1, 4, 3)** 크기의 텐서가 출력됩니다. 앞서 입력 데이터는 **(1, 4, 5)** 의 크기를 가지는 3D 텐서였고, 그 중 4가 시점(timesteps)에 해당하는 값이므로 모든 시점에 대해서 은닉 상태의 값을 출력하여 **(1, 4, 3)** 크기의 텐서를 출력하는 것입니다.

**return_state** 가 **True** 일 경우에는 **return_sequences** 의 **True/False** 여부와 상관없이 마지막 시점의 은닉 상태를 출력합니다. 가령, **return_sequences** 가 **True** 이면서, **return_state**를 **True** 로 할 경우 **SimpleRNN** 은 두 개의 출력을 리턴합니다.
```py
rnn = SimpleRNN(3, return_sequences=True, return_state=True)
hidden_states, last_state = rnn(train_X)

print('hidden states : \n{} shape: {}'.format(hidden_states, hidden_states.shape))
print('last hidden state : {} shape: {}'.format(last_state, last_state.shape))
```
```
[output]
hidden states : 
[[[ 0.05431597  0.9997256   0.07848608]
  [-0.5503005   0.96039253  0.61889   ]
  [ 0.7662684   0.48871595  0.8754748 ]
  [ 0.949467    0.854687   -0.16799167]]] shape: (1, 4, 3)
last hidden state : [[ 0.949467    0.854687   -0.16799167]] shape: (1, 3)
```


<br>


첫번째 출력은 **return_sequences=True** 로 인한 출력으로 모든 시점의 은닉 상태입니다. 두번째 출력은 **return_state=True** 로 인한 출력으로 마지막 시점의 은닉 상태입니다. 실제로 출력을 보면 모든 시점의 은닉 상태인 **(1, 4, 3)** 텐서의 마지막 벡터값이 **return_state=True** 로 인해 출력된 벡터값과 일치하는 것을 볼 수 있습니다. (둘 다 [-0.63698626 -0.6929572  -0.9387183 ])

그렇다면 **return_sequences** 는 **False** 인데, **retun_state** 가 **True** 인 경우를 살펴보겠습니다.
```py
rnn = SimpleRNN(3, return_sequences=False, return_state=True)
hidden_state, last_state = rnn(train_X)

print('hidden state : {} shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {} shape: {}'.format(last_state, last_state.shape))
```
```
[output]
hidden state : [[ 0.9969874   0.922215   -0.44041932]] shape: (1, 3)
last hidden state : [[ 0.9969874   0.922215   -0.44041932]] shape: (1, 3)
```


<br>


두 개의 출력 모두 마지막 시점의 은닉 상태를 출력하게 됩니다.


<br>




# 5. 케라스 LSTM 이해하기

이번에는 임의의 입력에 대해서 **LSTM** 을 사용할 경우를 보겠습니다. 우선 **return_sequences** 를 **False** 로 두고, **return_state** 가 **True** 인 경우를 보겠습니다.
```py
lstm = LSTM(3, return_sequences=False, return_state=True)
hidden_state, last_state, last_cell_state = lstm(train_X)

print('hidden state : {} shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {} shape: {}'.format(last_state, last_state.shape))
print('last cell state : {} shape: {}'.format(last_cell_state, last_cell_state.shape))
```
```
[output]
hidden state : [[-0.23502806 -0.45216066  0.06345625]] shape: (1, 3)
last hidden state : [[-0.23502806 -0.45216066  0.06345625]] shape: (1, 3)
last cell state : [[-0.38973868 -0.6697613   0.10231213]] shape: (1, 3)
```


<br>


**SimpleRNN** 때와는 달리, 세 개의 결과를 반환합니다. **return_sequences** 가 **False** 이므로 우선 첫번째 결과는 마지막 시점의 은닉 상태입니다. 그런데 **LSTM** 이 **SimpleRNN** 과 다른 점은 **return_state** 를 **True** 로 둔 경우에는 마지막 시점의 은닉 상태뿐만 아니라 셀 상태까지 반환한다는 점입니다. 이번에는 **return_sequences** 를 **True**로 바꿔보겠습니다.
```py
lstm = LSTM(3, return_sequences=True, return_state=True)
hidden_states, last_hidden_state, last_cell_state = lstm(train_X)

print('hidden states : \n{} shape: {}'.format(hidden_states, hidden_states.shape))
print('last hidden state : {} shape: {}'.format(last_hidden_state, last_hidden_state.shape))
print('last cell state : {} shape: {}'.format(last_cell_state, last_cell_state.shape))
```
```
[output]
hidden states : 
[[[0.06367525 0.42689556 0.25716597]
  [0.10355692 0.32117185 0.3720547 ]
  [0.03180264 0.4889893  0.34424222]
  [0.10580046 0.3162607  0.3505974 ]]] shape: (1, 4, 3)
last hidden state : [[0.10580046 0.3162607  0.3505974 ]] shape: (1, 3)
last cell state : [[0.23588046 0.63385934 1.5717858 ]] shape: (1, 3)
```


<br>

**return_state** 가 **True** 이므로 두번째 출력값이 마지막 은닉 상태, 세번째 출력값이 마지막 셀 상태인 것은 변함없지만 **return_sequences** 가 **True** 이므로 첫번째 출력값은 모든 시점의 은닉 상태가 출력됩니다.


<br>




# 6. Bidirectional(LSTM) 이해하기

**양방향 LSTM** 의 출력값을 확인해보겠습니다. **return_sequences** 가 **True** 인 경우와 **False** 인 경우에 대해서 은닉 상태의 값이 어떻게 바뀌는지 직접 비교하기 위해서 이번에는 출력되는 은닉 상태의 값을 고정시켜주겠습니다.
```py
k_init = tf.keras.initializers.Constant(value=0.1)
b_init = tf.keras.initializers.Constant(value=0)
r_init = tf.keras.initializers.Constant(value=0.1)
```


<br>


우선 **return_sequences** 가 **False** 이고, **return_state** 가 **True** 인 경우입니다.
```py
bilstm = Bidirectional(LSTM(3, return_sequences=False, return_state=True, \
                            kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state : {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state : {}, shape: {}'.format(backward_h, backward_h.shape))
```
```
[output]
hidden states : [[0.6301636 0.6301636 0.6301636 0.7037439 0.7037439 0.7037439]], shape: (1, 6)
forward state : [[0.6301636 0.6301636 0.6301636]], shape: (1, 3)
backward state : [[0.7037439 0.7037439 0.7037439]], shape: (1, 3)
```


<br>


이번에는 무려 5개의 값을 반환합니다. **return_state** 가 **True** 인 경우에는 정방향 LSTM의 은닉 상태와 셀 상태, 역방향 LSTM의 은닉 상태와 셀 상태 4가지를 반환하기 때문입니다. 다만, 셀 상태는 각각 forward_c와 backward_c에 저장만 하고 출력하지 않았습니다.

첫번째 출력값의 크기가 **(1, 6)** 인 것에 주목합시다. 이는 **return_sequences** 가 **False** 인 경우 **<u>정방향 LSTM의 마지막 시점의 은닉 상태와 역방향 LSTM의 첫번째 시점의 은닉 상태가 연결된 채 반환</u>** 되기 때문입니다. 그림으로 표현하면 아래와 같이 연결되어 다음층에서 사용됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/0ab8656e-b7de-4112-a8e5-001ed1e87245">
</p>

<br>



마찬가지로 **return_state** 가 **True** 인 경우에 **<u>반환한 은닉 상태의 값인 forward_h와 backward_h는 각각 정방향 LSTM의 마지막 시점의 은닉 상태와 역방향 LSTM의 첫번째 시점의 은닉 상태값</u>** 입니다. 그리고 **<u>이 두 값을 연결한 값이 hidden_states에 출력되는 값</u>** 입니다.


정방향 LSTM의 마지막 시점의 은닉 상태값과 역방향 LSTM의 첫번째 은닉 상태값을 기억해둡시다.

- 정방향 LSTM의 마지막 시점의 은닉 상태값 : [0.6303139 0.6303139 0.6303139]
- 역방향 LSTM의 첫번째 시점의 은닉 상태값 : [0.70387346 0.70387346 0.70387346]  

<br>

현재 은닉 상태의 값을 고정시켜두었기 때문에 **return_sequences** 를 **True** 로 할 경우, 출력이 어떻게 바뀌는지 비교가 가능합니다.
```py
bilstm = Bidirectional(LSTM(3, return_sequences=True, return_state=True, \
                            kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)

print('hidden states : \n{} shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state : {} shape: {}'.format(forward_h, forward_h.shape))
print('backward state : {} shape: {}'.format(backward_h, backward_h.shape))
```
```
[output]
hidden states : 
[[[0.35896602 0.35896602 0.35896602 0.7037439  0.7037439  0.7037439 ]
  [0.5509713  0.5509713  0.5509713  0.5884772  0.5884772  0.5884772 ]
  [0.5910032  0.5910032  0.5910032  0.39501813 0.39501813 0.39501813]
  [0.6301636  0.6301636  0.6301636  0.21932526 0.21932526 0.21932526]]] shape: (1, 4, 6)
forward state : [[0.6301636 0.6301636 0.6301636]] shape: (1, 3)
backward state : [[0.7037439 0.7037439 0.7037439]] shape: (1, 3)
```


<br>



**hidden states** 의 출력값에서는 이제 모든 시점의 은닉 상태가 출력됩니다. 역방향 LSTM의 첫번째 시점의 은닉 상태는 더 이상 정방향 LSTM의 마지막 시점의 은닉 상태와 연결되는 것이 아니라 정방향 LSTM의 첫번째 시점의 은닉 상태와 연결됩니다.

그림으로 표현하면 다음과 같이 연결되어 다음층의 입력으로 사용됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/77e952d3-20af-4bdf-b219-d8101af093d5">
</p>






