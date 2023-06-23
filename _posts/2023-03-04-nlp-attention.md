---
layout: post
title: 어텐션 메커니즘(Attention Mechanism) 이해하기
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





1. Seq2Seq 모델의 한계
2. 어텐션(Attention)의 아이디어
3. 어텐션 함수(Attention Function)
4. 닷-프로덕트(내적) 어텐션(Dot-Product Attention)
    - 4.1 어텐션 스코어(Attention Score)를 구한다.
    - 4.2 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.
    - 4.3 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.
    - 4.4 어텐션 값과 디코더의 $t$ 시점의 은닉 상태를 연결한다.(Concatenate)
    - 4.5 출력층 연산의 입력이 되는 $\tilde{s_t}$ 를 계산합니다.
    - 4.6 $\tilde{s_t}$ 를 출력층의 입력으로 사용합니다.
5. 바다나우 어텐션(Bahdanau Attention)
    - 5.1 어텐션 스코어(Attention Score)를 구한다.
    - 5.2 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.
    - 5.3 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.
    - 5.4 컨텍스트 벡터로부터 $\tilde{s_t}$ 를 구합니다.
6. 다양한 종류의 어텐션(Attention)

<br>
<br>





# 1. Seq2Seq 모델의 한계

**<span style="color:red">seq2seq 모델은 인코더</span>** 에서 **<span style="background-color: #fff5b1">입력 시퀀스를 컨텍스트 벡터라는 하나의 고정된 크기의 벡터 표현으로 압축</span>** 하고, **<span style="color:red">디코더</span>** 는 **<span style="background-color: #fff5b1">이 컨텍스트 벡터를 통해서 출력 시퀀스를 만들어</span>** 냈습니다. 하지만 이러한 RNN에 기반한 seq2seq 모델에는 크게 **<span style="color:red">두 가지 문제</span>** 가 있습니다.

1. 입력 시퀸스의 모든 정보를 하나의 고정된 크기의 벡터(컨텍스트 벡터)에 다 압축 요약하려 하다 보니 정보의 손실이 생길 수밖에 없습니다. 특히 시퀸스의 길이가 길다면 정보의 손실이 더 커집니다.
2. RNN 구조로 만들어진 모델이다 보니, 필연적으로 gradient vaninshing/exploding 현상이 발생합니다.


<br>


결국 이는 기계 번역 분야에서 입력 문장이 길면 번역 품질이 떨어지는 현상으로 나타났습니다. **<u>입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 seq2seq의 문제점을 해결</u>** 하기 위해 **<span style="color:red">Attention Mechanism(어텐션 메커니즘)</span>** 이 제안되었습니다.


<br>





# 2. 어텐션(Attention)의 아이디어

**어텐션의 기본 아이디어** 는 **<span style="color:red">디코더에서 출력 단어를 예측하는 매 시점(time step)마다, 인코더에서의 전체 입력 문장을 다시 한 번 참고한다는 점</span>** 입니다. **<u>단, 전체 입력 문장을 전부 다 동일한 비율로 참고하는 것이 아닌</u>**, **<span style="background-color: #fff5b1">해당 시점에서 예측해야할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중(attention)해서</span>** 보게 됩니다.


<br>





# 3. 어텐션 함수(Attention Function)

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/bebccd97-2472-43e6-a5e7-904f42dc049d">
</p>

<br>


어텐션을 함수로 표현하면 주로 다음과 같이 표현됩니다.

$$
Attention(Q,\ K,\ V) = Attention\ Value
$$


<br>

위 수식, **<span style="color:red">어텐션 함수</span>** 의 $Q$, $K$, $V$ 에 해당되는 각각의 **Query**, **Keys**, **Values** 는 각각 다음과 같습니다.

- $Q$ = **Query** : $t$ 시점의 디코더 셀에서의 은닉 상태
- $K$ = **Keys** : 모든 시점의 인코더 셀의 은닉 상태들
- $V$ = **Values** : 모든 시점의 인코더 셀의 은닉 상태들


<br>

**<span style="color:red">어텐션 함수의 작동방식</span>**

1. **'쿼리(Query)'** 에 대해서 모든 **'키(Key)'** 와의 유사도를 각각 구합니다.
2. 구해낸 이 유사도를 키와 맵핑되어있는 각각의 **'값(Value)'** 에 반영해줍니다.
3. 유사도가 반영된 **'값(Value)'** 을 모두 더해서 리턴합니다.


이때 값(Value)을 모두 더해서 **리턴하는 값** 은 **<span style="color:red">어텐션 값(Attention Value)</span>** 입니다. 간단한 어텐션 예제를 통해 어텐션을 이해해보겠습니다.

<br>





# 4. 닷-프로덕트(내적) 어텐션(Dot-Product Attention)

어텐션은 다양한 종류가 있는데, 여기에서는 **<span style="color:red">닷-프로덕트 어텐션(Dot-Product Attention)</span>** 을 통해 어텐션을 이해해보겠습니다. seq2seq에서 사용되는 어텐션 중에서 닷-프로덕트 어텐션과 다른 어텐션의 차이는 주로 중간 수식의 차이로 메커니즘 자체는 거의 유사합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/7707e7e4-e957-4445-a10b-c012dfdea3f0">
</p>

<br>

어텐션 메커니즘에 대해 위의 그림을 통해 전체적인 개요를 이해해보겠습니다.


위 그림은 디코더의 첫번째, 두번째 LSTM 셀이 이미 어텐션 메커니즘을 통해 je와 suis를 예측하는 과정을 거쳤다고 가정하에, **<u>디코더의 세번째 LSTM 셀에서 출력 단어를 예측할 때, 어텐션 메커니즘을 사용하는 모습</u>** 을 보여줍니다. **<span style="color:red">디코더의 세번째 LSTM 셀은 출력 단어를 예측하기 위해서 인코더의 모든 입력 단어들의 정보를 다시 한번 참고</span>** 하고자 합니다.

<br>

여기서 주목할 것은 **<span style="color:red">인코더의 소프트맥스 함수</span>** 입니다. 위 그림의 **<span style="background-color: #fff5b1">빨간 직사각형</span>** 은 인코더의 소프트맥스 함수를 통해 나온 결과값인, I, am, a, student **<u>단어 각각이 출력 단어를 예측할 때 얼마나 도움이 되는지의 정도를 수치화한 값</u>** 입니다. **<span style="color:red">각 입력 단어는 디코더의 예측에 도움이 되는 정도를 수치화하여 측정되면 이를 하나의 정보로 담아서(위 그림의 초록색 삼각형) 디코더로 전송</span>** 됩니다. 결과적으로, 디코더는 출력 단어를 더 정확하게 예측할 확률이 높아집니다.


<br>



## 4.1 어텐션 스코어(Attention Score)를 구한다.

우선, 인코더의 시점(time step)을 각각 1, 2, ..., $N$ 이라고 했을때, 인코더의 은닉 상태(hidden state)를 각각 $h_1$, $h_2$, ..., $h_N$ 이라고 하고, 디코더의 현재 시점(time step) $t$ 에서의 디코더의 은닉 상태(hidden state)를 $s_t$ 라고 하겠습니다. 또한 여기서는 인코더의 은닉 상태와 디코더의 은닉 상태의 차원이 같다고 가정합니다. 위의 그림의 경우에는 인코더의 은닉 상태와 디코더의 은닉 상태가 동일하게 차원이 4입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/80f6e4f9-f2c0-4330-a49f-bb2301e544a2">
</p>

<br>



**기존의 디코더** 는 현재 시점 $t$ 에서 출력 단어를 예측하기 위해서 **디코더의 셀은 두 개의 입력값을 필요** 로 했습니다. 하나는 **<u>이전 시점 $t-1$ 의 은닉 상태</u>** 와 다른 하나는 **<u>이전 시점 $t-1$ 에서 나온 출력 단어</u>** 입니다. 하지만 **<span style="color:red">어텐션 메커니즘</span>** 에선 출력 단어 예측에 또 다른 값을 필요로 하는데 바로 **<span style="color:red">어텐션 값(Attention Value)이라는 새로운 값을 추가로 필요</span>** 로 합니다. **<u>$t$ 번째 단어를 예측하기 위한 어텐션 값</u>** 을 **<span style="background-color: #fff5b1">$a_t$</span>** 라고 정의하겠습니다.

어텐션 값을 설명하기 이전에 어텐션 스코어를 먼저 이야기하겠습니다. **<span style="color:red">어텐션 스코어</span>** 란 **<u>현재 디코더의 시점 $t$ 에서 단어를 예측하기 위해</u>**, **<span style="color:red">인코더의 모든 은닉 상태 각각이 디코더의 현 시점의 은닉 상태와 얼마나 유사한지를 판단하는 스코어값</span>** 입니다. 위에서 잠깐 설명드렸듯이, 모든 단어를 동일한 비율로 참고하지 않고 **<span style="background-color: #fff5b1">연관성 있는 입력 단어 부분을 집중(attention!)</span>** 해서 본다고 했었습니다.

<br>


**<span style="color:red">닷-프로덕트 어텐션</span>** 에서는 이 스코어 값을 구하기 위해 $s_t$ 를 전치(transpose)하고 **<span style="color:red">각 은닉 상태와 내적(dot product)을 수행</span>** 합니다. **<u>즉, 모든 어텐션 스코어 값은 스칼라</u>** 입니다. 예를 들어 $s_t$ 와 인코더의 $i$ 번째 은닉 상태의 어텐션 스코어의 계산 방법은 아래와 같습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/ea03e0da-9d98-4a42-83a2-1b2dbe17a2d3">
</p>

<br>



어텐션 스코어 함수를 정의해보면 다음과 같습니다.

$$
score(s_t, h_i) = s^T_t h_i
$$

<br>

$s_t$ 와 인코더의 모든 은닉 상태의 어텐션 스코어의 모음값을 $e_t$ 라고 정의한다면, $e_t$ 의 수식은 다음과 같습니다.(여기서 $N$ 은 입력 단어 갯수, 위에선 "I am a student"로 총 4개)

$$
e^t = [s^T_t h_1,\ ...,\ s^T_t h_N]
$$

<br>






## 4.2 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/0efa5e3b-af5c-435d-a40e-1c4a977701cf">
</p>

<br>

**<u>$e^t$ 에 소프트맥스 함수</u>** 를 적용하면 모든 값을 합했을 때 1이 되는 확률 분포를 얻게 됩니다. 이를 **<span style="color:red">어텐션 분포(Attention Distribution)</span>** 라고 하며, 각각의 값은 **<span style="color:red">어텐션 가중치(Attention Weight)</span>** 라고 합니다. 예를 들어 소프트맥스 함수를 적용하여 얻은 출력값인 I, am, a, student의 어텐션 가중치를 각각 0.1, 0.4, 0.1, 0.4라고 했을때 이들의 합은 1이며, 위 그림에서 붉은색 사각형의 크기가 가중치 크기를 나타내고 있습니다.


디코더의 시점 $t$ 에서의 어텐션 가중치의 모음값인 어텐션 분포를 $\alpha_t$ 라고 할 때, 다음과 같이 정의할 수 있습니다.

$$
\alpha_t = softmax(e^t)
$$

<br>





## 4.3 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/def4e78b-af83-4797-a79b-53d1d28a1a39">
</p>

<br>


지금까지 준비해온 정보들을 하나로 합치는 단계로, **<u>어텐션의 최종 결과값을 얻기 위해서 각 인코더의 은닉 상태와 어텐션 가중치값들을 곱하고 모두 더합니다.</u>** 이때 나오는 값이 어텐션의 최종 결과. 즉, 어텐션 함수의 출력값인 **<span style="color:red">어텐션 값(Attention Value) $\alpha_t$</span>** 입니다.

$$
\alpha_t = \sum^N_{i=1} \alpha^t_i h_i
$$

<br>



이러한 어텐션 값 $\alpha_t$ 는 인코더의 문맥을 포함하고 있다고 해서 **<span style="color:red">컨텍스트 벡터(context vector)</span>** 라고도 불립니다.


<br>





## 4.4 어텐션 값과 디코더의 $t$ 시점의 은닉 상태를 연결한다.(Concatenate)

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/bb6d6c1d-7f93-42ee-b7a3-d4d23659bd52">
</p>

<br>



어텐션 값(Attention Value)인 **<span style="color:red">$\alpha_t$</span>** 가 구해지면, $\alpha_t$ 를 디코더의 은닉 상태(hidden state)인 **<span style="color:red">$s_t$ 와 결합(concatenate)하여 하나의 벡터로 만드는 작업을 수행</span>** 하며, 이를 **<span style="background-color: #fff5b1">$v_t$</span>** 라고 정의합니다.

$v_t$ 는 $\hat{y}$ 예측 연산의 입력으로 사용하여 인코더로부터 얻은 정보를 활용해 $\hat{y}$ 를 좀 더 잘 예측할 수 있게 합니다.

<br>





## 4.5 출력층 연산의 입력이 되는 $\tilde{s_t}$ 를 계산합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/fd4fe4ad-01ad-4981-ba9d-73731d11f3d9">
</p>

<br>

어텐션 매커니즘에선 위에서 구한 **<u> $v_t$ 를 바로 출력층으로 보내지 않고 그 전에 신경망 연산을 추가</u>** 합니다. 가중치 행렬과 곱한 후 하이퍼볼릭탄젠트 함수를 지나도록 하여 출력층 연산을 위한 새로운 벡터인 $\tilde{s_t}$ 를 얻습니다. 어텐션 메커니즘을 사용하지 않는 **seq2seq** 에서는 출력층의 입력이 $t$ 시점의 은닉 상태인 $s_t$ 였던 반면, **<span style="color:red">어텐션 메커니즘에서는 출력층의 입력이 $\tilde{s_t}$ 가 되는 셈</span>** 입니다.


식으로 표현하자면 다음과 같습니다. $W_c$ 는 학습 가능한 가중치 행렬, $b_c$ 는 편향, $[a_t ; s_t]$ 는 위에서 결합한 벡터 $v_t$ 를 나타냅니다.

$$
\tilde{s_t} = tanh(W_c [a_t ; s_t] + b_c)
$$

<br>






## 4.6 $\tilde{s_t}$ 를 출력층의 입력으로 사용합니다.

$\tilde{s_t}$ 를 출력층의 입력으로 사용하여 예측 벡터를 얻습니다.

$$
\hat{y}_t = Softmax(W_y \tilde{s_t} + b_y)
$$


<br>




# 5. 바다나우 어텐션(Bahdanau Attention)

이번에는 닷-프로덕트 어텐션보다는 조금 더 복잡하게 설계된 바다나우 어텐션 메커니즘을 이해해보겠습니다. 아래 어텐션 함수는 대체적으로 동일하되 다른 점이 하나 있습니다. 바로 **<span style="color:red">Query 가 디코더 셀의 $t$ 시점의 은닉 상태가 아닌 $t-1$ 시점의 은닉 상태라는 것</span>** 입니다.

$$
Attention(Q,\ K,\ V) = Attention\ Value
$$

<br>

- $t$ = 어텐션 메커니즘이 수행되는 디코더 셀의 현재 시점을 의미
- $Q$ = **Query** : $t-1$ 시점의 디코더 셀에서의 은닉 상태
- $K$ = **Keys** : 모든 시점의 인코더 셀의 은닉 상태들
- $V$ = **Values** : 모든 시점의 인코더 셀의 은닉 상태들

<br>





# 5.1 어텐션 스코어(Attention Score)를 구한다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/8048e43a-04f9-4550-974d-42568b7f4b9a">
</p>

<br>


앞서 닷-프로덕트 어텐션에서는 **Query** 로 디코더의 $t$ 시점의 은닉 상태를 사용한 것과는 달리, 이번에는 그림에서도 알 수 있듯이 **<u>디코더의 $t$ 시점 은닉 상태가 아닌 $t-1$ 시점의 은닉 상태 $s_{t-1}$ 를 사용</u>** 합니다. **<span style="color:red">바다나우 어텐션의 어텐션 스코어 함수</span>**. 즉, $s_{t-1}$ 와 인코더의 $i$ 번째 은닉 상태의 어텐션 스코어 계산 방법은 아래와 같습니다.

$$
score(s_{t-1}, H) = W_a^T tanh(W_b s_{t-1} + W_c h_i)
$$

<br>

여기서 $W_a$, $W_b$, $W_c$ 는 학습 가능한 가중치 행렬입니다. 그리고 $s_{t-1}$ 와 $h_1$, $h_2$, $h_3$, $h_4$ 의 어텐션 스코어를 각각 구해야하므로 병렬 연산을 위해 $h_1$, $h_2$, $h_3$, $h_4$ 를 하나의 행렬 $H$ 로 두면 수식은 아래처럼 변경됩니다.

$$
score(s_{t-1}, H) = W_a^T tanh(W_b s_{t-1} + W_c H)
$$

<br>

아래 그림을 통해 위 수식을 보면 이해가 쉬울 것입니다. $W_c H$ (주황색 박스)와 $W_b s_{t-1}$ (초록색 박스)는 아래와 같습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/2bcccc95-fb7a-4195-be54-a3e86c86c5e8">
</p>

<br>

이들을 더한 후에는 하이퍼볼릭탄젠트 함수를 지나도록 합니다.

$$
tanh(W_b s_{t-1} + W_c H)
$$

<br>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/c94e0bbb-bdfe-4ba5-b22c-0f135336169c">
</p>

<br>

이제 $W_a^T$ 와 곱하여 $s_{t-1}$ 와 $h_1$, $h_2$, $h_3$, $h_4$ 의 유사도가 기록된 어텐션 스코어 벡터 $e^t$ 를 얻습니다.

$$
e^t = W_a^T tanh(W_b s_{t-1} + W_c H)
$$

<br>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/67f8e691-c4e0-4d91-baa1-460f2d60025d">
</p>

<br>






# 5.2 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/fdf4775f-b8b9-412b-9c5c-4f93b701fcd3">
</p>

<br>

$e^t$ 에 소프트맥스 함수를 적용하면 모든 값의 합이 1이 되는 확률 분포를 얻어내고, 이를 **<span style="color:red">어텐션 분포(Attention Distribution)</span>** 라고 부릅니다. 또 각각의 값은 **<span style="color:red">어텐션 가중치(Attention Weight)</span>** 라고 합니다.


<br>






# 5.3 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/d5207522-4e7e-49e0-8616-eea019aed15d">
</p>

<br>

지금까지 얻은 정보들을 하나로 합치는 단계로, 각 인코더의 은닉 상태와 어텐션 가중치값들을 곱하고 최종적으로 모두 더합니다. 해당 벡터는 인코더의 문맥을 포함하고 있기 때문에 이를 **<span style="color:red">컨텍스트 벡터(context vector)</span>** 라고 부릅니다.

<br>





# 5.4 컨텍스트 벡터로부터 $s_t$ 를 구합니다.

기존의 **LSTM** 은 이전 시점의 셀로부터 전달받은 은닉 상태 $s_{t-1}$ 와 현재 시점의 입력 $x_t$ 를 가지고 연산하였습니다. 아래의 LSTM은 seq2seq의 디코더이며 현재 시점의 입력 $x_t$ 는 임베딩된 단어 벡터입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/17e75f17-cff7-471e-ae78-027ba7e8a91f">
</p>

<br>



**<span style="color:red">바다나우 어텐션 메커니즘에서의 LSTM 작동 방식</span>** 을 살펴보겠습니다. 기존의 LSTM 작동 방식과 달리 현재 시점의 입력 $x_t$ 가 특이합니다. 바다나우 어텐션 메커니즘에서는 **<u>컨텍스트 벡터와 현재 시점의 입력인 단어의 임베딩 벡터를 연결(concatenate)하고, 현재 시점의 새로운 입력으로 사용</u>** 하는 모습을 보여줍니다. 그리고 이전 시점의 셀로부터 전달받은 은닉 상태 $s_{t-1}$ 와 현재 시점의 새로운 입력으로부터 $s_t$ 를 구합니다.


기존의 LSTM이 임베딩된 단어 벡터를 입력으로 하는 것에서 컨텍스트 벡터와 임베딩된 단어 벡터를 연결(concatenate)하여 입력으로 사용하는 것이 달라졌습니다. 이후의 과정은 어텐션 메커니즘을 사용하지 않는 경우, 즉 일반적인 LSTM을 실행시키는 것과 동일합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/e04fa0cf-dd7f-4e4d-811f-8bf4ce8297d1">
</p>

<br>

사실 **concatenate** 는 닷-프로덕트 어텐션 메커니즘에서도 사용했던 방식입니다. 다른 점은 **<span style="color:red">바다나우 어텐션</span>** 에서는 컨텍스트 벡터를 다음 단어를 예측하는 데 입력 단어에 합친다는 것이고, **<span style="color:red">닷-프로덕트 어텐션</span>** 에서는 이전 단어의 예측값인 $s_t$ 에 컨텍스트 벡터를 합치고($v_t$), 가중치 행렬 $W_c$ 와 곱한 뒤 하이퍼볼릭탄젠트 함수를 거쳐 예측 벡터인 $\hat{s}_t$ 를 얻었습니다.


<br>





# 6. 다양한 종류의 어텐션(Attention)

어텐션의 종류는 다양합니다. 어텐션 차이는 주로 중간 수식인 어텐션 스코어 함수 차이를 말하며, 어텐션 스코어를 구하는 방법은 여러가지가 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/481a31f4-fa7d-4a2b-a68b-1d469db9f0b1">
</p>





