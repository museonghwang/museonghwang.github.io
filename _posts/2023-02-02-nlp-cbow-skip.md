---
layout: post
title: Word2Vec - CBOW의 이해
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





1. Word2Vec의 학습방식
2. CBoW(Continuous Bag of Words)
3. Skip-gram
4. CBOW Vs Skip-gram
    - 4.1 CBOW
    - 4.2 Skip-gram
5. NNLM Vs Word2Vec(CBOW)
6. 한계점

Word2Vec의 학습방식인 CBOW와 Skip-gram에 대해 살펴보겠습니다.

<br>
<br>





# 1. Word2Vec의 학습방식

**<span style="color:red">Word2Vec</span>** 은 기본적으로 NNLM을 개선한 모델로, 이전 단어들로부터 다음 단어를 예측하는 목표는 버리고, **<span style="color:red">임베딩 그 자체에만 집중</span>** 했습니다. **<u>Word2Vec는 학습방식에 따라 크게 2가지</u>** 로 나눌 수 있습니다. **CBOW(Continuous Bag of Words)** 와 **Skip-gram** 두 가지 방식이 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/558b2f30-3bae-4c7a-8a55-aecab410a2df">
</p>

<br>



**<span style="color:red">CBOW</span>** 는 **<u>주변 단어(Context Word)로 중간에 있는 단어인 중심 단어(Center Word)를 예측하는 방법</u>** 입니다. 반대로, **<span style="color:red">Skip-gram</span>** 은 **<u>중심 단어를 바탕으로 주변 단어들을 예측하는 방법</u>** 입니다. 메커니즘 자체는 거의 동일합니다. 먼저 CBOW에 대해서 알아보겠습니다.









# 2. CBoW(Continuous Bag of Words)

먼저 CBOW의 메커니즘에 대해서 알아보겠습니다. 다음과 같은 예문이 있다고 가정하겠습니다.

$$
"The\ \ fat\ \ cat\ \ sat\ \ on\ \ the\ \ mat"
$$

<br>


예를 들어서 갖고 있는 코퍼스에 위와 같은 예문이 있다고 했을때, **['The', 'fat', 'cat', 'on', 'the', 'mat']으로** 부터 **<span style="background-color: #fff5b1">sat을 예측하는 것은 CBOW가 하는 일</span>** 입니다. 이때 **<u>예측해야하는 단어 sat</u>** 을 **<span style="color:red">중심 단어(center word)</span>** 라고 하고, **<u>예측에 사용되는 단어들</u>** 을 **<span style="color:red">주변 단어(context word)</span>** 라고 합니다.


**<u>중심 단어를 예측하기 위해서 앞, 뒤로 몇 개의 단어를 볼지를 결정</u>** 해야 하는데 이 **범위** 를 **<span style="color:red">윈도우(window)</span>** 라고 합니다. 예를 들어 윈도우 크기가 2이고, 예측하고자 하는 중심 단어가 sat이라고 한다면 앞의 두 단어인 fat와 cat, 그리고 뒤의 두 단어인 on, the를 입력으로 사용합니다. 윈도우 크기가 $n$ 이라고 한다면, 실제 중심 단어를 예측하기 위해 참고하려고 하는 주변 단어의 개수는 $2n$ 입니다. **윈도우 크기가 정해지면 윈도우를 옆으로 움직여서 주변 단어와 중심 단어의 선택을 변경해가며 학습을 위한 데이터 셋을 만드는데 이 방법** 을 **<span style="color:red">슬라이딩 윈도우(sliding window)</span>** 라고 합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/c7616370-dc01-4103-ae41-4d691193104e">
</p>

<br>


즉, **<span style="color:red">CBOW는 주변 단어들로부터 중심 단어를 예측하는 과정에서 임베딩 벡터를 학습하며, 윈도우 크기를 정해준다면, 주어진 텍스트로부터 훈련 데이터를 자체 구축</span>** 합니다.

위 그림에서 좌측의 중심 단어와 주변 단어의 변화는 윈도우 크기가 2일때, 슬라이딩 윈도우가 어떤 식으로 이루어지면서 데이터 셋을 만드는지 보여줍니다. Word2Vec에서 입력은 모두 원-핫 벡터가 되어야 하는데, 우측 그림은 중심 단어와 주변 단어를 어떻게 선택했을 때에 따라서 각각 어떤 원-핫 벡터가 되는지를 보여줍니다. **<u>위 그림은 결국 CBOW를 위한 전체 데이터 셋을 보여주는 것</u>** 입니다.


<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/691b9c4b-5380-4e86-8e2e-569df10639da">
</p>

<br>

**CBOW의 인공 신경망** 을 간단히 도식화하면 위와 같습니다. 입력층(Input layer)의 입력으로서 앞, 뒤로 사용자가 정한 윈도우 크기 범위 안에 있는 주변 단어들의 원-핫 벡터가 들어가게 되고, 출력층(Output layer)에서 예측하고자 하는 중간 단어의 원-핫 벡터가 레이블로서 필요합니다.

위 그림에서 알 수 있는 사실은 Word2Vec은 NNLM과 달리 총 3개의 Layer로 구성되며, 은닉층(Hidden layer)이 1개인 얕은 신경망(shallow neural network)이라는 점입니다. 또한 Word2Vec의 은닉층은 일반적인 은닉층과는 달리 활성화 함수가 존재하지 않으며 Lookup Table 이라는 연산을 담당하는 층으로 투사층(Projection layer)이라고 부르기도 합니다.

<br>



CBOW의 인공 신경망을 좀 더 확대하여, 동작 메커니즘에 대해서 상세하게 알아보겠습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/ec60e704-ae19-40c4-be82-b30eda85a10c">
</p>

<br>


이 그림에서 주목해야할 것은 두 가지 입니다.

1. **투사층(Projection layer)의 크기는 $M$**
    - CBOW에서 투사층의 크기 $M$ 은 임베딩하고 난 벡터의 차원이 됨.
    - 위 그림에서 투사층의 크기는 $M=5$ 이므로 CBOW를 수행하고나서 얻는 각 단어의 임베딩 벡터의 차원은 5가 될 것.
2. **입력층과 투사층 사이의 가중치 $W$ 는 $V × M$ 행렬이며, 투사층에서 출력층사이의 가중치 $W'$ 는 $M × V$ 행렬임**
    - 여기서 $V$ 는 단어 집합의 크기를 의미
    - 즉, 위의 그림처럼 원-핫 벡터의 차원인 $V$ 가 7이고, $M$ 은 5라면 가중치 $W$ 는 $7 × 5$ 행렬이고, $W'$ 는 $5 × 7$ 행렬이 될 것.
    - 주의할 점은 이 두 행렬은 동일한 행렬을 전치(transpose)한 것이 아니라, 서로 다른 행렬이라는 점으로 인공 신경망의 훈련 전에 이 가중치 행렬 $W$ 와 $W'$ 는 랜덤 값을 가짐

<br>




즉, Word2Vec은 총 2개의 가중치 행렬을 가지며 $W$ 는 단어 집합의 크기인 $V$ 행, 임베딩 행렬의 차원인 $M$ 열($M$ 은 하이퍼파라미터), $W'$는 그 반대의 크기를 가지며, $W$ 와 $W'$ 는 동일한 행렬을 전치(transpose)한 것이 아닙니다. 이때 **<span style="color:red">CBOW는 주변 단어로 중심 단어를 더 정확히 맞추기 위해 계속해서 이 $W$ 와 $W'$ 를 학습해가는 구조</span>** 입니다.


입력으로 들어오는 주변 단어의 원-핫 벡터와 가중치 $W$ 행렬의 곱이 어떻게 이루어지는지 보겠습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/0cc73c61-6d4b-412d-94b7-60248481776d">
</p>

<br>

위 그림에서는 각 주변 단어의 원-핫 벡터를 $x$ 로 표기하였습니다. 입력 벡터는 원-핫 벡터입니다. $i$ 번째 인덱스에 1이라는 값을 가지고 그 외의 0의 값을 가지는 입력 벡터와 가중치 $W$ 행렬의 곱은 사실 $W$ 행렬의 $i$ 번째 행을 그대로 읽어오는 것과(lookup) 동일합니다. 이 작업을 룩업 테이블(lookup table)이라고 합니다. **<span style="background-color: #fff5b1">즉, Projection layer에서는 입력된 원-핫 벡터와 가중치 행렬 $W$ 의 곱 입니다.</span>** 앞서 **<u>CBOW의 목적은 $W$ 와 $W'$ 를 잘 훈련시키는 것</u>** 이라고 언급한 적이 있는데, 그 이유가 **<span style="color:red">여기서 lookup해온 $W$ 의 각 행벡터가 Word2Vec 학습 후에는 각 단어의 $M$ 차원의 임베딩 벡터로 간주되기 때문</span>** 입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/bc6ea2e6-c699-44f4-9b14-df1fb90ac582">
</p>

<br>


이렇게 **주변 단어의 원-핫 벡터에 대해서 가중치 $W$ 가 곱해서 생겨진 결과 벡터들은 투사층에서 만나 이 벡터들의 평균인 벡터를 구하게 됩니다.** 만약 윈도우 크기 $n=2$ 라면, 입력 벡터의 총 개수는 $2n$ 이므로 중간 단어를 예측하기 위해서는 총 4개가 입력 벡터로 사용됩니다. 그렇기 때문에 평균을 구할 때는 4개의 결과 벡터에 대해서 평균을 구하게 됩니다. **<span style="color:red">즉, Projection layer에서의 최종 연산시 Projection layer에서 모든 embedding vector들은 평균값을 구하여 $M$ 차원의 벡터를 얻습니다.</span>**


**<span style="background-color: #fff5b1">투사층에서 벡터의 평균을 구하는 부분은 CBOW가 Skip-Gram과 다른 차이점</span>** 이기도 합니다. Skip-Gram은 입력이 중심 단어 하나이기 때문에 투사층에서 벡터의 평균을 구하지 않습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/0aaf20c3-242f-40bb-b73c-9b398b135c12">
</p>

<br>


이렇게 구해진 평균 벡터는 두번째 가중치 행렬 $W'$ 와 곱해져, 곱셈의 결과로 차원이 $V$ 벡터인 원-핫 벡터가 나옵니다. 이 벡터에 CBOW는 소프트맥스(softmax) 함수를 지나면서 벡터의 각 원소들의 값은 0과 1사이의 실수로, 총 합은 1이 됩니다. 다중 클래스 분류 문제를 위한 일종의 **<span style="color:red">스코어 벡터(score vector)</span>** 입니다. **<span style="color:red">스코어 벡터의 $j$ 번째 인덱스가 가진 0과 1사이의 값은 $j$ 번째 단어가 중심 단어일 확률을 나타냅니다.</span>**


그리고 이 스코어 벡터의 값은 레이블에 해당하는 벡터인 중심 단어 원-핫 벡터의 값에 가까워져야 합니다. 스코어 벡터를 $\hat{y}$ 라고 하고, 중심 단어의 원-핫 벡터를 $y$ 로 했을 때, 이 두 벡터값의 오차를 줄이기위해 CBOW는 손실 함수(loss function)로 크로스 엔트로피(cross-entropy) 함수를 사용합니다. 크로스 엔트로피 함수에 중심 단어인 원-핫 벡터와 스코어 벡터를 입력값으로 넣고, 이를 식으로 표현하면 다음과 같습니다. 아래의 식에서 $V$ 는 단어 집합의 크기입니다.

$$
cost(\hat{y}, y) = -\sum^V_{j=1} y_j log(\hat{y_j})
$$

<br>


정리하면, Projection layer의 M차원의 벡터는 가중치 행렬 $W'$ 와 곱하여 소프트맥스 함수를 통과한 뒤, 이 결과값은 CBoW의 예측값으로 실제 중심 단어의 원-핫 벡터와 loss를 구하고 역전파합니다. 역전파(Back Propagation)를 수행하면 $W$ 와 $W'$ 가 학습이 되는데, 학습이 다 되었다면 **$M$ 차원의 크기를 갖는 $W$ 의 행렬의 행을 각 단어의 임베딩 벡터로 사용** 하거나 **$W$ 와 $W'$ 행렬 두 가지 모두를 가지고 임베딩 벡터를 사용** 하기도 합니다.


<br>

CBOW를 다시한번 그림으로 정리하겠습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/97609551-c288-4390-8caf-e13bb5684053">
</p>

<br>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/47c64675-8691-4e51-a575-47b7b50d0e3f">
</p>

<br>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/6ff26485-da28-4785-bc45-66ed037090c1">
</p>

<br>






# 3. Skip-gram

**CBOW** 에서는 주변 단어를 통해 중심 단어를 예측했다면, **<span style="color:red">Skip-gram</span>** 은 **<span style="color:red">중심 단어로부터 주변 단어를 예측</span>** 합니다. 앞서 언급한 예문에 대해서 동일하게 윈도우 크기가 2일 때, 데이터셋은 다음과 같이 구성됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/38b9c538-db8a-4bbb-8890-04007ef7c598">
</p>

<br>


인공 신경망을 도식화해보면 아래와 같습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/8229af8d-8ece-467e-bab4-8742cd912bd7">
</p>

<br>



CBOW와 마찬가지로 Skip-gram은 입력층, 투사층, 출력층 3개의 층으로 구성된 신경망이며, 소프트맥스 함수를 지난 예측값(Prediction)과 실제값으로부터 오차(error)를 구합니다. 소프트맥스 함수를 지난 예측값(Prediction)과 실제값으로부터 오차(error)를 구하고, 이로부터 embedding table을 update 합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/2a053546-0c06-4c64-a9a1-595e9247aa2f">
</p>

<br>

**<u>중심 단어에 대해서 주변 단어를 예측하므로 투사층에서 벡터들의 평균을 구하는 과정은 없습니다.</u>** 여러 논문에서 성능 비교를 진행했을 때 CBOW보다 **<span style="background-color: #fff5b1">전반적으로 Skip-gram이 더 성능이 좋다고 알려</span>** 져 있습니다.


<br>




# 4. CBOW Vs Skip-gram

Word2Vec 방법 중 하나인 CBOW와 비교했을 때, **<span style="background-color: #fff5b1">Skip-gram이 성능이 우수하여 더욱 많이 사용</span>** 되고 있습니다. **<span style="color:red">왜냐하면, 모델 학습 시 Skip-gram이 CBOW에 비해 여러 문맥을 고려하기 때문</span>** 입니다. 앞선 예문을 다시 활용하여 살펴보겠습니다. 

$$
"The\ \ fat\ \ cat\ \ sat\ \ on\ \ the\ \ mat"
$$

<br>


CBOW와 Skip-gram 각각에 대해서 예측하는 단어마다 몇 번의 서로 다른 문맥을 고려했는지 확인해 보겠습니다.

<br>



## 4.1 CBOW

**CBOW** 는 **주변 단어로부터 오직 1개의 타겟 단어를 예측 및 학습** 합니다. Input과 Output 간의 관계를 나타내면 아래의 표와 같습니다.

| Input | Output |
| :----: | :----: |
| fat, cat | The |
| The, cat, sat | fat |
| The, fat, sat, on | cat |
| fat, cat, on, the | sat |
| cat, sat, the, table | on |
| sat, on, table | the |
| on, the | table |

<br>

즉, 단어를 예측 및 학습할 때 고려하는 문맥은 오직 아래의 표와 같이 1개뿐입니다. 예를 들어, 'sat'이라는 단어를 예측할 때는 'fat', 'cat', 'on', 'the'라는 주변 단어를 활용한 게 전부입니다.

| Word | Count |
| :----: | :----: |
| The | 1 |
| fat | 1 |
| cat | 1 |
| sat | 1 |
| on | 1 |
| the | 1 |
| table | 1 |

<br>



## 4.2 Skip-gram

**Skip-gram** 은 **타겟 단어를 바탕으로 여러 문맥 단어를 예측하고 학습** 합니다. Input과 Output 간의 관계를 나타내면 아래의 표와 같습니다.

| Input | Output |
| :----: | :----: |
| fat, cat | The |

| The | fat, cat |
| fat | The, cat, sat |
| cat | The, fat, sat, on |
| sat | fat, cat, on, the |
| on | cat, sat, the, table |
| the | sat, on, table |
| table | on, the |

<br>


위에서 예측 및 학습되는 출력값마다 고려하는 단어의 개수가 몇 개인지 계산해 보면 아래의 표와 같습니다. 이처럼, **<span style="background-color: #fff5b1">Skip-gram은 여러 문맥에 걸쳐 단어를 학습하기 때문</span>** 에, 대부분의 상황에서 **<u>Skip-gram이 CBOW보다 좋은 성능을 보입니다.</u>**

| Word | Count |
| :----: | :----: |
| The | 1 |
| fat | 1 |
| cat | 1 |
| sat | 1 |
| on | 1 |
| the | 1 |
| table | 1 |

<br>





# 5. NNLM Vs Word2Vec(CBOW)

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/388bffbd-f0ac-4ab8-8e6a-c8459d3785d9">
</p>

<br>


피드 포워드 신경망 언어 모델(NNLM)은 단어 벡터 간 유사도를 구할 수 있도록 워드 임베딩의 개념을 도입하였고, 워드 임베딩 자체에 집중하여 NNLM의 느린 학습 속도와 정확도를 개선하여 탄생한 것이 Word2Vec입니다.


NNLM과 Word2Vec의 차이를 비교해보겠습니다.

1. **예측하는 대상이 달라짐.**
    - NNLM
        - 다음 단어를 예측하는 언어 모델이 목적이므로 다음 단어를 예측함
    - Word2Vec(CBOW)
        - 워드 임베딩 자체가 목적이므로 다음 단어가 아닌 중심 단어를 예측하게 하여 학습함
        - 중심 단어를 예측하므로 NNLM이 예측 단어의 이전 단어들만을 참고하였던 것과는 달리, Word2Vec은 예측 단어의 전, 후 단어들을 모두 참고합니다.
2. **구조가 달라짐.**
    - NNLM
        - 입력층, 투사층, 출력층 전부 존재
        - 연산량 : $(n×m)+(n×m×h)+(h×V)$
    - Word2Vec(CBOW)
        - NNLM에 존재하던 활성화 함수가 있는 은닉층이 제거되었으므로, 이에 따라 투사층 다음에 바로 출력층으로 연결되는 구조임.
        - Word2Vec이 NNLM보다 학습 속도에서 강점을 가짐.
        - 연산량 : $(n×m)+(m×log(V))$
        - 추가적인 기법을 사용하면 출력층에서의 연산에서 $V$ 를 $log(V)$ 로 변경가능

<br>

Word2Vec이 NNLM보다 학습 속도에서 강점을 가지는 이유는 은닉층을 제거한 것뿐만 아니라 추가적으로 사용되는 기법들 덕분이기도 합니다. 대표적인 기법으로 **계층적 소프트맥스(hierarchical softmax)** 와 **네거티브 샘플링(negative sampling)** 이 있습니다.

<br>





# 6. 한계점

Skip-gram뿐만 아니라 CBOW 역시 출력층에서 소프트맥스 함수를 거쳐 단어 집합 크기의 벡터와 실제 참값인 원-핫 벡터와의 오차를 계산합니다. 이를 통해 가중치를 수정하고 모든 단어에 대한 임베딩 벡터 값을 업데이트 합니다.


그런데, 만약 단어 집합의 크기가 수만, 수십만 이상에 달하면 위와 같은 일련의 작업은 시간 소모가 큰 무거운 작업입니다. **<span style="background-color: #fff5b1">즉, Word2Vec의 학습 모델 자체가 무거워집니다.</span>** 이를 해결하는 방법으로 Hierarchical Softmax와 Negative Sampling 방법이 있습니다.





