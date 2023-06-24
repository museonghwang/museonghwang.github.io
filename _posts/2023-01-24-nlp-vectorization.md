---
layout: post
title: 단어의 표현 방법(Vectorization)
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---



텍스트를 컴퓨터가 이해하고, 효율적으로 처리하게 하기 위해서는 컴퓨터가 이해할 수 있도록 텍스트를 적절히 숫자로 변환해야 합니다. 자연어 처리에서 텍스트를 수치적으로 표현하는 방법으로는 여러가지 방법이 있습니다. 왜냐하면 단어를 표현하는 방법에 따라서 자연어 처리의 성능이 크게 달라지기 때문입니다.

텍스트를 수치적으로 표현하면, 통계적인 접근 방법을 통해 여러 문서로 이루어진 텍스트 데이터가 있을 때 어떤 단어가 특정 문서 내에서 얼마나 중요한 것인지를 나타내거나, 문서의 핵심어 추출, 검색 엔진에서 검색 결과의 순위 결정, 문서들 간의 유사도를 구하는 등의 용도로 사용할 수 있습니다.

<br>
<br>





# 1. 단어의 표현 방법

**<span style="background-color: #fff5b1">단어의 표현 방법</span>** 은 크게 두 가지가 있습니다.

1. **<span style="color:red">국소 표현(Local Representation) 방법</span>**
    - **이산 표현 (Discrete Representation)**
    - 국소 표현 방법은 **<u>해당 단어 그 자체만 보고, 각 단어에 특정값(숫자)을 맵핑하여 단어를 표현하는 방법</u>**
    - 예를 들어 puppy(강아지), cute(귀여운), lovely(사랑스러운)라는 단어가 있을 때, 각 단어에 1번, 2번, 3번 같이 숫자를 맵핑(mapping)하여 부여한다면 이는 국소 표현 방법에 해당
2. **<span style="color:red">분산 표현(Distributed Representation) 방법</span>**
    - **연속 표현 (Continuous Representation)**
    - 분산 표현 방법은 **<u>단어를 표현하기위해 주변을 참고하여 단어를 표현하는 방법</u>**
    - 예를 들어 puppy(강아지), cute(귀여운), lovely(사랑스러운)라는 단어가 있을 때, puppy(강아지)라는 단어 근처에는 주로 cute(귀여운), lovely(사랑스러운)이라는 단어가 자주 등장하므로, puppy라는 단어는 cute, lovely한 느낌이다로 단어를 정의한다면 이는 분산 표현 방법에 해당

<br>


이렇게 되면 이 두 방법의 차이는 **국소 표현 방법** 은 단어의 의미나 뉘앙스를 표현할 수 없지만, **<u>분산 표현 방법은 단어의 뉘앙스를 표현할 수 있게 됩니다.</u>**


<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/cc83c15e-c14f-4306-ae6e-9db78f001fea">
</p>

<br>

**One-hot Vector**, **N-gram**, **Bag of Words** 는 **<span style="background-color: #fff5b1">국소 표현(Local Representation)</span>** 에 속합니다. 특히 **Bag of Words**는 단어의 빈도수를 카운트(Count)하여 단어를 수치화하는 단어 표현 방법이며, 그의 확장인 **DTM**, 그리고 이러한 빈도수 기반 단어 표현에 단어의 중요도에 따른 가중치를 줄 수 있는 **TF-IDF** 등이 있습니다.

**워드 임베딩** 은 **<span style="background-color: #fff5b1">연속 표현(Continuous Representation)</span>** 에 속하면서, 예측(prediction)을 기반으로 단어의 뉘앙스를 표현하는 **워드투벡터(Word2Vec)** 와 그의 확장인 **패스트텍스트(FastText)** 가 있고, 예측과 카운트라는 두 가지 방법이 모두 사용된 **글로브(GloVe)** 방법 등이 존재합니다.

<br>





# 2. 국소 표현(Local Representation)

단어를 표현하는 가장 기본적인 방법은 **<span style="color:red">원-핫 인코딩(one-hot encoding)</span>** 방식 이며, **대표적인 Local Representation 방법** 입니다. 원-핫 인코딩은 **<u>전체 단어 집합의 크기(중복은 카운트하지 않은 단어들의 집합)를 벡터의 차원으로 가지며, 각 단어에 고유한 정수 인덱스를 부여하고, 해당 인덱스의 원소는 1, 나머지 원소는 0을 가지는 벡터로 만듭니다.</u>**

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/abe3f83d-46e6-400b-a784-ca025839d44d">
</p>

<br>


- **장점**
    - 방법 자체가 매우 간단하고 이해하기 쉬움
- **단점**
    - 단어 벡터가 단어의 의미나 특성을 전혀 표현할 수 없으므로, **<span style="color:red">단어 벡터 간 유의미한 유사도를 구할 수 없다는 한계가 존재</span>**
    - **<span style="color:red">단어의 벡터의 크기가 매우 크고, 값이 희소(sparse)</span>**. 즉 실제 사용하는 값은 1이 되는 값 하나 뿐이므로 매우 비효율적


<br>

이 문제를 해결하기 위해 **<span style="background-color: #fff5b1">벡터의 크기가 작으면서도 벡터의 단어의 의미를 표현할 수 있는 방법</span>** 들이 제안되었습니다. 이러한 방법들은 **<span style="color:red">분포 가설(Distributed hypothesis)을 기반</span>** 으로 합니다.

<br>





# 3. 분산 표현(Distributed Representation)

분포 가설이란 "같은 문맥의 단어, 즉 비슷한 위치에 나오는 단어는 비슷한 의미를 가진다."라는 개념으로, 분포 가설을 기반으로 하는 벡터의 크기가 작으면서도 단어의 의미를 표현할 수 있는 방법은 크게 두 가지 방법으로 나뉩니다.

- **카운트 기반(count-base) 방법**
    - **특정 문맥 안에서 단어들이 동시에 등장하는 횟수를 직접 세는 방법**
    - 기본적으로 동시 출현 행렬(Co-occurence Matrix)을 만들고 그 행렬들을 변형하는 방식을 사용
- **예측(predictive) 방법**
    - **신경망 등을 통해 문맥 안의 단어들을 예측하는 방법**
    - 신경망 구조 혹은 어떠한 모델을 사용해 특정 문맥에서 어떤 단어가 나올지를 예측하면서 단어를 벡터로 만드는 방식을 사용
        - Word2vec
        - NNLM(Nenural Netwrok Language Model)
        - RNNLM(Recurrent Neural Netwrok Language Model)


<br>




# 4. 워드 임베딩(Word Embedding)

**<span style="background-color: #fff5b1">워드 임베딩(Word Embedding)(벡터 공간(Vector)으로 + 끼워넣는다(embed))은 단어를 벡터로 표현하는 방법</span>** 으로, **연속 표현(Continuous Representation)** 에 속하며, **<span style="background-color: #fff5b1">단어를 밀집 표현으로 변환</span>** 합니다. 

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/92afc9a3-f741-43a9-bc02-b2f45de6f453">
</p>

<br>

희소 표현, 밀집 표현, 그리고 워드 임베딩에 대한 용어 정리를 하겠습니다.

<br>




## 4.1 희소 표현(Sparse Representation)

데이터를 벡터 또는 행렬을 기반으로 수치화하여 표현할 때 극히 일부의 인덱스만 특정 값으로 표현하고, 대부분의 나머지 인덱스는 의미 없는 값으로 표현하는 방법, 즉 **<u>값의 대부분이 0으로 표현되는 방법</u>** 을 **<span style="color:red">희소 표현(sparse representation)</span>** 이라고 합니다. 원-핫 벡터는 전체 단어 집합의 크기를 갖는 벡터에서 표현할 단어의 인덱스만 1로 나타내고, 나머지는 모두 0으로 표현하는 기법으로 **<span style="color:red">희소 벡터(sparse vector)</span>** 입니다.


**<span style="color:red">희소 벡터의 문제점은 단어의 개수가 늘어나면 벡터의 차원이 한없이 커져 고차원의 벡터가 되어 불필요한 벡터 공간의 낭비를 유발한다는 점</span>** 입니다. 예를 들어 단어가 10,000개 있고 인덱스가 0부터 시작하면서 'you'이라는 단어의 인덱스가 3였다면 원 핫 벡터는 다음과 같이 표현되며, 이때 1 뒤의 0의 수는 9996개 입니다.

$$
you = [0, 0, 1, 0, 0, ..., 0]
$$

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/f7a8314d-0761-4d4c-9c14-1f2d1d82be34">
</p>

<br>

이러한 **희소 벡터 표현은 공간적 낭비를 유발** 합니다. 더불어, 카운트 기반의 단어 표현기법 중 하나인 단어문서행렬(Document Term Matrix, DTM) 역시 희소표현을 사용합니다. 이러한 **<span style="color:red">원-핫 인코딩, 카운트 기반 단어 표현기법은 단어의 의미를 반영하지 못합니다.</span>** 왜냐하면 단어의 의미는 고려하지 않고 단순히 어떤 단어가 문서에서 몇 번 등장했는지만 고려하기 때문입니다.

<br>




# 4.2 밀집 표현(Dense Representation)

**<span style="color:red">밀집 표현(dense representation)</span>** 은 희소 표현과 반대되는 단어표현 방법으로써 **<u>텍스트를 실숫값으로 구성하고, 벡터의 차원을 단어 집합의 크기로 상정하지 않고 사용자가 설정한 차원의 벡터로 표현합니다.</u>** 희소 표현에서는 단어 집합의 크기만큼 벡터 또는 행렬의 차원이 결정되고, 표현할 단어만 1이 아닌 정수로, 나머지는 0으로 표현했습니다. 반면, **밀집 표현** 은 사용자가 임의로 설정한 차원에서, 단순히 0 또는 1의 값만으로 데이터를 표현하는 것이 아닌 실숫값으로 표현합니다.


앞서 활용한 예시에서, 벡터는 단어의 인덱스를 제외한 단어 집합의 크기만큼 0의 값을 가졌습니다. 사용자가 벡터의 크기를 64로 설정했다면, 밀집 표현으로 해당 단어를 표현하면 다음과 같이 64차원을 갖는 벡터 형태로 나타낼 수 있습니다.

$$
Apple = [0.4, 0.7, ..., -0.1]
$$

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/b04a4c18-6df0-41fd-9c32-4964afde6c92">
</p>

<br>

이 경우 벡터의 차원의 크기를 줄이고(조밀해짐) 실숫값을 밀집하여 표현했다고 해서 **<span style="color:red">밀집 벡터(Dense Vector)</span>** 라고 부릅니다.

<br>





# 4.3 워드 임베딩(Word Embedding)

**<span style="color:red">워드 임베딩(Word Embedding)</span>** 이란 단어를 밀집 벡터(dense vector)의 형태로 표현하는 기법입니다. 여기서 임베딩을 통해 얻은 결과인 밀집 터를 **<span style="color:red">임베딩 벡터(Embedding Vector)</span>** 라고 부릅니다. 워드 임베딩 방법론으로는 LSA, Word2Vec, Glove, FastText, ELMO 등이 있습니다.

| - | 원-핫 벡터 | 임베딩 벡터 |
| :-- | :------ | :---------- |
| 차원 | 고차원(단어 집합의 크기) | 저차원 |
| 다른 표현	| 희소 벡터의 일종 | 밀집 벡터의 일종 |
| 표현 방법 | 수동 | 훈련 데이터로부터 학습함 |
| 값의 타입 | 1과 0 | 실수 |

<br>





# 5. Vectorization

결국 텍스트를 컴퓨터가 이해하고, 효율적으로 처리하게 하기 위해서는 컴퓨터가 이해할 수 있도록 텍스트를 적절히 숫자로 변환해야 하며, 이를 **<span style="color:red">Vectorization</span>** 이라 지칭할 수 있습니다. 상황별 Vectorization을 정리하겠습니다.


1. **벡터화에 신경망을 사용하지 않을 경우**
    - 단어에 대한 벡터 표현 방법 : 원-핫 인코딩
    - 문서에 대한 벡터 표현 방법 : Document Term Matrix(DTM), TF-IDF
2. **벡터화에 신경망을 사용하는 경우 (2008 ~ 2018)**
    - 단어에 대한 벡터 표현 방법 : 워드 임베딩(Word2Vec, GloVe, FastText, Embedding layer)
    - 문서에 대한 벡터 표현 방법 : Doc2Vec, Sent2Vec
3. **문맥을 고려한 벡터 표현 방법 (2018 - present)**
    - ELMo, BERT
    - Pretrained Language Model의 시대




