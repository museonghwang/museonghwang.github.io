---
layout: post
title: 
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---



# 1. 희소 표현(Sparse Representation)의 한계

데이터를 벡터 또는 행렬을 기반으로 수치화하여 표현할 때 극히 일부의 인덱스만 특정 값으로 표현하고, 대부분의 나머지 인덱스는 의미 없는 값으로 표현하는 방법, 즉 **<u>값의 대부분이 0으로 표현되는 방법</u>** 을 **<span style="color:red">희소 표현(sparse representation)</span>** 이라고 합니다. 대표적인 희소 표현인 원-핫 벡터는 전체 단어 집합의 크기를 갖는 벡터에서 표현할 단어의 인덱스만 1로 나타내고, 나머지는 모두 0으로 표현하는 기법으로 **<span style="color:red">희소 벡터(sparse vector)</span>** 입니다.


**<span style="color:red">희소 벡터의 문제점</span>** 은 단어의 개수가 늘어나면 벡터의 차원이 한없이 커져 고차원의 벡터가 되어 **<span style="color:red">불필요한 벡터 공간의 낭비를 유발한다는 점</span>** 입니다. 또한 **<span style="color:red">원-핫 인코딩, 카운트 기반 단어 표현기법은 단어의 의미를 반영하지 못합니다.</span>** 왜냐하면 단어의 의미는 고려하지 않고 단순히 어떤 단어가 문서에서 몇 번 등장했는지만 고려하기 때문입니다.

<br>

즉, **<span style="background-color: #fff5b1">희소 표현(Sparse Representation)</span>** 은 **<span style="background-color: #fff5b1">고차원에 각 차원이 분리된 표현 방법</span>** 입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/abe3f83d-46e6-400b-a784-ca025839d44d">
</p>

<br>



이 문제를 해결하기 위해 **<span style="background-color: #fff5b1">벡터의 크기가 작으면서도 벡터의 단어의 의미를 표현할 수 있는 방법</span>** 들이 제안되었습니다. 이러한 방법들은 **<span style="color:red">분포 가설(Distributed hypothesis)을 기반</span>** 으로 합니다.

<br>





# 2. 분산 표현(Distributed Representation)의 등장

**<span style="color:red">분산 표현(Distributed Representation)</span>** 이란 **<u>분포 가설(Distibutional Hypothesis) 가정 하</u>** 에 **<span style="color:red">저차원에 단어 의미를 분산하여 표현하는 기법</span>** 입니다. **<span style="background-color: #fff5b1">분포가설</span>** 은 **<span style="background-color: #fff5b1">"비슷한 문맥에 등장한 단어는 비슷한 의미를 갖는다"라는 가정</span>** 입니다. 예를 들어 강아지란 단어는 귀엽다, 예쁘다, 애교 등의 단어가 주로 함께 등장하는데 분포 가설에 따라서 해당 내용을 가진 텍스트의 단어들을 벡터화한다면 해당 단어 벡터들은 유사한 벡터값을 가집니다. **이렇게 표현된 벡터들은 원-핫 벡터처럼 벡터의 차원이 단어 집합(vocabulary)의 크기일 필요가 없으므로**, **<span style="color:red">벡터의 차원이 상대적으로 저차원으로 줄어듭니다.</span>**


정리하면, **<span style="color:red">분산 표현은 분포 가설을 이용하여 텍스트를 학습하고, 단어의 의미를 벡터의 여러 차원에 분산하여 표현</span>** 합니다.



<br>


여기서 **단어를 벡터화하는 작업** 을 **<span style="color:red">워드 임베딩(Word Embedding)</span>** 이라고 부르며 **임베딩 된 벡터** 를 **<span style="color:red">임베딩 벡터(Embedding Vector)</span>** 라고 부릅니다. 워드 임베딩은 단어를 밀집 표현으로 변환합니다. **밀집 표현(dense representation)** 은 희소 표현과 반대되는 단어표현 방법으로써 텍스트를 실숫값으로 구성하고, 벡터의 차원을 단어 집합의 크기로 상정하지 않고 사용자가 설정한 차원의 벡터로 표현합니다. 

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/92afc9a3-f741-43a9-bc02-b2f45de6f453">
</p>

<br>


요약하면 **희소 표현(Sparse Representation)** 이 **<u>고차원에 각 차원이 분리된 표현 방법</u>** 이었다면, **분산 표현(Distributed Representation)** 은 **<u>저차원에 단어의 의미를 여러 차원에다가 분산 하여 표현</u>** 합니다. 이런 표현 방법을 사용하면 **<span style="color:red">단어 벡터 간 유의미한 유사도를 계산할 수 있습니다.</span>** 이를 위한 대표적인 학습 방법이 **Word2Vec** 입니다.

<br>





# 3. Word2Vec 개념

**<span style="color:red">Word2Vec</span>** 는 Word to Vector라는 이름에서 알 수 있듯이 **<u>단어(Word)를 컴퓨터가 이해할 수 있도록 수치화된 벡터(Vector)로 표현하는 기법 중 하나</u>** 입니다. 구체적으로는 **분산 표현(Distributed Representation) 기반의 워드 임베딩(Word Embedding)** 기법 중 하나입니다.


즉, **<span style="color:red">Word2Vec</span>** 는 **<span style="background-color: #fff5b1">단어의 의미를 반영한 임베딩 벡터를 만드는 대표적인 방법</span>** 으로, 벡터가 된 단어들은 이제 **<u>수치화된 벡터(Vector)이므로 서로 연산 또한 가능</u>** 하므로 **<span style="background-color: #fff5b1">단어 벡터 간 유의미한 유사도를 계산할 수 있습니다.</span>**

<br>



Word2Vec가 어떤 일을 할 수 있는지 확인해보겠습니다. 아래 사이트는 한국어 단어에 대해서 벡터 연산을 해볼 수 있는 사이트로, 단어들(실제로는 Word2Vec 벡터)로 더하기, 빼기 연산을 할 수 있습니다.

- [한국어 Word2Vec 연산](http://w.elnn.kr/search/)

<br>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/b2e6e2b5-1b2c-4e84-bc93-04f9889ce4b3">
</p>

<br>


이런 연산이 가능한 이유는 **<span style="color:red">각 단어 벡터가 단어 벡터 간 유사도를 반영한 값을 가지고 있기 때문</span>** 입니다.

<br>



**Word2Vec은 기본적으로 NNLM을 개선한 모델** 입니다. **<u>이전 단어들로부터 다음 단어를 예측하는 목표는 버리고, 임베딩 그 자체에만 집중</u>** 했습니다. Word2Vec는 학습방식에 따라 크게 2가지로 나눌 수 있습니다. CBOW(Continuous Bag of Words)와 Skip-gram 두 가지 방식이 있습니다.

**<span style="color:red">CBOW</span>** 는 주변 단어(Context Word)로 중간에 있는 단어를 예측하는 방법입니다. 여기서 중간에 있는 단어를 중심 단어(Center Word) 또는 타겟 단어(Target Word)라고 부릅니다. 반대로, **<span style="color:red">Skip-gram</span>** 은 중심 단어를 바탕으로 주변 단어들을 예측하는 방법입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/558b2f30-3bae-4c7a-8a55-aecab410a2df">
</p>








