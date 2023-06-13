---
layout: post
title: 언어 모델(Language Model, LM)
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---

1. 언어 모델(Language Model)이란?
    - 1.1 전체 단어 시퀀스의 확률
    - 1.2 이전 단어들이 주어졌을 때 다음 단어의 등장 확률
2. 언어 모델의 간단한 직관
3. 통계적 언어 모델(Statistical Language Model, SLM)
4. N-gram 언어 모델(N-gram Language Model)
5. Sparsity Problem

<br>
<br>



**<span style="color:red">언어 모델(Language Model, LM)</span>** 은 언어라는 현상을 모델링하고자 **<u>단어 시퀀스(문장)에 확률을 할당(assign)하는 모델</u>** 입니다.

언어 모델을 만드는 방법은 크게는 **<span style="color:red">통계를 이용한 방법</span>** 과 **<span style="color:red">인공 신경망을 이용한 방법</span>** 으로 구분할 수 있습니다. 최근에는 통계를 이용한 방법보다는 인공 신경망을 이용한 방법이 더 좋은 성능을 보여주고 있습니다.

<br>



# 1. 언어 모델(Language Model)이란?

**<span style="color:red">언어 모델(Language Model, LM)</span>** 은 **<span style="color:red">단어 시퀀스에 확률을 할당(assign) 하는 일을 하는 모델</span>** 입니다. 이를 조금 풀어서 쓰면, **<span style="background-color: #fff5b1">언어 모델은 가장 자연스러운 단어 시퀀스를 찾아내는 모델</span>** 로, 단어 시퀀스에 확률을 할당하는 이유는 Language Model을 통해 더 그럴듯한 문장을 선택할 수 있기 때문입니다.


가장 보편적으로 사용되는 방법은 언어 모델이 이전 단어들이 주어졌을 때 다음 단어를 예측하는 모델, 또는 주어진 양쪽의 단어들로부터 가운데 비어있는 단어를 예측하는 언어 모델 등이 있습니다. 자연어 처리에서 단어 시퀀스에 확률을 할당하는 일이 왜 필요한지 살펴보겠습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/adb5835b-d8bb-45e8-b1ae-4a2d3b09bd95">
</p>

<br>

**기계 번역(Machine Translation)** 에서 언어 모델은 두 문장을 비교하여 좌측의 문장의 확률이 더 높다고 판단하고, **음성 인식(Speech Recognition)** 에서 언어 모델은 두 문장을 비교하여 우측의 문장의 확률이 더 높다고 판단하며, **오타 교정(Spell Correction)** 에서는 언어 모델은 두 문장을 비교하여 좌측의 문장의 확률이 더 높다고 판단합니다. 즉 **<span style="background-color: #fff5b1">언어 모델은 단어 시퀀스에 확률을 할당하여, 확률을 통해 보다 적절한 문장을 판단</span>** 합니다.


Languae Model은 어떤 문장이 가장 그럴듯한지를 판단하며, 일반적으로 주어진 단어 시퀀스 다음에 어떤 단어가 등장할 지 예측하는 것이므로 이를 조건부 확률로 표현해보겠습니다.

<br>

## 1.1 전체 단어 시퀀스의 확률

하나의 단어를 $𝑤$, 단어 시퀀스를 대문자 $𝑊$ 라고 한다면, $𝑛$ 개의 단어가 등장하는 단어 시퀀스 $𝑊$ 의 확률은 다음과 같습니다. 즉 $𝑛$ 개의 단어가 동시에 등장할 확률 입니다.

$$
𝑃(𝑊) = 𝑃(𝑤_1,𝑤_2,𝑤_3,𝑤_4,𝑤_5,...,𝑤_𝑛)
$$

<br>

## 1.2 이전 단어들이 주어졌을 때 다음 단어의 등장 확률

다음 단어 등장 확률을 식으로 표현해보겠습니다. $𝑛‐1$ 개의 단어가 나열된 상태에서 $𝑛$ 번째 단어의 확률은 다음과 같습니다.

$$
𝑃(𝑤_𝑛|𝑤_1,...,𝑤_{𝑛−1})
$$

<br>

다섯번째 단어의 등장 확률은 다음과 같이 표현할 수 있습니다.

$$
𝑃 (𝑤_5|𝑤_1, 𝑤_2, 𝑤_3, 𝑤_4)
$$

<br>

전체 단어 시퀀스 $𝑊$ 의 확률은 모든 단어가 예측되고 나서야 알 수 있으므로, 이를 일반화하면 전체 단어 시퀀스의 확률은 다음과 같습니다.

$$
𝑃(𝑊) = 𝑃(𝑤_1,𝑤_2,𝑤_3,𝑤_4,𝑤_5,...𝑤_𝑛) = ∏^n_{i=1}𝑃(𝑤_𝑖|𝑤_1,...,𝑤_{𝑖−1})
$$

<br>


문장 **'its water is so transparent'** 의 확률 $𝑃 (its\ water\ is\ so\ transparent)$ 를 식으로 표현해봅시다.

**각 단어는 문맥이라는 관계로 인해 이전 단어의 영향을 받아 나온 단어** 입니다. 그리고 **모든 단어로부터 하나의 문장이 완성** 됩니다. 그렇기 때문에 문장의 확률을 구하고자 조건부 확률을 사용하겠습니다. 앞서 언급한 조건부 확률의 일반화 식을 문장의 확률 관점에서 다시 적어보면 **<span style="color:red">문장의 확률</span>** 은 **<u>각 단어들이 이전 단어가 주어졌을 때 다음 단어로 등장할 확률의 곱으로 구성</u>** 됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/ce5ea39d-cd70-4efe-a50b-0a819b6b6db6">
</p>

<br>

문장의 확률을 구하기 위해서 다음 단어에 대한 예측 확률들을 곱합니다.

<br>





# 2. 언어 모델의 간단한 직관

**비행기를 타려고 공항에 갔는데 지각을 하는 바람에 비행기를 [?]** 라는 문장이 있습니다. **'비행기를'** 다음에 어떤 단어가 오게 될지 사람은 쉽게 **'놓쳤다'** 라고 예상할 수 있습니다. 우리 지식에 기반하여 나올 수 있는 여러 단어들을 후보에 놓고 놓쳤다는 단어가 나올 확률이 가장 높다고 판단하였기 때문입니다.

그렇다면 기계에게 위 문장을 주고, **'비행기를'** 다음에 나올 단어를 예측해보라고 한다면 과연 어떻게 최대한 정확히 예측할 수 있을까요? 기계도 비슷합니다. **앞에 어떤 단어들이 나왔는지 고려하여 후보가 될 수 있는 여러 단어들에 대해서 확률을 예측해보고 가장 높은 확률을 가진 단어를 선택** 합니다.


<br>




# 3. 통계적 언어 모델(Statistical Language Model, SLM)


문장의 확률을 구하기 위해서 다음 단어에 대한 예측 확률을 모두 곱한다는 것은 알았습니다. **<u>언어 모델의 전통적인 접근 방법</u>** 인 **<span style="color:red">통계적 언어 모델(Statistical Language Model, SLM)</span>** 은 **<span style="color:red">카운트에 기반하여 이전 단어로부터 다음 단어에 대한 확률을 계산</span>** 합니다. 즉, **<span style="background-color: #fff5b1">통계적 방법중 카운트 기반 접근의 핵심 아이디어는 "훈련 데이터에서 단순 카운트하고 나누는 것은 어떨까?"</span>** 입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/91d934d7-c465-4a05-b0e3-5595b8443579">
</p>

<br>

확률은 위와 같습니다. 예를 들어 기계가 학습한 코퍼스 데이터에서 **its water is so transparent that** 가 100번 등장했는데 그 다음에 **the** 가 등장한 경우는 30번이라고 합시다. 이 경우 $𝑃(the|its\ water\ is\ so\ transparent\ that)$ 는 30%입니다.

하지만 **<span style="background-color: #fff5b1">훈련 데이터가 정말 방대하지 않은 이상 제대로 카운트할 수 있는 경우가 거의 없습니다.</span>**

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/b8477c6e-3d90-474b-b58c-8994209866e5">
</p>

<br>



**<span style="color:red">SLM의 한계는 훈련 코퍼스에 확률을 계산하고 싶은 문장이나 단어가 없을 수 있다는 점</span>** 입니다. 확률을 계산하고 싶은 문장이 길어질수록 갖고있는 코퍼스에서 그 문장이 존재하지 않을 가능성이 높으며, 다시 말하면 **카운트할 수 없을 가능성이 높습니다.**


그런데 다음과 같이 참고하는 단어들을 줄이면 카운트를 할 수 있을 가능성을 높일 수 있습니다.

$$
𝑃(is|An\ adorable\ little\ boy)≈ 𝑃(is|boy)
$$

<br>

가령, **An adorable little boy** 가 나왔을 때 **is** 가 나올 확률을 그냥 **boy** 가 나왔을 때 **is** 가 나올 확률로 생각해본다면, 갖고있는 코퍼스에 **An adorable little boy is** 가 있을 가능성 보다는 **boy is** 라는 더 짧은 단어 시퀀스가 존재할 가능성이 더 높습니다. 조금 지나친 일반화로 느껴진다면 아래와 같이 **little boy** 가 나왔을 때 **is** 가 나올 확률로 생각하는 것도 대안입니다.

$$
𝑃(is|An\ adorable\ little\ boy)≈ 𝑃(is|little\ boy)
$$

<br>

즉, 앞에서는 **An adorable little boy** 가 나왔을 때 **is** 가 나올 확률을 구하기 위해서는 **An adorable little boy** 가 나온 횟수와 **An adorable little boy is** 가 나온 횟수를 카운트해야만 했지만, **<span style="color:red">이제는 단어의 확률을 구하고자 기준 단어의 앞 단어를 전부 포함해서 카운트하는 것이 아니라, 앞 단어 중 임의의 개수만 포함해서 카운트하여 근사하자는 것</span>** 입니다. 이렇게 하면 갖고 있는 코퍼스에서 해당 단어의 시퀀스를 카운트할 확률이 높아집니다.

<br>





# 4. N-gram 언어 모델(N-gram Language Model)

**<span style="color:red">n-gram 언어 모델</span>** 은 여전히 카운트에 기반한 통계적 접근을 사용하고 있으므로 **SLM의 일종** 입니다. 다만, 앞서 배운 언어 모델과는 달리 **<u>이전에 등장한 모든 단어를 고려하는 것이 아니라</u>** **<span style="color:red">일부 단어만 고려하는 접근 방법을 사용</span>** 합니다. 이때 **<u>일부 단어를 몇 개 보느냐를 결정</u>** 하는데 이것이 n-gram에서의 **<span style="color:red">n</span>** 이 가지는 의미입니다. 예를 들어서 문장 "An adorable little boy is spreading smiles" 이 있을 때, 각 $n$ 에 대해서 n-gram 을 전부 구해보면 다음과 같습니다.

- (1-grams)unigrams : an, adorable, little, boy, is, spreading, smiles
- (2-grams)bigrams : an adorable, adorable little, little boy, boy is, is spreading, spreading smiles
- (3-grams)trigrams : an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles
- 4-grams : an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

<br>

정리하면 **<span style="color:red">n-gram</span>** 은 **<span style="color:red">n개의 연속적인 단어 나열을 의미</span>** 하고 n은 사용자가 정하는 값이며, **<span style="color:red">이전 단어들이 주어졌을때 다음 단어의 등장 확률의 추정을 앞의 n-1개의 단어에만 의존</span>** 합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/793d5271-3056-4e84-add0-87929c4cc477">
</p>

<br>

하지만 **<span style="color:red">n-gram은 앞의 단어 몇 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음하지 못하는 경우가 생긴다는 점</span>** 입니다. 문장을 읽다 보면 앞 부분과 뒷부분의 문맥이 전혀 연결 안 되는 경우도 생길 수 있습니다. 예를 들어 $n=2$ 이면 **Bigram Language Model** 이라고 하며 오직 이전 단어 하나만 고려하여 카운트하여 확률 추정하게됩니다. 즉, n을 선택하는 **<span style="background-color: #fff5b1">n-gram trade-off 문제</span>** 가 있습니다.

- **n을 크게 선택**
    - 실제 훈련 코퍼스에서 해당 n-gram을 카운트할 수 있는 확률은 적어지므로 희소 문제는 점점 심각해집니다.
    - 모델 사이즈가 커진다는 문제점도 있습니다. 기본적으로 코퍼스의 모든 n-gram에 대해서 카운트를 해야 하기 때문.
- **n을 작게 선택**
    - 훈련 코퍼스에서 카운트는 잘 되겠지만 근사의 정확도는 현실의 확률분포와 멀어집니다.
    
<br>



결론만 말하자면 n이 작으면 얼토당토 않는 문장이 되버리지만 n이 크면 카운트하기가 어려워 지므로 적절한 n을 선택해야하며, **<span style="color:red">전체 문장을 고려한 언어 모델보다는 정확도가 떨어질 수밖에 없습니다.</span>**

<br>





# 5. Sparsity Problem

**<span style="color:red">언어 모델</span>** 은 **<u>실생활에서 사용되는 언어의 확률 분포를 근사 모델링</u>** 합니다. 실제로 정확하게 알아볼 방법은 없겠지만 현실에서도 **An adorable little boy** 가 나왔을 때 **is** 가 나올 확률이라는 것이 존재합니다. **<span style="color:red">기계에게 많은 코퍼스를 훈련시켜서 언어 모델을 통해 현실에서의 확률 분포를 근사하는 것이 언어 모델의 목표</span>** 입니다. 그런데 카운트 기반 또는 N-gram 으로 접근하려고 한다면 갖고있는 코퍼스(corpus). 즉, 다시 말해 **<span style="background-color: #fff5b1">통계적 언어 모델(Statistical Language Model, SLM)에서 기계가 훈련하는 데이터는 정말 방대한 양이 필요</span>** 합니다.


<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/60c50ed4-2797-4da6-a6c5-4cbe52466ed2">
</p>

<br>



Sparsity Problem 문제를 완화하는 방법으로 스무딩이나 백오프와 같은 여러가지 일반화(generalization) 기법이 존재하지만, 희소 문제에 대한 근본적인 해결책은 되지 못하였습니다. 결국 이러한 한계로 인해 언어 모델의 트렌드는 통계적 언어 모델에서 **<span style="color:red">인공 신경망 언어 모델(NNLM)</span>** 로 넘어가게 됩니다.






