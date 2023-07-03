---
layout: post
title: 자모 단위의 한국어 FastText 이해와 실습
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





1. FastText Review
2. FastText Pre-training Review
3. FastText Summaray
4. 한국어 FastText
    - 4.1 음절 단위
    - 4.2 자모 단위
5. 자모 단위 한국어 FastText 실습
    - 5.1 필요 패키지 설치
    - 5.2 네이버 쇼핑 리뷰 데이터 로드
    - 5.3 HGTK 튜토리얼
    - 5.4 자모 단위 토큰화(전처리)
    - 5.5 FastText 학습하기

<br>
<br>




# 1. FastText Review

**<span style="color:red">FastText</span>** 는 **<u>Word2Vec의 개량 알고리즘으로 Subword를 고려한 알고리즘</u>** 입니다. **Word2Vec** 이후에 나온 것이기 때문에 메커니즘 자체는 **Word2Vec** 의 확장이라고 볼 수 있습니다.



예를 들어 **eat** 과 **eating**이 있다고 가정해보겠습니다. 훈련 데이터에서 **eat** 는 충분히 많이 등장해서, 학습이 충분히 잘 되었지만 **eating**은 잘 등장하지 않아서 제대로 된 임베딩 값을 얻지 못한다고 가정해보겠습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/c9fa85df-3073-4bf9-9f40-fa780593d1ed">
</p>

<br>


즉, **Word2Vec의 문제점** 으로 **<u>OOV(Out-of-Vocabulary) 문제와, 하나의 단어에 고유한 벡터를 할당하므로 단어의 형태학적 특징을 반영할 수 없다는 문제</u>** 가 있습니다. 이때 **<span style="color:red">FastText</span>** 의 아이디어는 **eat** 이라는 **<span style="background-color: #fff5b1">공통적인 내부 단어를 가지고 있는데 이를 활용할 수는 없을까</span>** 라는 의문에서 시작됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/1dfb0c31-cd8a-4502-82f6-6b1876d5b21b">
</p>

<br>




# 2. FastText Pre-training Review

**<span style="color:red">FastText는 단어를 Character 단위의 n-gram으로 간주하며, $n$ 을 몇으로 하느냐에 따라서 단어가 얼마나 분리되는지가 결정</span>** 됩니다. 단어 'eating'을 예를 들어보겠습니다.


단어 eating에 시작과 끝을 의미하는 '<'와 '>'를 추가합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/78c497fe-f64d-4988-b286-933f931d245a">
</p>

<br>



n-gram을 기반으로 단어를 분리합니다. 이때 n = 3.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/7671b493-af2e-4ff8-8301-1b6fe100ab40">
</p>

<br>


실제로는, 주로 n은 범위로 설정해줍니다. 이때 n = 3 ~ 6.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/b8944178-89fc-47e6-85df-a51ce99f618c">
</p>

<br>



훈련 데이터를 N-gram의 셋으로 구성하였다면, **<u>훈련 방법 자체는 SGNS(Skip-gram with Negative Sampleing)와 동일</u>** 합니다. 단**<span style="color:red">, Word가 아니라 subwords들이 최종 학습 목표</span>** 이며, **<span style="color:red">이들의 합을 Word의 vector로 간주</span>**합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/d4f91688-987b-4745-ad0c-b0de92264d1f">
</p>

<br>




의문점이 있을 수 있는데, **<span style="color:red">단어에 <, >를 해주는 이유</span>** 를 살펴보면 다음과 같습니다.

단어 양끝에 <, >를 해주지 않으면 실제로 독립적인 단어와 특정 단어의 n-gram인 경우를 구분하기 어렵습니다. 가령, where의 n-gram 중 하나인 her도 존재하지만 독립적인 단어 her 또한 Vocabulary에 존재할 수 있습니다. 이 때, 독립적인 단어 her는 **'\<her\>'** 가 되므로서 where 내의 her와 구분할 수 있습니다.

<br>



이제 FastText의 훈련 과정을 이해해보겠습니다. 여기서는 SGNS(Skip-gram with Negative Sampleing)을 사용합니다. **<u>현재의 목표는 중심 단어 eating으로부터 주변 단어 am과 food를 예측하는 것</u>** 입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/c368d585-739c-4bc6-ab37-6c6cffad395b">
</p>

<br>



앞서 언급하였듯이 단어 eating은 아래와 같이 n-gram들의 합으로 나타냅니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/31e66fcd-740f-4304-9b90-b616969fce51">
</p>

<br>


**<span style="color:red">우리가 해야하는 것은 eating으로부터 am과 food를 예측하는 것</span>** 입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/31a52bdd-2904-4bd6-9bb4-152abb4cfa3d">
</p>

<br>





**<u>Negative Sampling이므로 실제 주변 단어가 아닌 단어들도 필요</u>** 합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/335a5079-651b-4712-a233-e7e1d4fce06f">
</p>

<br>




우리가 해야하는 것은 eating으로부터 am과 food를 예측하는 것이므로 **eating과 am 그리고 eating과 food의 내적값에 시그모이드 함수를 지난 값은 1이 되도록 학습** 하고, **eating과 paris 그리고 eating과 earth의 내적값에 시그모이드 함수를 지난 값은 0이 되도록 학습** 합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/94129028-96af-4e6c-8489-f6a8c26af7a5">
</p>

<br>


이런 방법은 Word2Vec에서는 얻을 수 없었던 강점을 가집니다. 예를 들어 단어 Orange에 대해서 FastText를 학습했다고 해보겠습니다. n의 범위는 2-5로 하겠습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/63837697-8ad2-42de-aff6-c668ad312d34">
</p>

<br>





그 후 Oranges라는 OOV 또는 희귀 단어가 등장했다고 해보겠습니다. Orange의 n-gram 벡터들을 이용하여 Oranges의 벡터값을 얻습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/f49103be-8e2d-4fb7-82f6-8abca27818fb">
</p>

<br>





또한 FastText는 오타에도 강건합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/c1fd07a9-bf14-49cc-98da-85f822d04449">
</p>

<br>






# 3. FastText Summaray

결과적으로 **Word2Vec** 는 학습 데이터에 존재하지 않는 단어. **<u>즉, 모르는 단어에 대해서는 임베딩 벡터가 존재하지 않기 때문에 단어의 유사도를 계산할 수 없습니다.</u>** **<span style="color:red">하지만 FastText는 유사한 단어를 계산해서 출력합니다.</span>** 정리하면 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, **<span style="background-color: #fff5b1">FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다. 내부 단어. 즉, 서브워드(subword)를 고려하여 학습</span>** 합니다.



**FastText** 는 다음과 같은 **강점** 을 가집니다.

- **<span style="color:red">모르는 단어(Out Of Vocabulary, OOV)에 대한 대응</span>**
    - FastText의 인공 신경망을 학습한 후에는 데이터 셋의 모든 단어의 각 n-gram에 대해서 워드 임베딩이 됩니다. 만약 데이터 셋만 충분한다면 내부 단어(Subword)를 통해 모르는 단어(Out Of Vocabulary, OOV)에 대해서도 다른 단어와의 유사도를 계산할 수 있습니다.
    - 가령, FastText에서 birthplace(출생지)란 단어를 학습하지 않은 상태라고 했을때, 다른 단어에서 birth와 place라는 내부 단어가 있었다면, FastText는 birthplace의 벡터를 얻을 수 있습니다.
- **<span style="color:red">단어 집합 내 빈도 수가 적었던 단어(Rare Word)에 대한 대응</span>**
    - Word2Vec의 경우에는 등장 빈도 수가 적은 단어(rare word)에 대해서는 임베딩의 정확도가 높지 않다는 단점이 있었는데, FastText의 경우, 만약 단어가 희귀 단어라도, 그 단어의 n-gram이 다른 단어의 n-gram과 겹치는 경우라면, Word2Vec과 비교하여 비교적 높은 임베딩 벡터값을 얻습니다.
- **<span style="color:red">단어 집합 내 노이즈가 많은 코퍼스에 대한 대응</span>**
    - Word2Vec에서는 오타가 섞인 단어는 임베딩이 제대로 되지 않지만, FastText는 이에 대해서도 일정 수준의 성능을 보입니다.
    - 예를 들어 단어 apple과 오타로 p를 한 번 더 입력한 appple의 경우에는 실제로 많은 개수의 동일한 n-gram을 가질 것입니다.



<br>


충분히 잘 학습된 FastText는 전체 Word가 아니라 Subword들의 유사도를 반영함을 확인할 수 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/3c970188-950d-4908-976c-4fe4955f66bc">
</p>

<br>






# 4. 한국어 FastText

한국어의 경우에도 OOV 문제를 해결하기 위해 FastText를 적용하고자 하는 시도들이 있었습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/e649c87e-4812-43bb-bd5c-6b449c30275c">
</p>

<br>


대표적으로 음절 단위와 자-모 단위가 있습니다.


<br>




## 4.1 음절 단위

예를 들어서 **<span style="color:red">음절 단위의 임베딩</span>** 의 경우에 n=3일때 '자연어처리'라는 단어에 대해 n-gram을 만들어보면 다음과 같습니다.
```
<자연, 자연어, 연어처, 어처리, 처리>
```

<br>




## 4.2 자모 단위

우선 **한국어는 다양한 용언 형태를 가지는데**, Word2Vec의 경우 다양한 용언 표현들이 서로 독립된 단어로 표현됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/40833cb1-35af-41aa-955c-7ac6fe9a7986">
</p>

<br>


한국어의 경우에는 이를 대응하기 위해 **<span style="color:red">한국어 FastText의 n-gram 단위</span>** 를 음절 단위가 아니라, **<span style="color:red">자모 단위(초성, 중성, 종성)</span>** 로 하기도 합니다. **<span style="background-color: #fff5b1">자모 단위로 가게 되면 오타나 노이즈 측면에서 더 강한 임베딩을 기대</span>** 해볼 수 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/6c99a30d-f71c-4eb6-89d8-cfa30933a449">
</p>

<br>




**FastText는 하나의 단어에 대하여 벡터를 직접 학습하지 않습니다.** **<span style="color:red">대신에 subwords의 벡터들을 바탕으로 word의 벡터를 추정합니다.</span>** 좀 더 자세히 말하자면 v(어디야)는 직접 학습되지 않습니다. 하지만 v(어디야)는 [v(어디), v(디야)]를 이용하여 추정됩니다. 즉 '어디야'라는 단어는 '어디', '디야'라는 **<span style="background-color: #fff5b1">subwords를 이용하여 추정</span>** 되는 것입니다.


그런데, 이 경우에는 **오탈자에 민감** 하게 됩니다. '어딛야' 같은 경우에는 [v(어딛), v(딛야)]를 이용하기 때문에 [v(어디), v(디야)]와 겹치는 subwords가 없어서 비슷한 단어로 인식되기가 어렵습니다. **<u>한국어의 오탈자는 초/중/종성에서 한군데 정도가 틀리기 때문에 자음/모음을 풀어서 FastText를 학습하는게 좋습니다.</u>** 즉 어디야는 **'ㅇㅓ_ㄷㅣ_ㅇㅑ_'** 로 표현됩니다. 종성이 비어있을 경우에는 **'_'** 으로 표시하였습니다. FastText가 word를 학습할 때 띄어쓰기를 기준으로 나누기 때문입니다.

- [참고자료](https://lovit.github.io/nlp/representation/2018/10/22/fasttext_subword/)

<br>




음절 단위의 단어를 다시 예시로 들면, '자연어처리'라는 단어에 대해서 초성, 중성, 종성을 분리하고, 만약, 종성이 존재하지 않는다면 '_'라는 토큰을 사용한다고 가정했을때, '자연어처리'라는 단어는 아래와 같이 분리가 가능합니다.
```
ㅈㅏ_ㅇㅕㄴㅇㅓ_ㅊㅓ_ㄹㅣ_
```

<br>

그리고 분리된 결과에 대해서 n=3일 때, n-gram을 적용하여, 임베딩을 한다면 다음과 같습니다.
```
<ㅈㅏ, ㅈㅏ_, ㅏ_ㅇ, ... 중략>
```

<br>







# 5. 자모 단위 한국어 FastText 실습

네이버 쇼핑 리뷰 데이터를 이용하여 자모 단위 FastText를 학습해보겠습니다.

<br>





## 5.1 필요 패키지 설치

여기서는 형태소 분석기 **Mecab** 을 사용합니다. 본 실습은 **Mecab** 을 편하게 사용하기 위해서 구글의 **Colab** 을 사용하였습니다. 참고로 **Colab** 에서 실습하는 경우가 아니라면 아래의 방법으로 **Colab** 이 설치되지 않습니다.
```py
!pip install konlpy
!pip install mecab-python
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```


<br>

한글 자모 단위 처리 패키지인 **hgtk** 를 설치합니다.
```py
# 한글 자모 단위 처리 패키지 설치
!pip install hgtk
```


<br>



이번 실습에 사용할 패키지인 **fasttext** 를 설치합니다. **gensim** 의 **fasttext** 와는 별도의 패키지입니다.
```py
# fasttext 설치
!git clone https://github.com/facebookresearch/fastText.git
%cd fastText
!make
!pip install .
```


<br>





## 5.2 네이버 쇼핑 리뷰 데이터 로드

- [네이버 쇼핑 리뷰 데이터](https://github.com/bab2min/corpus/tree/master/sentiment)

<br>

필요한 라이브러리를 불러오고, 네이버 쇼핑 리뷰 데이터를 다운하겠습니다.
```py
import re
import pandas as pd
import urllib.request
from tqdm import tqdm
import hgtk
from konlpy.tag import Mecab

urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")
```
```
[output]
('ratings_total.txt', <http.client.HTTPMessage at 0x7f2e448b2820>)
```


<br>



위의 링크로부터 전체 데이터에 해당하는 **ratings_total.txt** 를 다운로드합니다. 해당 데이터에는 열제목 이 별도로 없습니다. 그래서 임의로 두 개의 열제목인 **'ratings'** 와 '**reviews'** 를 추가해주겠습니다.
```py
total_data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews'])
print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력
```
```
[output]
전체 리뷰 개수 : 200000
```


<br>


총 20 만개의 샘플이 존재합니다. 상위 5개의 샘플만 출력해보겠습니다.
```py
total_data[:5]
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/5768702b-500e-44f6-9f59-bd9a1cf8ee7b">
</p>

<br>





## 5.3 HGTK 튜토리얼

한글의 자모를 처리하는 패키지인 **hgtk** 를 사용하기에 앞서 간단히 사용법을 익혀보겠습니다. **hgtk** 의 **checker** 를 사용하면 입력이 한글인지 아닌지를 판단하여 True 또는 False를 리턴합니다.
```py
# 한글인지 체크
hgtk.checker.is_hangul('ㄱ')
```
```
[output]
True
```


<br>


```py
# 한글인지 체크
hgtk.checker.is_hangul('28')
```
```
[output]
False
```


<br>

**hgtk** 의 **letter** 를 사용하면 음절을 자모 단위로 분리하거나, 자모의 시퀀스를 다시 음절로 조합할 수 있습니다. 이는 각각 **decompose** 와 **compose** 로 가능합니다.
```py
# 음절을 초성, 중성, 종성으로 분해
hgtk.letter.decompose('남')
```
```
[output]
('ㄴ', 'ㅏ', 'ㅁ')
```


<br>


```py
# 초성, 중성을 결합
hgtk.letter.compose('ㄴ', 'ㅏ')
```
```
[output]
'나'
```


<br>


```py
# 초성, 중성, 종성을 결합
hgtk.letter.compose('ㄴ', 'ㅏ', 'ㅁ')
```
```
[output]
'남'
```


<br>


한글이 아닌 입력이 들어오거나 음절로 조합할 수 없는 경우 **NotHangulException** 을 발생시킵니다.
```py
# 한글이 아닌 입력에 대해서는 에러 발생.
hgtk.letter.decompose('1')
```
```
[output]
NotHangulException: 
```


<br>


```py
# 결합할 수 없는 상황에서는 에러 발생
hgtk.letter.compose('ㄴ', 'ㅁ', 'ㅁ')
```
```
[output]
NotHangulException: No valid Hangul character index
```


<br>





## 5.4 자모 단위 토큰화(전처리)

위에서 사용했던 **hgtk.letter.decompose()** 를 사용하여 특정 단어가 들어오면 이를 초성, 중성, 종성으로 나누는 함수 **word_to_jamo** 를 구현하겠습니다. 단, 종성이 없는 경우에는 해당 위치에 종성이 없었다는 것을 표시해주기 위해서 종성의 위치에 특수문자 **'‐'** 를 넣어주었습니다.
```py
def word_to_jamo(token):
    def to_special_token(jamo):
        if not jamo:
            return '-'
        else:
            return jamo

    decomposed_token = ''
    for char in token:
        try:
            # char(음절)을 초성, 중성, 종성으로 분리
            cho, jung, jong = hgtk.letter.decompose(char)

            # 자모가 빈 문자일 경우 특수문자 -로 대체
            cho = to_special_token(cho)
            jung = to_special_token(jung)
            jong = to_special_token(jong)
            decomposed_token = decomposed_token + cho + jung + jong

        # 만약 char(음절)이 한글이 아닐 경우 자모를 나누지 않고 추가
        except Exception as exception:
            if type(exception).__name__ == 'NotHangulException':
                decomposed_token += char

    # 단어 토큰의 자모 단위 분리 결과를 추가
    return decomposed_token
```

<br>


해당 함수에 임의의 단어 '남동생' 을 넣어 정상적으로 분리하는지 테스트해보겠습니다.
```py
word_to_jamo('남동생')
```
```
[output]
'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'
```


<br>


'남동생' 이 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ' 으로 분리된 것을 확인할 수 있습니다. 이번에는 임의의 단어 '여동생'을 넣어서 테스트해보겠습니다.
```py
word_to_jamo('여동생')
```
```
[output]
'ㅇㅕ-ㄷㅗㅇㅅㅐㅇ'
```


<br>


'여동생'의 경우 여에 종성이 없으므로 종성의 위치에 특수문자 '-'가 대신 들어간 것을 확인할 수 있습니다. 단순 형태소 분석을 했을 경우와 형태소 분석 후에 다시 자모 단위로 분해하는 경우를 동일한 예문을 통해 비교해보겠습니다. 우선 단순 형태소 분석을 했을 경우입니다.
```py
mecab = Mecab()
print(mecab.morphs('선물용으로 빨리 받아서 전달했어야 하는 상품이었는데 머그컵만 와서 당황했습니다.'))
```
```
[output]
['선물', '용', '으로', '빨리', '받', '아서', '전달', '했어야', '하', '는', '상품', '이', '었', '는데', '머그', '컵', '만', '와서', '당황', '했', '습니다', '.']
```


<br>

우리가 일반적으로 봐왔던 형태소 분석 결과입니다.



**word_to_jamo** 함수를 형태소 분석 후 호출하도록 하여 형태소 토큰들을 자모 단위로 분해하는 함수 **tokenize_by_jamo** 를 정의합니다. 이후 형태소 분석 후 자모 단위로 다시 한 번 분해한 경우입니다.
```py
def tokenize_by_jamo(s):
    return [word_to_jamo(token) for token in mecab.morphs(s)]

print(tokenize_by_jamo('선물용으로 빨리 받아서 전달했어야 하는 상품이었는데 머그컵만 와서 당황했습니다.'))
```
```
[output]
['ㅅㅓㄴㅁㅜㄹ', 'ㅇㅛㅇ', 'ㅇㅡ-ㄹㅗ-', 'ㅃㅏㄹㄹㅣ-', 'ㅂㅏㄷ', 'ㅇㅏ-ㅅㅓ-', 'ㅈㅓㄴㄷㅏㄹ', 'ㅎㅐㅆㅇㅓ-ㅇㅑ-', 'ㅎㅏ-', 'ㄴㅡㄴ', 'ㅅㅏㅇㅍㅜㅁ', 'ㅇㅣ-', 'ㅇㅓㅆ', 'ㄴㅡㄴㄷㅔ-', 'ㅁㅓ-ㄱㅡ-', 'ㅋㅓㅂ', 'ㅁㅏㄴ', 'ㅇㅘ-ㅅㅓ-', 'ㄷㅏㅇㅎㅘㅇ', 'ㅎㅐㅆ', 'ㅅㅡㅂㄴㅣ-ㄷㅏ-', '.']
```


<br>



**자모 단위 FastText 에서는 위와 같이 각 형태소 분석 결과 토큰들이 추가적으로 자모 단위로 분해된 토큰들을 가지고 학습을 하게 됩니다.** 전체 데이터에 대해서 위의 자모 단위 토큰화를 적용하겠습니다.
```py
from tqdm import tqdm

tokenized_data = []

for sample in total_data['reviews'].to_list():
    tokenzied_sample = tokenize_by_jamo(sample) # 자소 단위 토큰화
    tokenized_data.append(tokenzied_sample)
```


<br>


첫번째 샘플을 출력해보겠습니다.
```py
tokenized_data[0]
```
```
[output]
['ㅂㅐ-ㄱㅗㅇ', 'ㅃㅏ-ㄹㅡ-', 'ㄱㅗ-', 'ㄱㅜㅅ']
```


<br>


'배공빠르고 굿'이라는 기존 샘플이 형태소 분석 후에는 ['배공', '빠르', '고', '굿']으로 분해되었으며, 이를 다시 자모 단위로 나누면서 ['ㅂㅐ-ㄱㅗㅇ', 'ㅃㅏ-ㄹㅡ-', 'ㄱㅗ-', 'ㄱㅜㅅ']라는 결과가 됩니다.



그런데 이렇게 바꾸고나니 원래 단어가 무엇이었는지 알아보기 힘들다는 문제가 있습니다. 출력했을 때, 사용자가 기존의 단어가 무엇이었는지를 쉽게 알아보기 위해 초성, 중성, 종성을 입력받으면 역으로 단어로 바꿔주는 **jamo_to_word** 함수를 구현합니다.
```py
def jamo_to_word(jamo_sequence):
    tokenized_jamo = []
    index = 0

    # 1. 초기 입력
    # jamo_sequence = 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'

    while index < len(jamo_sequence):
        # 문자가 한글(정상적인 자모)이 아닐 경우
        if not hgtk.checker.is_hangul(jamo_sequence[index]):
            tokenized_jamo.append(jamo_sequence[index])
            index = index + 1

        # 문자가 정상적인 자모라면 초성, 중성, 종성을 하나의 토큰으로 간주.
        else:
            tokenized_jamo.append(jamo_sequence[index:index + 3])
            index = index + 3

    # 2. 자모 단위 토큰화 완료
    # tokenized_jamo : ['ㄴㅏㅁ', 'ㄷㅗㅇ', 'ㅅㅐㅇ']

    word = ''
    try:
        for jamo in tokenized_jamo:

            # 초성, 중성, 종성의 묶음으로 추정되는 경우
            if len(jamo) == 3:
                if jamo[2] == "-":
                    # 종성이 존재하지 않는 경우
                    word = word + hgtk.letter.compose(jamo[0], jamo[1])
                else:
                    # 종성이 존재하는 경우
                    word = word + hgtk.letter.compose(jamo[0], jamo[1], jamo[2])
            # 한글이 아닌 경우
            else:
                word = word + jamo

    # 복원 중(hgtk.letter.compose) 에러 발생 시 초기 입력 리턴.
    # 복원이 불가능한 경우 예시) 'ㄴ!ㅁㄷㅗㅇㅅㅐㅇ'
    except Exception as exception:
        if type(exception).__name__ == 'NotHangulException':
            return jamo_sequence

    # 3. 단어로 복원 완료
    # word : '남동생'

    return word
```


<br>


해당 함수의 내부 동작 방식을 설명하기 위해 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'이라는 임의의 입력이 들어왔을 때를 가정해보겠습니다.


초기 입력이 들어왔을 때는 **jamo_sequence** 라는 변수에 저장되어져 있습니다. while 문 내부에서는 **jamo_sequences** 의 각 문자에 대해서 세 개씩 분리하여 초성, 중성, 종성을 하나의 묶음으로 간주합니다. while문을 지나고나면 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'이라는 문자열은 ['ㄴㅏㅁ', 'ㄷㅗㅇ', 'ㅅㅐㅇ']이라는 리스트로 변환이 되며, 해당 리스트는 **tokenized_jamo** 라는 변수에 저장됩니다. 그리고 각 리스트의 원소를 **hgtk.letter.compose()** 의 입력으로 넣어 기존의 음절로 복원합니다.

<br>


결과적으로 '남동생'이라는 단어로 복원되고 해당 함수는 '남동생'을 최종 결과로서 리턴합니다. 실제로 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'을 입력으로 넣어 결과를 확인해보겠습니다.
```py
jamo_to_word('ㄴㅏㅁㄷㅗㅇㅅㅐㅇ')
```
```
[output]
'남동생'
```


<br>






## 5.5 FastText 학습하기

자모 단위로 토큰화 된 데이터를 가지고 **FastText** 를 학습시켜보겠습니다.
```py
import fasttext
```


<br>


**FastText** 학습을 위해서 기존 훈련 데이터를 txt 파일 형식으로 저장해야합니다.
```py
with open('tokenized_data.txt', 'w') as out:
    for line in tqdm(tokenized_data, unit=' line'):
        out.write(' '.join(line) + '\n')
```
```
[output]
100%|██████████| 200000/200000 [00:00<00:00, 473187.85 line/s]
```


<br>




두 가지 모델 **Skip-gram** 과 **CBoW** 중 **CBoW** 를 선택했습니다.
```py
model = fasttext.train_unsupervised('tokenized_data.txt', model='cbow')
model.save_model("fasttext.bin") # 모델 저장
model = fasttext.load_model("fasttext.bin") # 모델 로드
```


<br>






학습이 완료되었습니다. 임의로 '남동생'이라는 단어의 벡터값을 확인해보겠습니다. **주의할 점은 학습 시 자모 단위로 분해하였기 때문에 모델에서 벡터값을 확인할 때도 자모 단위로 분해 후에 입력으로 사용해야 합니다.**
```py
model[word_to_jamo('남동생')] # 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'
```
```
[output]
array([ 2.85749108e-01,  5.32697797e-01,  9.72192764e-01, -1.74561024e-01,
       -9.40615535e-01, -8.52213144e-01,  1.84142113e-01,  7.23624051e-01,
       -6.02583587e-01, -4.90640759e-01,  9.03603256e-01,  2.96650290e-01,
       -2.10492730e-01,  6.45978987e-01, -5.46604753e-01,  6.41264975e-01,
        6.83590710e-01,  5.07914782e-01,  1.90157950e-01, -3.27531621e-02,
        5.80425084e-01, -5.27899086e-01,  6.99647903e-01, -1.41876325e-01,
        5.80996238e-02, -3.13640386e-01,  7.06844479e-02,  1.19922531e+00,
        9.48624730e-01, -7.08683491e-01,  5.12313426e-01, -8.10058236e-01,
        2.31832623e-01, -2.90871948e-01,  1.51137781e+00, -2.15766624e-01,
        4.38416228e-02, -2.49742463e-01,  9.85836610e-02,  1.48401380e-01,
        4.82708663e-01, -9.81113911e-02,  3.92483830e-01, -8.28986242e-02,
       -2.54946172e-01, -1.10600853e+00,  7.52483681e-03,  3.19441855e-01,
       -6.56547397e-02, -2.13221177e-01, -2.11511150e-01, -2.59903312e-01,
        1.69138134e-01,  1.49033908e-02, -9.99034107e-01, -3.28279957e-02,
        1.10627757e-02,  2.43498445e-01, -2.38837197e-01,  1.86610088e-01,
       -1.39049098e-01, -1.18185975e-01,  1.61835730e-01,  7.25804329e-01,
       -4.35180724e-01,  3.77287447e-01, -4.06595647e-01, -1.76645592e-01,
       -2.67820716e-01,  4.91925776e-01, -2.82297432e-01, -6.00573897e-01,
        4.94795799e-01,  1.35222033e-01, -1.17796496e-01, -7.76124895e-01,
        2.27492508e-02,  1.36140555e-01,  3.97971332e-01,  9.36240926e-02,
        8.48273218e-01,  7.88985193e-01,  5.37583753e-02,  6.32351160e-01,
        7.73415864e-01,  6.23026609e-01, -8.15240979e-01, -7.78561473e-01,
        7.49277830e-01,  1.29948840e-01,  6.60207570e-01, -4.03202087e-01,
       -6.72111869e-01, -9.39618289e-01, -8.69688034e-01,  8.82879972e-01,
       -1.33745838e-02,  4.36232805e-01, -2.32288629e-01, -1.67192949e-04],
      dtype=float32)
```


<br>



남동생 '벡터'와 가장 유사도가 높은 벡터들을 뽑아보겠습니다. 이는 **get_nearest_neighbors()** 를 사용하여 가능합니다. 두번째 인자인 **k** 값으로 10을 주면, 가장 유사한 벡터 상위 10개를 출력합니다.
```py
model.get_nearest_neighbors(word_to_jamo('남동생'), k=10)
```
```
[output]
[(0.8671373724937439, 'ㄷㅗㅇㅅㅐㅇ'),
 (0.8345811367034912, 'ㄴㅏㅁㅊㅣㄴ'),
 (0.7394193410873413, 'ㄴㅏㅁㅍㅕㄴ'),
 (0.7316157817840576, 'ㅊㅣㄴㄱㅜ-'),
 (0.7173355221748352, 'ㅅㅐㅇㅇㅣㄹ'),
 (0.7168329358100891, 'ㄴㅏㅁㅇㅏ-'),
 (0.7005258202552795, 'ㅈㅗ-ㅋㅏ-'),
 (0.6888477802276611, 'ㅈㅜㅇㅎㅏㄱㅅㅐㅇ'),
 (0.6667895317077637, 'ㅇㅓㄴㄴㅣ-'),
 (0.6643229126930237, 'ㄴㅏㅁㅈㅏ-')]
```


<br>



그런데 출력으로 나오는 벡터들도 자모 단위로 분해해서 나오기 때문에 읽기가 어렵습니다. 이전에 만들어준 **jamo_to_word** 함수를 사용하여 출력 결과를 좀 더 깔끔하게 확인할 수 있습니다.
```py
def transform(word_sequence):
    return [(jamo_to_word(word), similarity) for (similarity, word) in word_sequence]
```

지금부터 결과들을 나열해보겠습니다.
```py
print(transform(model.get_nearest_neighbors(word_to_jamo('남동생'), k=10)))
```
```
[output]
[('동생', 0.8671373724937439), ('남친', 0.8345811367034912), ('남편', 0.7394193410873413), ('친구', 0.7316157817840576), ('생일', 0.7173355221748352), ('남아', 0.7168329358100891), ('조카', 0.7005258202552795), ('중학생', 0.6888477802276611), ('언니', 0.6667895317077637), ('남자', 0.6643229126930237)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('남동쉥'), k=10)))
```
```
[output]
[('남동생', 0.8909438252449036), ('남친', 0.8003354668617249), ('남매', 0.7774966955184937), ('남김', 0.7451346516609192), ('남긴', 0.7383974194526672), ('남짓', 0.7368336319923401), ('남녀', 0.7326962351799011), ('남아', 0.7286370992660522), ('남여', 0.7266424894332886), ('남길', 0.7219088077545166)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('남동셍ㅋ'), k=10)))
```
```
[output]
[('남동생', 0.8234370946884155), ('남친', 0.7265772819519043), ('남김', 0.7082480788230896), ('남길', 0.6784865260124207), ('남녀', 0.6686286330223083), ('남매', 0.6675403714179993), ('남여', 0.6633204817771912), ('남겼', 0.6621609926223755), ('남짓', 0.6599602103233337), ('남긴', 0.6571483016014099)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('난동생'), k=10)))
```
```
[output]
[('남동생', 0.8642392158508301), ('난생', 0.8244139552116394), ('남편', 0.8014969229698181), ('남친', 0.7568559646606445), ('동생', 0.7568278312683105), ('남아', 0.754828929901123), ('나눴', 0.7011327147483826), ('중학생', 0.7001649737358093), ('남자', 0.6799314022064209), ('신랑', 0.6761581897735596)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('낫동생'), k=10)))
```
```
[output]
[('남동생', 0.9303204417228699), ('동생', 0.8740969300270081), ('남편', 0.7611657381057739), ('남친', 0.7513895034790039), ('친구', 0.7390786409378052), ('중학생', 0.7209896445274353), ('조카', 0.7082139253616333), ('남아', 0.7011557817459106), ('난생', 0.7001751661300659), ('나눴', 0.6832748055458069)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('납동생'), k=10)))
```
```
[output]
[('남동생', 0.9049338102340698), ('동생', 0.8326806426048279), ('남편', 0.7896609902381897), ('남친', 0.7583615183830261), ('난생', 0.7417805790901184), ('중학생', 0.7253825664520264), ('남아', 0.7192257046699524), ('친구', 0.7001274824142456), ('나눴', 0.697450578212738), ('고등학생', 0.694034218788147)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('냚동생'), k=10)))
```
```
[output]
[('동생', 0.967219889163971), ('남동생', 0.8974405527114868), ('친구', 0.8116076588630676), ('조카', 0.7770885229110718), ('언니', 0.7635160088539124), ('딸', 0.7545560598373413), ('생일', 0.7490536570549011), ('딸애', 0.7439687252044678), ('중학생', 0.7377141714096069), ('남편', 0.7292447686195374)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('고품질'), k=10)))
```
```
[output]
[('품질', 0.8602216839790344), ('음질', 0.795451819896698), ('땜질', 0.72904372215271), ('퀄리티', 0.7188094854354858), ('찜질', 0.6836755871772766), ('군것질', 0.6558197736740112), ('고감', 0.6491621732711792), ('사포질', 0.6487373113632202), ('성질', 0.6361984014511108), ('퀄러티', 0.6332342624664307)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('고품쥘'), k=10)))
```
```
[output]
[('고품질', 0.8344911932945251), ('재고품', 0.7825105786323547), ('소모품', 0.7390694618225098), ('재품', 0.7284044027328491), ('반제품', 0.7194015979766846), ('고퀄', 0.7192908525466919), ('중고품', 0.6962606310844421), ('제품', 0.6944983005523682), ('화학제품', 0.6882331967353821), ('타제품', 0.6859912276268005)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('노품질'), k=10)))
```
```
[output]
[('고품질', 0.8985658884048462), ('품질', 0.874873697757721), ('음질', 0.7521613836288452), ('퀄리티', 0.7211642265319824), ('땜질', 0.7002740502357483), ('화질', 0.668508768081665), ('찜질', 0.6579814553260803), ('퀄러티', 0.6188082098960876), ('질', 0.6149688363075256), ('가격', 0.6097935438156128)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('보품질'), k=10)))
```
```
[output]
[('고품질', 0.8238304853439331), ('품질', 0.7731610536575317), ('음질', 0.755185067653656), ('땜질', 0.6912787556648254), ('화질', 0.6854788661003113), ('재질', 0.6820527911186218), ('보풀', 0.6702924370765686), ('찜질', 0.668892502784729), ('퀄리티', 0.6623635292053223), ('사포질', 0.6457920670509338)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('제품'), k=10)))
```
```
[output]
[('반제품', 0.8848034739494324), ('완제품', 0.872488796710968), ('상품', 0.8489471077919006), ('타제품', 0.8351007699966431), ('재품', 0.8256241083145142), ('중품', 0.8064692616462708), ('최상품', 0.7975529432296753), ('화학제품', 0.7878970503807068), ('명품', 0.7761856317520142), ('제풍', 0.7698684930801392)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('제품ㅋ'), k=10)))
```
```
[output]
[('제품', 0.8449548482894897), ('완제품', 0.7483550906181335), ('최상품', 0.7337923645973206), ('제풍', 0.704480767250061), ('상품', 0.7037019729614258), ('반제품', 0.6991694569587708), ('성품', 0.6684542298316956), ('타제품', 0.6669901609420776), ('재품', 0.6596834063529968), ('완성품', 0.6593179106712341)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('제품^^'), k=10)))
```
```
[output]
[('제품', 0.9371058344841003), ('제풍', 0.791787326335907), ('반제품', 0.7801757454872131), ('완제품', 0.7779927849769592), ('상품', 0.7686623930931091), ('타제품', 0.762442946434021), ('최상품', 0.7511221766471863), ('재품', 0.7040295004844666), ('화학제품', 0.695855438709259), ('중품', 0.6953531503677368)]
```


<br>


```py
print(transform(model.get_nearest_neighbors(word_to_jamo('제푼ㅋ'), k=10)))
```
```
[output]
[('제풍', 0.6460399627685547), ('제품', 0.5884508490562439), ('최상품', 0.5207403898239136), ('완제품', 0.504409670829773), ('젝', 0.49801012873649597), ('제왕', 0.4525451362133026), ('반제품', 0.4501373767852783), ('제습', 0.44745227694511414), ('최상급', 0.4468994140625), ('상품', 0.4445939064025879)]
```





