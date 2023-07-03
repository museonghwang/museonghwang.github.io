---
layout: post
title: FastText의 이해
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---






1. FastText 개요
2. Word2Vec의 대표적인 문제점
    - 2.1 OOV(Out-of-Vocabulary) 문제
    - 2.2 형태학적 특징을 반영할 수 없는 문제
3. FastText 내부 단어(subword)의 이해
4. FastText Pre-training
5. FastText의 강점
    - 5.1 모르는 단어(Out Of Vocabulary, OOV)에 대한 대응
    - 5.2 단어 집합 내 빈도 수가 적었던 단어(Rare Word)에 대한 대응
    - 5.3 단어 집합 내 노이즈가 많은 코퍼스에 대한 대응
6. Word2Vec Vs. FastText
    - 6.1 gensim 패키지 버전확인
    - 6.2 Libaray Import
    - 6.3 훈련 데이터 이해
    - 6.4 훈련 데이터 전처리
    - 6.5 Word2Vec 훈련
7. FastText Summaray

<br>



단어를 벡터로 만드는 또 다른 방법으로는 페이스북에서 개발한 FastText가 있습니다. Word2Vec 이후에 나온 것이기 때문에, 메커니즘 자체는 Word2Vec의 확장이라고 볼 수 있습니다. Word2Vec와 FastText와의 가장 큰 차이점이라면 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다. 내부 단어. 즉, 서브워드(subword)를 고려하여 학습합니다.

<br>
<br>






# 1. FastText 개요

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






# 2. Word2Vec의 대표적인 문제점

<br>


# 2.1 OOV(Out-of-Vocabulary) 문제

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/1fce8892-3b10-46de-af2d-cb669828d7fc">
</p>

<br>


**Word2Vec** 의 **Vocabulary** 에 **"tensor"** 와 **"flow"** 가 있더라도, **"tensorflow"** 라는 단어가 **Vocabulary** 에 없다면, **"tensorflow"** 의 벡터값을 얻을 수 없습니다.


<br>


# 2.2 형태학적 특징을 반영할 수 없는 문제

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/4954395b-66e4-4b1d-9a26-12fad5d1701e">
</p>

<br>


위 단어들은 **eat** 이라는 동일한 어근을 가집니다. 하지만 **Word2Vec** 에서의 각 단어는 각 벡터의 값을 가질뿐입니다. 즉, 하나의 단어에 고유한 벡터를 할당하므로 단어의 형태학적 특징을 반영할 수 없다는 문제가 있습니다.

<br>





# 3. FastText 내부 단어(subword)의 이해

**<span style="color:red">FastText에서는 각 단어는 글자 단위 n-gram의 구성으로 취급</span>** 합니다. $n$ 을 몇으로 결정하는지에 따라서 단어들이 얼마나 분리되는지 결정됩니다. 예를 들어서 $n$ 을 3으로 잡은 트라이그램(tri-gram)의 경우, **apple** 은 **app**, **ppl**, **ple** 로 분리하고 이들을 벡터로 만듭니다. **<u>더 정확히는 시작과 끝을 의미하는 <, >를 도입하여 아래의 5개 내부 단어(subword) 토큰을 벡터로 만듭니다.</u>**
```
# n = 3인 경우
<ap, app, ppl, ple, le>
```

<br>



그리고 **<u>여기에 추가적으로 하나를 더 벡터화</u>** 하는데, **기존 단어에 <, 와 >를 붙인 토큰** 입니다.
```
# 특별 토큰
<apple>
```

<br>



다시 말해 $n=3$ 인 경우, **FastText** 는 단어 **apple** 에 대해서 다음의 6개의 토큰을 벡터화하는 것입니다.
```
# n = 3인 경우
<ap, app, ppl, ple, le>, <apple>
```

<br>


그런데 **실제 사용할 때는 $n$ 의 최소값과 최대값으로 범위를 설정** 할 수 있는데, 기본값으로는 각각 3과 6으로 설정되어져 있습니다. 다시 말해 최소값 = 3, 최대값 = 6인 경우라면, 단어 apple에 대해서 **FastText** 는 아래 내부 단어들을 벡터화합니다.
```
# n = 3 ~ 6인 경우
<ap, app, ppl, ppl, le>, <app, appl, pple, ple>, <appl, pple>, ..., <apple>
```

<br>


여기서 **<span style="color:red">내부 단어들을 벡터화한다는 의미</span>** 는 **<span style="background-color: #fff5b1">저 단어들에 대해서 Word2Vec을 수행한다는 의미</span>** 입니다. 위와 같이 내부 단어들의 벡터값을 얻었다면, 단어 **apple** 의 **<span style="color:red">벡터값은 저 위 벡터값들의 총 합으로 구성</span>** 합니다.
```
apple = <ap + app + ppl + ppl + le> + <app + appl + pple + ple> + <appl + pple> + , ..., +<apple>
```

<br>





다른 예문으로 한번 더 살펴보고 확실하게 이해해보겠습니다. 위에서 **<span style="color:red">FastText는 단어를 Character 단위의 n-gram으로 간주하며, $n$ 을 몇으로 하느냐에 따라서 단어가 얼마나 분리되는지가 결정</span>** 된다고 했습니다. 단어 'eating'을 예를 들어보겠습니다.


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





# 4. FastText Pre-training

FastText의 훈련 과정을 이해해보겠습니다. 여기서는 SGNS(Skip-gram with Negative Sampleing)을 사용합니다. **<u>현재의 목표는 중심 단어 eating으로부터 주변 단어 am과 food를 예측하는 것</u>** 입니다.

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





# 5. FastText의 강점

<br>


# 5.1 모르는 단어(Out Of Vocabulary, OOV)에 대한 대응

**FastText** 의 인공 신경망을 학습한 후에는 **<u>데이터 셋의 모든 단어의 각 n-gram에 대해서 워드 임베딩이 됩니다.</u>** 이렇게 되면 장점은 **<span style="background-color: #fff5b1">데이터 셋만 충분한다면 위와 같은 내부 단어(Subword)를 통해 모르는 단어(Out Of Vocabulary, OOV)에 대해서도 다른 단어와의 유사도를 계산할 수 있다는 점</span>** 입니다.

가령, FastText에서 birthplace(출생지)란 단어를 학습하지 않은 상태라고 해봅시다. 하지만 다른 단어에서 birth와 place라는 내부 단어가 있었다면, FastText는 birthplace의 벡터를 얻을 수 있습니다. 이는 모르는 단어에 제대로 대처할 수 없는 Word2Vec, GloVe와는 다른 점입니다.

<br>



# 5.2 단어 집합 내 빈도 수가 적었던 단어(Rare Word)에 대한 대응

**<u>Word2Vec의 경우에는 등장 빈도 수가 적은 단어(rare word)에 대해서는 임베딩의 정확도가 높지 않다는 단점</u>** 이 있었습니다. 참고할 수 있는 경우의 수가 적다보니 정확하게 임베딩이 되지 않는 경우입니다.


하지만 **FastText** 의 경우, **<span style="background-color: #fff5b1">만약 단어가 희귀 단어라도, 그 단어의 n-gram이 다른 단어의 n-gram과 겹치는 경우라면, Word2Vec과 비교하여 비교적 높은 임베딩 벡터값을 얻습니다.</span>**

<br>




# 5.3 단어 집합 내 노이즈가 많은 코퍼스에 대한 대응

FastText가 노이즈가 많은 코퍼스에서 강점을 가진 것 또한 이와 같은 이유입니다. 모든 훈련 코퍼스에 오타(Typo)나 맞춤법이 틀린 단어가 없으면 이상적이겠지만, 실제 많은 비정형 데이터에는 오타가 섞여있습니다. 그리고 **<u>오타가 섞인 단어는 당연히 등장 빈도수가 매우 적으므로 일종의 희귀 단어가 됩니다.</u>** 즉, Word2Vec에서는 오타가 섞인 단어는 임베딩이 제대로 되지 않지만, **<span style="background-color: #fff5b1">FastText는 이에 대해서도 일정 수준의 성능을 보입니다.</span>**

예를 들어 단어 apple과 오타로 p를 한 번 더 입력한 appple의 경우에는 실제로 많은 개수의 동일한 n-gram을 가질 것입니다.


<br>




# 6. Word2Vec Vs. FastText

간단한 실습을 통해 영어 Word2Vec와 FastText의 차이를 비교해보도록 하겠습니다.

<br>



## 6.1 gensim 패키지 버전확인

파이썬의 **gensim** 패키지에는 **Word2Vec** 을 지원하고 있어, **gensim** 패키지를 이용하면 손쉽게 단어를 임베딩 벡터로 변환시킬 수 있습니다. **Word2Vec** 을 학습하기 위한 **gensim** 패키지 버전을 확인합니다.
```py
import gensim
gensim.__version__
```
```
[output]
'4.3.1'
```


<br>




## 6.2 Libaray Import

영어로 된 코퍼스를 다운받아 전처리를 수행하고, 전처리한 데이터를 바탕으로 **Word2Vec** 작업을 진행하겠습니다. 우선 필요 라이브러리를 불러옵니다.
```py
import re
from lxml import etree
import urllib.request
import zipfile

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize
```
```
[output]
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
```


<br>




## 6.3 훈련 데이터 이해

Word2Vec을 학습하기 위해서 데이터를 다운로드합니다. 사용할 훈련 데이터는, ted 영상들의 자막 데이터입니다. 파일의 형식은 xml 파일입니다. [해당 링크](https://github.com/GaoleMeng/RNN-and-FFNN-textClassification/blob/master/ted_en-20160408.zip)를 통해 내려받아 **ted_en-20160408.xml** 라는 이름의 파일을 설치할 수도 있고, 파이썬 코드를 통해 자동으로 설치할 수도 있습니다.
```py
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
```
```
[output]
('ted_en-20160408.xml', <http.client.HTTPMessage at 0x7fbaa9a92620>)
```


<br>


위의 코드를 통해 **xml** 파일을 내려받으면, 다음과 같은 파일을 볼 수 있습니다.
```xml
<?xml version="1.0" encoding="UTF-8"?>
<xml language="en"><file id="1">
  <head>
    <url>http://www.ted.com/talks/knut_haanaes_two_reasons_companies_fail_and_how_to_avoid_them</url>

    ...

    <content>Here are two reasons companies fail: they only do more of the same, or they only do what's new.

    ...

    So let me leave you with this. Whether you're an explorer by nature or whether you tend to exploit what you already know, don't forget: the beauty is in the balance.
    Thank you.
    (Applause)</content>
</file>
<file id="2">
  <head>
    <url>http://www.ted.com/talks/lisa_nip_how_humans_could_evolve_to_survive_in_space</url>
    
    ...

    (Applause)</content>
</file>
</xml>
```

<br>

훈련 데이터 파일은 **xml** 문법으로 작성되어 있어 자연어를 얻기 위해서는 전처리가 필요합니다. 얻고자 하는 실질적 데이터는 영어문장으로만 구성된 내용을 담고 있는 **'\<content\>'** 와 **'\</content\>'** 사이의 내용입니다. 전처리 작업을 통해 **xml** 문법들은 제거하고, 해당 데이터만 가져와야 합니다. 뿐만 아니라, **'\<content\>'** 와 **'\</content\>'** 사이의 내용 중에는 (Laughter)나 (Applause)와 같은 배경음을 나타내는 단어도 등장하는데 이 또한 제거해야 합니다.


<br>




## 6.4 훈련 데이터 전처리

```py
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)
```


<br>


현재 영어 텍스트가 **content_text** 에 저장되어져 있습니다. 이에 대해서 **NLTK** 의 **sent_tokenize** 를 통해서 문장을 구분해보겠습니다.
```py
print('영어 텍스트의 개수 : {}'.format(len(content_text)))
```
```
[output]
영어 텍스트의 개수 : 24062319
```


<br>



```py
# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]

print('총 샘플의 개수 : {}'.format(len(result)))
```
```
[output]
총 샘플의 개수 : 273424
```


<br>



총 문장 샘플의 개수는 273,424개입니다. 샘플 3개만 출력해보겠습니다.
```py
for line in result[:3]: # 샘플 3개만 출력
    print(line)
```
```
[output]
['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 's', 'new']
['to', 'me', 'the', 'real', 'real', 'solution', 'to', 'quality', 'growth', 'is', 'figuring', 'out', 'the', 'balance', 'between', 'two', 'activities', 'exploration', 'and', 'exploitation']
['both', 'are', 'necessary', 'but', 'it', 'can', 'be', 'too', 'much', 'of', 'a', 'good', 'thing']
```


<br>

상위 3개 문장만 출력해보았는데 토큰화가 잘 수행되었음을 볼 수 있습니다. 이제 **Word2Vec** 모델에 텍스트 데이터를 훈련시킵니다.

<br>






## 6.5 Word2Vec 훈련

여기서 **Word2Vec** 의 하이퍼파라미터값은 다음과 같습니다.

- **vector_size** = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
- **window** = 컨텍스트 윈도우 크기
- **min_count** = 단어 최소 빈도 수 제한(빈도가 적은 단어들은 학습하지 않음)
- **workers** = 학습을 위한 프로세스 수
- **sg**
    - 0 : CBOW
    - 1 : Skip-gram

<br>

```py
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=result,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=0
)
```


<br>



**Word2Vec** 에 대해서 학습을 진행하였습니다. **Word2Vec** 는 입력한 단어에 대해서 가장 유사한 단어들을 출력하는 **model.wv.most_similar**을 지원합니다. 특정 단어와 가장 유사한 단어들을 추출해보겠습니다. 이때 코사인 유사도라는 것을 유사도 메트릭으로 사용하며, 값의 범위는 **-1 ~ 1** 입니다.
```py
model.wv.most_similar("drink")
```
```
[output]
[('drinking', 0.7223415374755859),
 ('milk', 0.7076201438903809),
 ('buy', 0.6865729093551636),
 ('eat', 0.6797118186950684),
 ('rent', 0.6692221760749817),
 ('coffee', 0.6465736627578735),
 ('burn', 0.6400552988052368),
 ('wash', 0.6392359137535095),
 ('wear', 0.6316686868667603),
 ('steal', 0.6246187686920166)]
```


<br>


입력 단어에 대해서 유사한 단어를 찾아내는 코드에 이번에는 electrofishing이라는 단어를 넣어보겠습니다. 해당 코드는 정상 작동하지 않고 에러를 발생시킵니다.
```py
model.wv.most_similar("electrofishing")
```
```
[output]
KeyError: "Key 'electrofishing' not present in vocabulary"
```


<br>


에러 메시지는 단어 집합(Vocabulary)에 electrofishing이 존재하지 않는다고 합니다. 이처럼 **<span style="color:red">Word2Vec는 학습 데이터에 존재하지 않는 단어. 즉, 모르는 단어에 대해서는 임베딩 벡터가 존재하지 않기 때문에 단어의 유사도를 계산할 수 없습니다.</span>**

<br>




## 6.6 FastText 훈련

이번에는 전처리 코드는 그대로 사용하고 Word2Vec 학습 코드만 FastText 학습 코드로 변경하여 실행해보겠습니다.
```py
model = FastText(
    result,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1
)
```


<br>

```py
model.wv.most_similar("drink")
```
```
[output]
[('drinks', 0.8812541961669922),
 ('drinkable', 0.7786468267440796),
 ('sweat', 0.7572851181030273),
 ('drinking', 0.7571462988853455),
 ('cigarette', 0.7453484535217285),
 ('cheat', 0.7437908053398132),
 ('alcohol', 0.739535927772522),
 ('grin', 0.738060474395752),
 ('burn', 0.735490083694458),
 ('sweater', 0.7331915497779846)]
```


<br>



이번에는 electrofishing에 대해서 유사 단어를 찾아보도록 하겠습니다.
```py
model.wv.most_similar("electrofishing")
```
```
[output]
[('electrolyte', 0.859164297580719),
 ('electrolux', 0.8576809763908386),
 ('electroencephalogram', 0.8483548760414124),
 ('electroshock', 0.8433628082275391),
 ('electro', 0.8416800498962402),
 ('electrogram', 0.8294049501419067),
 ('electrochemical', 0.8202094435691833),
 ('electron', 0.8182265758514404),
 ('electric', 0.8150432705879211),
 ('airbus', 0.8121421933174133)]
```


<br>


**<span style="color:red">Word2Vec는 학습하지 않은 단어에 대해서 유사한 단어를 찾아내지 못 했지만, FastText는 유사한 단어를 계산해서 출력하고 있음을 볼 수 있습니다.</span>**



<br>





# 7. FastText Summaray

충분히 잘 학습된 FastText는 전체 Word가 아니라 Subword들의 유사도를 반영함을 확인할 수 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/3c970188-950d-4908-976c-4fe4955f66bc">
</p>





