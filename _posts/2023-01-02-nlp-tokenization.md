---
layout: post
title: Tokenization 개념과 영어 및 한국어 특성
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---




- Tokenization 개념
- 단어 토큰화(Word Tokenization)
- Tokenization 고려사항
- 영어 Word Tokenization
    1. NLTK의 토크나이저 - word_tokenize
    2. NLTK의 토크나이저 - WordPunctTokenizer
    3. NLTK의 토크나이저 - TreebankWordTokenizer
    4. Keras의 토크나이저 - text_to_word_sequence
    5. 띄어쓰기를 기준으로 하는 단어 토큰화(잘 되는 것 같아도 하지마세요)
- 한국어 토큰화의 특징
    1. 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않는다. -> 띄어쓰기 보정 : PyKoSpacing
    2. 한국어는 주어 생략은 물론 어순도 중요하지 않다.
    3. 한국어는 교착어이다.
- 한국어 Word Tokenization(KoNLPy)
    1. 띄어쓰기를 기준으로 하는 단어 토큰화(가급적 하지마세요)
    2. 형태소 분석기 KoNLPy 설치
    3. KoNLPy - 형태소 분석기 Okt
    4. KoNLPy - 형태소 분석기 꼬꼬마
    5. KoNLPy - 형태소 분석기 코모란
    6. KoNLPy - 형태소 분석기 한나눔
    7. KoNLPy - 형태소 분석기 Mecab
- 문장 토큰화(Sentence Tokenization)
- 영어 Sentence Tokenization(NLTK)
- 한국어 Sentence Tokenization(KSS)

<br>





# Tokenization 개념

주어진 **<u>텍스트(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업</u>** 을 **<span style="color:red">토큰화(tokenization)</span>** 라고 부릅니다. 토큰의 단위가 상황에 따라 다르지만, **<span style="color:red">보통 의미있는 단위로 토큰을 정의</span>** 합니다. 일반적으로 토큰의 단위는 크게는 **<span style="background-color: #fff5b1">'문장'</span>** 작게는 **<span style="background-color: #fff5b1">'단어'</span>** 라고 보시면 됩니다.


자연어 처리를 위해서는 우선 텍스트에 대한 정보를 단위별로 나누는 것이 일반적인데, **<span style="background-color: #fff5b1">왜냐하면 기계에게 어느 구간까지가 문장이고, 단어인지를 알려주어야 하기 때문</span>** 입니다. **<span style="color:red">문장 토큰화</span>**, **<span style="color:red">단어 토큰화</span>**, **<span style="color:red">subword 토큰화</span>** 등 다양한 단위의 토큰화가 존재합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/66980804-59d2-42b7-8e1f-e6f0ca203044">
</p>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/9748e413-ec34-484f-889d-39fcb22b22c6">
</p>

<br>





# 단어 토큰화(Word Tokenization)


**<u>토큰의 기준을 단어(word)(단어구, 의미를 갖는 문자열 등)</u>** 로 하는 경우, **<span style="background-color: #fff5b1">단어 토큰화(word tokenization)</span>** 라고 합니다.


예를들어 아래의 입력으로부터 **구두점(punctuation)** 과 같은 문자를 제외시키는 간단한 **단어 토큰화** 작업을 해보겠습니다. 구두점이란 마침표(.), 컴마(,), 물음표(?), 세미콜론(;), 느낌표(!) 등과 같은 기호를 말합니다.

- 입력 : Time is an illusion. Lunchtime double so!
- 출력 : "Time", "is", "an", "illustion", "Lunchtime", "double", "so"

<br>


위 예제에서 토큰화 작업은 굉장히 간단합니다. 구두점을 지운 뒤에 띄어쓰기(whitespace)를 기준으로 잘라냈습니다. 하지만, **<u>보통 토큰화 작업은 단순히 구두점이나 특수문자를 전부 제거하는 정제(cleaning) 작업을 수행하는 것만으로 해결되지 않습니다.</u>** 구두점이나 특수문자를 전부 제거하면 토큰이 의미를 잃어버리는 경우가 발생 하기도 하며, 심지어 띄어쓰기 단위로 자르면 사실상 단어 토큰이 구분되는 영어와 달리, 한국어는 띄어쓰기만으로는 단어 토큰을 구분하기 어렵습니다.

<br>




# Tokenization 고려사항

**<u>토큰화 작업은 단순하게 코퍼스에서 구두점을 제외하고 공백 기준으로 잘라내는 작업이라고 간주할 수는 없습니다.</u>** 이러한 일은 보다 섬세한 알고리즘이 필요한데 그 이유를 정리하면 다음과 같습니다.

- **단순 띄어쓰기로 단어를 구분하면 안됨**
    - We are the One!! -> ['We', 'are', 'the', 'One!!']
    - We are the One -> ['We', 'are', 'the', 'One']
    - **<span style="color:red">특수문자로 인해 다른 단어로 인식</span>** 되는 경우가 발생
- **구두점 및 특수문자를 단순 제외하면 안됨**
    - 구두점 : 마침표(.)의 경우 문장의 경계를 알 수 있는데 도움이 됨.
    - 단어 자체에 구두점 : Ph.D -> Ph D -> ['Ph', 'D']
    - 가격 : $45.55 -> 45 55 -> ['45', '55']
    - 날짜 : 01/02/06 -> 01 02 06 -> ['01', '02', '06']
    - **<span style="color:red">본래의 의미가 상실</span>** 되는 경우가 발생
- **줄임말과 단어 내에 띄어쓰기가 있는 경우**
    - 줄임말
        - what're -> what are의 줄임말(re:접어(clitic))
        - we're -> we are의 줄임말
        - I'm -> I am의 줄임말(m:접어(clitic))
    - 단어 내 띄어쓰기
        - New York
        - rock 'n' roll

<br>




# 영어 Word Tokenization

영어로 토큰화를 할 때는 일반적으로 **NLTK** 라는 패키지로, **<span style="color:red">영어 자연어 처리를 위한 패키지</span>** 라고 보면 됩니다. **NLTK** 에서는 다양한 영어 토크나이저(토큰화를 수행하는 도구)를 제공하고 있으며, **<u>토큰화 결과는 토크나이저마다 규칙이 조금씩 다르므로 어떤 토크나이저를 사용할 지 정답은 없습니다.</u>**


토큰화를 하다보면, 예상하지 못한 경우가 있어서 **<span style="color:red">토큰화의 기준</span>** 을 생각해봐야 하는 경우가 발생합니다. 물론, 이러한 선택은 **<span style="background-color: #fff5b1">해당 데이터를 가지고 어떤 용도로 사용할 것인지에 따라서 그 용도에 영향이 없는 기준</span>** 으로 정하면 됩니다.

<br>


## 1. NLTK의 토크나이저 - word_tokenize

```py
import nltk
nltk.download('punkt')
```


<br>


아래의 문장을 보면 Don't와 Jone's에는 아포스트로피(')가 들어가있습니다.
```py
sentence = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
```


<br>


아포스트로피가 들어간 상황에서 Don't와 Jone's는 **word_tokenize** 에 의해 어떻게 토큰화되는지 살펴보겠습니다.
```py
from nltk.tokenize import word_tokenize
print(word_tokenize(sentence))
```
```
[output]
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
```


<br>

**word_tokenize** 는 Don't를 Do와 n't로 분리하였으며, Jone's는 Jone과 's로 분리한 것을 확인할 수 있습니다.

<br>




## 2. NLTK의 토크나이저 - WordPunctTokenizer

```py
from nltk.tokenize import WordPunctTokenizer  
print(WordPunctTokenizer().tokenize(sentence))
```
```
[output]
['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
```


<br>

**WordPunctTokenizer** 는 Don't를 Don과 '와 t로 분리하였으며, Jone's를 Jone과 '와 s로 분리한 것을 확인할 수 있습니다.

<br>



## 3. NLTK의 토크나이저 - TreebankWordTokenizer

**TreebankWordTokenizer** 는 표준 토큰화 규칙인 **Penn Treebank Tokenization** 를 따르는 토크나이저 입니다.

- 규칙 1. 하이푼으로 구성된 단어는 하나로 유지한다.
- 규칙 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.

<br>

```py
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))
```
```
[output]
['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
```


<br>

```py
print(tokenizer.tokenize(sentence))
```
```
[output]
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
```


<br>



## 4. Keras의 토크나이저 - text_to_word_sequence

```py
from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence(sentence))

```
```
[output]
["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
```


<br>


지금까지 주어진 문자열로부터 토크나이저를 사용하여 단어 토큰화를 수행해봤습니다. **<u>다시 말하지만 뭐가 더 좋은지 정답은 없습니다.</u>** 사실 **<span style="color:red">토크나이저마다 각자 규칙이 다르기 때문에 사용하고자 하는 목적에 따라 토크나이저를 선택하는 것이 중요</span>** 합니다.

<br>



## 5. 띄어쓰기를 기준으로 하는 단어 토큰화(잘 되는 것 같아도 하지마세요)

**사실 영어는 띄어쓰기를 기준으로 단어 토큰화를 한다고 하더라도 꽤 잘 되는 편입니다.** 하지만 그럼에도 띄어쓰기를 기준으로 단어 토큰화를 하는 것은 하지 않는 것이 좋은데 그 이유를 이해해보겠습니다.


우선 다음과 같은 영어 문장에 대해 **NLTK** 로 토큰화를 하겠습니다.
```py
from nltk.tokenize import word_tokenize

en_text = "A Dog Run back corner near spare bedrooms!!!!"
print(word_tokenize(en_text))
```
```
[output]
['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms', '!', '!', '!', '!']
```


<br>


잘 동작합니다. 이번에는 **NLTK** 가 아닌 그냥 띄어쓰기 단위로 토큰화를 해보겠습니다. 파이썬은 주어진 문자열에 **.split()** 을 하면 띄어쓰기를 기준으로 전부 원소를 잘라서 리스트 형태로 리턴합니다.
```py
print(en_text.split())
```
```
[output]
['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms!!!!']
```


<br>

이데 바로 띄어쓰기를 기준으로 단어 토큰화를 수행한 결과입니다.

사실 영어는 **NLTK** 라는 패키지를 사용하면 좀 더 섬세한 토큰화를 하기는 하지만, 띄어쓰기를 하는 것만으로도 거의 토큰화가 잘 되는 편입니다. 하지만 그럼에도 띄어쓰기를 기준으로 하는 것을 지양(하지마세요)하라는 것은 이유가 있습니다. 예를 들어 영어 문장에 특수 문자를 추가하여 **NLTK** 로 토큰화를 하겠습니다.
```py
from nltk.tokenize import word_tokenize

en_text = "A Dog Run back corner near spare bedrooms... bedrooms!!"
print(word_tokenize(en_text))
```
```
[output]
['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms', '...', 'bedrooms', '!', '!']
```


<br>


보면 특수문자들도 알아서 다 띄워서 bedrooms이 정상적으로 분리되었습니다. 하지만 띄어쓰기 단위로 토큰화를 한다면 어떻게 되는지 보겠습니다.
```py
print(en_text.split())
```
```
[output]
['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms...', 'bedrooms!!']
```


<br>



bedrooms와 ...가 붙어서 bedrooms...가 나오고, bedrooms와 !!!가 붙어서 bedrooms!!!가 나옵니다. 파이썬이 보기에 이들은 전부 다른 단어로 인식합니다.
```py
if 'bedrooms' == 'bedrooms...': 
    print('이 둘은 같습니다.')
else:
    print('이 둘은 다릅니다.')
```
```
[output]
이 둘은 다릅니다.
```


<br>


**NLTK** 가 훨씬 섬세하게 동작한다는 것을 알 수 있습니다.


<br>




# 한국어 토큰화의 특징

**<span style="color:red">한국어 토큰화는 영어보다 자연어 처리가 훨씬 어렵습니다.</span>** 대표적으로 다음과 같은 이유가 있습니다.

1. 한국어는 교착어이다.
2. 한국어는 띄어쓰기가 잘 지켜지지 않는다.
3. 한국어는 어순이 그렇게 중요하지 않다.
4. 한자어라는 특성상 하나의 음절조차도 다른 의미를 가질 수 있다.
5. 주어가 손쉽게 생략된다.
6. 데이터가 영어에 비해 너무 부족하다.
7. 언어 특화 오픈 소스 부족
8. OpenAI, Meta의 언어 모델은 영어 위주이므로 한국어에 대한 성능은 상대적으로 저하.

<br>



영어는 **New York** 과 같은 합성어나 **he's** 와 같이 줄임말에 대한 예외처리만 한다면, 띄어쓰기(whitespace)를 기준으로 하는 띄어쓰기 토큰화를 수행해도 단어 토큰화가 잘 작동합니다. **<span style="background-color: #fff5b1">하지만 한국어는 영어와는 달리 띄어쓰기만으로는 토큰화를 하기에 부족</span>** 합니다.



**<u>한국어의 경우에는 띄어쓰기 단위가 되는 단위</u>** 를 **<span style="color:red">'어절'</span>** 이라고 하는데 어절 토큰화와 단어 토큰화는 같지 않기 때문에, **<span style="color:red">어절 토큰화는 한국어 NLP에서 지양</span>** 되고 있습니다. 그 근본적인 이유는 **<span style="color:red">한국어</span>** 가 영어와는 다른 형태를 가지는 언어인 **<span style="color:red">교착어</span>** 라는 점에서 기인합니다. 한국어 토큰화가 어려운 점을 살펴보겠습니다.


<br>




## 1. 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않는다. -> 띄어쓰기 보정 : PyKoSpacing

**한국어는 영어권 언어와 비교하여 띄어쓰기가 어렵고 잘 지켜지지 않는 경향** 이 있습니다. 그 이유는 여러 견해가 있으나, 가장 기본적인 견해는 한국어의 경우 띄어쓰기가 지켜지지 않아도 글을 쉽게 이해할 수 있는 언어라는 점입니다. 대부분의 데이터에서 띄어쓰기가 잘 지켜지지 않는 경향이 있습니다.


띄어쓰기를 전혀 하지 않은 한국어와 영어 두 가지 경우를 봅시다.

- 제가이렇게띄어쓰기를전혀하지않고글을썼다고하더라도글을이해할수있습니다.
- Tobeornottobethatisthequestion

<br>


영어의 경우에는 띄어쓰기를 하지 않으면 손쉽게 알아보기 어려운 문장들이 생깁니다. 이는 한국어(모아쓰기 방식)와 영어(풀어쓰기 방식)라는 언어적 특성의 차이에 기인 하므로, 결론적으로 한국어는 수많은 코퍼스에서 띄어쓰기가 무시되는 경우가 많아 자연어 처리가 어려워졌다는 것입니다. **<span style="color:red">결국 띄어쓰기를 보정해주어야 하는 전처리가 필요할 수도 있습니다.</span>**

<br>

**<span style="color:red">PyKoSpacing</span>** 은 띄어쓰기가 되어있지 않은 문장을 띄어쓰기를 한 문장으로 변환해주는 딥러닝 기반의 패키지입니다. **PyKoSpacing** 은 대용량 코퍼스를 학습하여 만들어진 띄어쓰기 딥 러닝 모델로 준수한 성능을 가지고 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/3591ee1e-2783-40a9-8024-b7e516cb19d6">
</p>

<br>

```
pip install git+https://github.com/haven-jeon/PyKoSpacing.git
```
```py
sent = '김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'
```

<br>


임의의 문장을 임의로 띄어쓰기가 없는 문장으로 만들고, 이를 **PyKoSpacing** 의 입력으로 사용하여 원 문장과 비교해보겠습니다.
```py
new_sent = sent.replace(" ", '') # 띄어쓰기가 없는 문장 임의로 만들기
print(new_sent)
```
```
[output]
김철수는극중두인격의사나이이광수역을맡았다.철수는한국유일의태권도전승자를가리는결전의날을앞두고10년간함께훈련한사형인유연재(김광수분)를찾으러속세로내려온인물이다.
```


<br>


```py
from pykospacing import Spacing

spacing = Spacing()
kospacing_sent = spacing(new_sent) 

print(sent)
print(kospacing_sent)
```
```
[output]
김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.
김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.
```


<br>




## 2. 한국어는 주어 생략은 물론 어순도 중요하지 않다.

같은 의미의 문장을 다음과 같이 자유롭게 쓸수있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/7cbbd3fa-f6a3-4fd6-b8e8-10eb747fd77e">
</p>

<br>

3번의 경우 주어까지 생략했지만 의미를 알아차릴 수 있습니다. **<span style="color:red">즉, 다음 단어를 예측하는 Language Model에게는 매우 혼란스러운 상황입니다.</span>**

<br>




## 3. 한국어는 교착어이다.

**<span style="color:red">교착어</span>** 란 **<u>실질적인 의미를 가지는 어간에 조사나, 어미와 같은 문법 형태소들이 결합하여 문법적인 기능이 부여되는 언어</u>** 를 말합니다. 가령, 한국어에는 영어에 없는 '은, 는, 이, 가, 를' 등과 같은 **<span style="background-color: #fff5b1">조사</span>** 가 존재합니다. 예를 들어 한국어에 '그(he/him)' 라는 주어나 목적어가 들어간 문장이 있다고 할 때 이 경우, '그' 라는 단어 하나에도 '그가', '그에게', '그를', '그와', '그는'과 같이 다양한 조사가 '그' 라는 글자 뒤에 띄어쓰기 없이 바로 붙게됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/522939bb-7249-4a14-a871-a0268e332034">
</p>

<br>


이 예시문을 띄어쓰기 단위로 토큰화를 할 경우에는 '사과가', '사과에', '사과를', '사과' 가 전부 다른 단어로 간주됩니다. **<span style="color:red">즉, 같은 단어임에도 서로 다른 조사가 붙어서 다른 단어로 인식</span>** 이 되면 자연어 처리가 힘들고 번거로워지는 경우가 많으므로 **<span style="color:red">대부분의 한국어 NLP에서 조사는 분리해줄 필요가 있습니다.</span>**


띄어쓰기 단위가 영어처럼 독립적인 단어라면 띄어쓰기 단위로 토큰화를 하면 되겠지만 **<u>한국어는 어절이 독립적인 단어로 구성되는 것이 아니라, 조사 등의 무언가가 붙어있는 경우가 많아서 이를 전부 분리해줘야 한다는 의미</u>** 입니다.

<br>



**한국어 토큰화** 에서는 **형태소(morpheme)** 란 개념을 반드시 이해해야 합니다. **<span style="color:red">형태소(morpheme)</span>** 란 **<u>뜻을 가진 가장 작은 말의 단위</u>** 를 말합니다. 형태소에는 두 가지 형태소가 있는데 **자립 형태소** 와 **의존 형태소** 입니다.

- **자립 형태소**
    - 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소. 그 자체로 단어가 된다.
    - 체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사 등이 있다.
- **의존 형태소**
    - 다른 형태소와 결합하여 사용되는 형태소.
    - 접사, 어미, 조사, 어간을 말한다.

<br>


예를 들어 다음과 같은 문장에 **띄어쓰기 단위 토큰화** 를 수행한다면 다음과 같은 결과를 얻습니다.

- 문장 : 에디가 책을 읽었다
- 결과 : ['에디가', '책을', '읽었다']

<br>



하지만 이를 **<span style="color:red">형태소 단위로 분해</span>** 하면 다음과 같습니다.

- 자립 형태소 : 에디, 책
- 의존 형태소 : -가, -을, 읽-, -었, -다

<br>



'에디'라는 사람 이름과 '책'이라는 명사를 얻어낼 수 있습니다. 이를 통해 유추할 수 있는 것은 **<span style="color:red">한국어</span>** 에서 영어에서의 단어 토큰화와 유사한 형태를 얻으려면 **<span style="color:red">어절 토큰화가 아니라 형태소 토큰화를 수행해야</span>** 한다는 것입니다. **<u>교착어인 한국어의 특성으로 인해 한국어는 토크나이저로 형태소 분석기를 사용하는 것이 보편적</u>** 입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/9e558e92-5540-45bf-9ddf-4aa3a2e89b36">
</p>

<br>



한국어 자연어 처리를 위해서는 **<span style="color:red">KoNLPy(코엔엘파이)</span>** 라는 파이썬 패키지를 사용할 수 있습니다. 코엔엘파이를 통해서 사용할 수 있는 한국어 형태소 분석기로 **Okt(Open Korea Text)**, **메캅(Mecab)**, **코모란(Komoran)**, **한나눔(Hannanum)**, **꼬꼬마(Kkma)**, **Khaii**, **Soynlp** 등이 있습니다. 다양한 형태소 분석기가 존재하므로 원하는 Task에 맞는 형태소 분석기를 선택하면됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/c21adff9-5110-4969-a3a7-bc35377ed4ab">
</p>

<br>





# 한국어 Word Tokenization(KoNLPy)

## 1. 띄어쓰기를 기준으로 하는 단어 토큰화(가급적 하지마세요)


영어는 띄어쓰기 단위로 토큰화를 해도 단어들 간 구분이 꽤나 명확한 편이지만, 한국어의 경우에는 토큰화 작업이 훨씬 까다롭습니다. 그 이유는 **한국어는 조사, 접사 등으로 인해 단순 띄어쓰기 단위로 나누면 같은 단어가 다른 단어로 인식되는 경우가 너무 너무 많기 때문** 입니다. **<span style="color:red">한국어는 띄어쓰기로 토큰화하는 것은 명확한 실험 목적이 없다면 거의 쓰지 않는 것이 좋습니다.</span>** 예시를 통해서 이해해봅시다.
```py
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
print(kor_text.split())
```
```
[output]
['사과의', '놀라운', '효능이라는', '글을', '봤어.', '그래서', '오늘', '사과를', '먹으려고', '했는데', '사과가', '썩어서', '슈퍼에', '가서', '사과랑', '오렌지', '사왔어']
```


<br>


위의 예제에서는 '사과'란 단어가 총 4번 등장했는데 모두 '의', '를', '가', '랑' 등이 붙어있어 이를 제거해주지 않으면 기계는 전부 다른 단어로 인식하게 됩니다.
```py
print('사과' == '사과의')
print('사과의' == '사과를')
print('사과를' == '사과가')
print('사과가' == '사과랑')
```
```
[output]
False
False
False
False
```


<br>




## 2. 형태소 분석기 KoNLPy 설치

단어 토큰화를 위해서 영어에 **NLTK** 가 있다면 **한국어에는 형태소 분석기 패키지** 인 **<span style="color:red">KoNLPy(코엔엘파이)</span>** 가 존재합니다.
```
pip install konlpy
```



<br>


**NLTK** 도 내부적으로 여러 토크나이저가 있던 것처럼 **KoNLPy** 또한 **다양한 형태소 분석기** 를 가지고 있습니다. 또한 **<span style="color:red">Mecab</span>** 이라는 형태소 분석기는 특이하게도 별도 설치를 해주어야 합니다.
```
# Colab에 Mecab 설치
!pip install konlpy
!pip install mecab-python
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```


<br>


```py
from konlpy.tag import *

hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
okt = Okt()
mecab = Mecab()
```


<br>

위 형태소 분석기들은 공통적으로 아래의 함수를 제공합니다.

- **nouns** : 명사 추출
- **morphs** : 형태소 추출
- **pos** : 품사 부착


<br>



## 3. KoNLPy - 형태소 분석기 Okt

```py
print('Okt 명사 추출 :', okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('Okt 형태소 분석 :', okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('Okt 품사 태깅 :', okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```
```
[output]
Okt 명사 추출 : ['코딩', '당신', '연휴', '여행']
Okt 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
Okt 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
```


<br>


## 4. KoNLPy - 형태소 분석기 꼬꼬마

```py
print('kkma 명사 추출 :', kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('kkma 형태소 분석 :', kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('kkma 품사 태깅 :', kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```
```
[output]
kkma 명사 추출 : ['코딩', '당신', '연휴', '여행']
kkma 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
kkma 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
```


<br>



## 5. KoNLPy - 형태소 분석기 코모란

```py
print('komoran 명사 추출 :', komoran.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('komoran 형태소 분석 :', komoran.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('komoran 품사 태깅 :', komoran.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```
```
[output]
komoran 명사 추출 : ['코', '당신', '연휴', '여행']
komoran 형태소 분석 : ['열심히', '코', '딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가', '아', '보', '아요']
komoran 품사 태깅 : [('열심히', 'MAG'), ('코', 'NNG'), ('딩', 'MAG'), ('하', 'XSV'), ('ㄴ', 'ETM'), ('당신', 'NNP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKB'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가', 'VV'), ('아', 'EC'), ('보', 'VX'), ('아요', 'EC')]
```


<br>




## 6. KoNLPy - 형태소 분석기 한나눔

```py
print('hannanum 명사 추출 :', hannanum.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('hannanum 형태소 분석 :', hannanum.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('hannanum 품사 태깅 :', hannanum.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```
```
[output]
hannanum 명사 추출 : ['코딩', '당신', '연휴', '여행']
hannanum 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에는', '여행', '을', '가', '아', '보', '아']
hannanum 품사 태깅 : [('열심히', 'M'), ('코딩', 'N'), ('하', 'X'), ('ㄴ', 'E'), ('당신', 'N'), (',', 'S'), ('연휴', 'N'), ('에는', 'J'), ('여행', 'N'), ('을', 'J'), ('가', 'P'), ('아', 'E'), ('보', 'P'), ('아', 'E')]
```


<br>



## 7. KoNLPy - 형태소 분석기 Mecab

```py
print('mecab 명사 추출 :', mecab.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('mecab 형태소 분석 :', mecab.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('mecab 품사 태깅 :', mecab.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```
```
[output]
mecab 명사 추출 : ['코딩', '당신', '연휴', '여행']
mecab 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에', '는', '여행', '을', '가', '봐요']
mecab 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('한', 'XSA+ETM'), ('당신', 'NP'), (',', 'SC'), ('연휴', 'NNG'), ('에', 'JKB'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가', 'VV'), ('봐요', 'EC+VX+EC')]
```


<br>



'못' 이라는 단어는 명사일때도있고, 부사일때도 있습니다.
```py
print('mecab 명사 추출 :', mecab.nouns("망치로 못을 두드리다."))
print('mecab 명사 추출 :', mecab.nouns("나 그 일 못해요."))
```
```
[output]
mecab 명사 추출 : ['망치', '못']
mecab 명사 추출 : ['나', '일']
```


<br>


각 형태소 분석기는 성능과 결과가 다르게 나오기 때문에, **형태소 분석기의 선택은 사용하고자 하는 필요 용도에 어떤 형태소 분석기가 가장 적절한지를 판단하고 사용하면 됩니다.** 예를 들어서 속도를 중시한다면 메캅을 사용할 수 있습니다.



<br>



# 문장 토큰화(Sentence Tokenization)

**<span style="color:red">토큰의 단위가 문장(sentence)</span>** 인 문장 단위로 토큰화할 필요가 있는 경우가 있습니다. 보통 갖고있는 코퍼스가 정제되지 않은 상태라면, 코퍼스는 문장 단위로 구분되어 있지 않아서 이를 사용하고자 하는 용도에 맞게 문장 토큰화가 필요할 수 있습니다.


직관적으로 생각해봤을 때 **온점(.) 이나 '!' 나 '?' 로 구분하면 되지 않을까?** 란 **<span style="color:red">착각</span>** 을 할 수 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/70797153-1ea0-433c-b7e0-f529663a8114">
</p>

<br>


**<u>! 나 ? 는 문장의 구분을 위한 꽤 명확한 구분자(boundary) 역할을 하지만 마침표는 그렇지 않기 때문</u>** 입니다. 마침표는 문장의 끝이 아니더라도 등장할 수 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/e165d05f-f265-4c4c-8acb-60f4c23fb3d2">
</p>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/49d0cbb9-30eb-4521-be50-c5c9976a8cf9">
</p>

<br>



**<span style="color:red">제대로 된 결과라면 Ph.D.는 분리되어서는 안되며, 문장 구분조차 보다 섬세한 규칙이 필요합니다.</span>**


<br>



# 영어 Sentence Tokenization(NLTK)

우선 문자열이 주어졌을 떄, 문장 단위로 나눠보겠습니다. **문자열.split('자르는 기준')** 을 사용하면 해당 기준으로 문자열들을 분리하여 리스트 형태로 반환합니다. 아래의 코드는 온점을 기준으로 문자열을 자르는 코드입니다.
```py
temp = 'Yonsei University is a private research university in Seoul, South Korea. Yonsei University is deemed as one of the three most prestigious institutions in the country. It is particularly respected in the studies of medicine and business administration.'
temp.split('. ')
```
```
[output]
['Yonsei University is a private research university in Seoul, South Korea',
 'Yonsei University is deemed as one of the three most prestigious institutions in the country',
 'It is particularly respected in the studies of medicine and business administration.']
```


<br>



직관적으로 생각해봤을 때는 ?나 온점(.)이나 ! 기준으로 문장을 잘라내면 되지 않을까라고 생각할 수 있지만, 꼭 그렇지만은 않습니다. !나 ?는 문장의 구분을 위한 꽤 명확한 구분자(boundary) 역할을 하지만 온점은 꼭 그렇지 않기 때문입니다. **<span style="color:red">다시 말해, 온점은 문장의 끝이 아니더라도 등장할 수 있습니다. 온점을 기준으로 문장을 구분할 경우에는 예외사항이 너무 많습니다.</span>**



**NLTK** 에서는 영어 문장의 토큰화를 수행하는 **sent_tokenize** 를 지원하고 있습니다.
```py
text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
from nltk.tokenize import sent_tokenize
print(sent_tokenize(text))
```
```
[output]
['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to mae sure no one was near.']
```


<br>

```py
text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))
```
```
[output]
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```


<br>





# 한국어 Sentence Tokenization(KSS)


한국어 문장 토크나이저 라이브러리로는 대표적으로 **KSS** 와 **kiwi** 가 있습니다.
```
pip install kss
```
```py
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 이제 해보면 알걸요?'
print(kss.split_sentences(text))
```
```
[output]
['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어려워요.', '이제 해보면 알걸요?']
```





