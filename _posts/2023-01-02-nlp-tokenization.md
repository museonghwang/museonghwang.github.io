---
layout: post
title: Tokenization 개념과 영어 및 한국어 특성
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





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



## 띄어쓰기를 기준으로 하는 단어 토큰화(잘 되는 것 같아도 하지마세요)

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



영어는 **New York** 과 같은 합성어나 **he's** 와 같이 줄임말에 대한 예외처리만 한다면, 띄어쓰기(whitespace)를 기준으로 하는 띄어쓰기 토큰화를 수행해도 단어 토큰화가 잘 작동합니다. **<u>하지만 한국어는 영어와는 달리 띄어쓰기만으로는 토큰화를 하기에 부족합니다.</u>**



한국어의 경우에는 띄어쓰기 단위가 되는 단위를 '어절'이라고 하는데 어절 토큰화는 한국어 NLP에서 지양되고 있습니다. 어절 토큰화와 단어 토큰화는 같지 않기 때문입니다. 그 근본적인 이유는 한국어가 영어와는 다른 형태를 가지는 언어인 교착어라는 점에서 기인합니다. 교착어란 조사, 어미 등을 붙여서 말을 만드는 언어를 말합니다.












