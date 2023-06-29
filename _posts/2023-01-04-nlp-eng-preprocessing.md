---
layout: post
title: IDMB 리뷰 데이터를 이용한 전처리(토큰화, 정수 인코딩, 패딩)
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





IDMB 리뷰 데이터를 이용하여 전처리 및 간단한 EDA와 정수 인코딩, 그리고 패딩 과정까지 진행해보겠습니다. IDMB 리뷰 데이터는 리뷰에 대한 텍스트와 해당 리뷰가 긍정인 경우 1을 부정인 경우 0으로 표시한 레이블로 구성된 데이터입니다.

- [데이터 다운로드 링크](https://github.com/ukairia777/pytorch-nlp-tutorial/tree/main/10.%20RNN%20Text%20Classification/dataset)

<br>



실습을 진행하기 전에 간단하게 자연어 전처리 과정을 확인하고나서 실습을 진행하겠습니다.

<br>



1. Text Preprocessing
2. IDMB 리뷰 데이터 다운로드
3. 데이터 개수, 중복값, Null 확인
4. 단어 토큰화
5. 토큰화기반 통계적 전처리
6. 정수 인코딩
7. 패딩

<br>
<br>






# 1. Text Preprocessing

기계에게는 단어와 문장의 경계를 알려주어야 하는데 이를 위해서 특정 단위로 토큰화 또는 토크나이징을 해줍니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/d6b4ae2f-3e3d-42a6-959d-85b699a42e6d">
</p>

<br>


기계가 알고있는 단어들의 집합을 단어 집합(Vocabulary)이라고 합니다. 단어 집합이란 훈련 데이터에 있는 단어들의 중복을 제거한 집합을 의미합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/5fd0950a-8f9a-4cc0-bcff-efc8f6569d8c">
</p>

<br>


단어 집합에 있는 각 단어에는 고유한 정수가 부여됩니다. 단어 집합을 기반으로 하므로 중복은 허용되지 않습니다. 이는 앞으로 입력된 모든 텍스트를 정수 시퀀스로 변환하기 위함입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/52d2a6e1-01cb-4839-80f2-0f1a10321729">
</p>

<br>


정수 인코딩을 진행하면 다음과 같습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/e8171418-7796-4b50-a09b-fe060429e616">
</p>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/8ee87287-9fbe-40c8-a7b1-ab06ae26c4a9">
</p>

<br>



만약 단어 집합에 없는 단어로 인해 생기는 문제를 OOV 문제라고 합니다. 이렇게 생긴 단어들을 일괄적으로 하나의 토큰으로 맵핑해주기도 합니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/1bc52522-4a92-4690-97d2-c3efd2bab541">
</p>

<br>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/67155b13-a9d4-4d78-affd-dc331687d926">
</p>

<br>


단어 집합에 있는 각 단어에는 고유한 정수가 부여됩니다. 이는 앞으로 입력된 모든 텍스트를 정수 시퀀스로 변환하기 위함입니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/66d231d5-4cc8-4e52-8918-dded44c819ce">
</p>

<br>


여러 문장을 병렬적으로 처리하고 싶은 경우, 이를 하나의 행렬로 인식시켜줄 필요가 있습니다. 이때, 서로 다른 문장의 길이를 패딩을 통해 동일하게 만들어줄 수 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/a41a2e58-9f69-45d2-910a-d58ec75c56a0">
</p>

<br>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/766d5b51-6d03-4b80-b7f3-5d3d5c0ebec7">
</p>

<br>

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/54d51444-a995-49b1-a120-e5440493b4ea">
</p>

<br>






# 2. 네이버 영화 리뷰 데이터 다운로드

필요한 라이브러리와 네이버 영화 리뷰 데이터를 다운로드 하겠습니다.
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

import torch
import urllib.request
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/pytorch-nlp-tutorial/main/10.%20RNN%20Text%20Classification/dataset/IMDB%20Dataset.csv", filename="IMDB Dataset.csv")
```


<br>

**IMDB Dataset.csv** 가 다운로드 된 것을 볼 수 있습니다.

<br>





# 3. 데이터 개수, 중복값, Null 확인

데이터를 읽고 **간단한 EDA** 를 진행하겠습니다.
```py
df = pd.read_csv('IMDB Dataset.csv')
df
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/e1d80027-f931-4b52-bb72-657b054a358d">
</p>

<br>


결측값과 label의 분포와 개수를 확인해보겠습니다.
```py
df.info()
```
```
[output]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 50000 entries, 0 to 49999
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   review     50000 non-null  object
 1   sentiment  50000 non-null  object
dtypes: object(2)
memory usage: 781.4+ KB
```


<br>

```py
print('결측값 여부 :',df.isnull().values.any())
```
```
[output]
결측값 여부 : False
```


<br>

```py
df['sentiment'].value_counts().plot(kind='bar')
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/9fbc8048-a9ff-4162-b03e-b81a7b601ef1">
</p>

<br>


```py
print('레이블 개수')
print(df.groupby('sentiment').size().reset_index(name='count'))
```
```
[output]
레이블 개수
  sentiment  count
0  negative  25000
1  positive  25000
```


<br>





label의 분포는 같으며, label은 각각 25000개를 가지고 있습니다. label 값을 정수로 바꾸겠습니다. 즉 positive를 1로, negative를 0으로 변경하겠습니다.
```py
df['sentiment'] = df['sentiment'].replace(['positive','negative'],[1, 0])
df.head()
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/1778ed2b-c403-471c-8f95-f9e7758beba9">
</p>


<br>



이제 영화 리뷰의 개수, 레이블의 개수 각각을 확인해보고 훈련 데이터와 테스트 데이터로 분리하겠습니다.
```py
X_data = df['review']
y_data = df['sentiment']
print('영화 리뷰의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))
```
```
[output]
영화 리뷰의 개수: 50000
레이블의 개수: 50000
```


<br>

```py
X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.5,
    random_state=0,
    stratify=y_data
)

print('--------훈련 데이터의 비율-----------')
print(f'긍정 리뷰 = {round(y_train.value_counts()[0]/len(y_train) * 100,3)}%')
print(f'부정 리뷰 = {round(y_train.value_counts()[1]/len(y_train) * 100,3)}%')
print('--------테스트 데이터의 비율-----------')
print(f'긍정 리뷰 = {round(y_test.value_counts()[0]/len(y_test) * 100,3)}%')
print(f'부정 리뷰 = {round(y_test.value_counts()[1]/len(y_test) * 100,3)}%')
```
```
[output]
--------훈련 데이터의 비율-----------
긍정 리뷰 = 50.0%
부정 리뷰 = 50.0%
--------테스트 데이터의 비율-----------
긍정 리뷰 = 50.0%
부정 리뷰 = 50.0%
```


<br>






# 4. 단어 토큰화

전처리된 데이터를 바탕으로 **nltk** 의 **word_tokenize** 를 이용해서 **단어 토큰화** 를 진행하겠습니다. 우선 샘플 하나를 가지고 진행하겠습니다.
```py
print(X_train[0])
```
```
[output]
One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. 

... (중략) ...

Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side.
```


<br>

```py
sample = word_tokenize(X_train[0])
print(sample)
```
```
[output]
['One', 'of', 'the', 'other', 'reviewers', 'has', 'mentioned', 'that', 'after', 'watching', 'just', '1', 'Oz', 'episode', 'you', "'ll", 'be', 'hooked', '.', 'They', 'are', 'right', ',', 'as', 'this', 'is', 'exactly', 'what', 'happened', 'with', 'me.', '<', 'br', '/', '>', '<', 'br', '/', '>', 'The', 'first', 'thing', 'that', 'struck', 'me', 'about', 'Oz', 'was', 'its', 'brutality', 'and', 'unflinching', 'scenes', 'of', 'violence', ',', 'which', 'set', 'in', 'right', 'from', 'the', 'word', 'GO', '.',

... (중략) ... ,

'Not', 'just', 'violence', ',', 'but', 'injustice', '(', 'crooked', 'guards', 'who', "'ll", 'be', 'sold', 'out', 'for', 'a', 'nickel', ',', 'inmates', 'who', "'ll", 'kill', 'on', 'order', 'and', 'get', 'away', 'with', 'it', ',', 'well', 'mannered', ',', 'middle', 'class', 'inmates', 'being', 'turned', 'into', 'prison', 'bitches', 'due', 'to', 'their', 'lack', 'of', 'street', 'skills', 'or', 'prison', 'experience', ')', 'Watching', 'Oz', ',', 'you', 'may', 'become', 'comfortable', 'with', 'what', 'is', 'uncomfortable', 'viewing', '....', 'thats', 'if', 'you', 'can', 'get', 'in', 'touch', 'with', 'your', 'darker', 'side', '.']
```


<br>


단어 토큰화가 된 데이터에 대해 소문자화 시키겠습니다.
```py
lower_sample = [word.lower() for word in sample]
print(lower_sample)
```
```
[output]
['one', 'of', 'the', 'other', 'reviewers', 'has', 'mentioned', 'that', 'after', 'watching', 'just', '1', 'oz', 'episode', 'you', "'ll", 'be', 'hooked', '.', 'they', 'are', 'right', ',', 'as', 'this', 'is', 'exactly', 'what', 'happened', 'with', 'me.', '<', 'br', '/', '>', '<', 'br', '/', '>', 'the', 'first', 'thing', 'that', 'struck', 'me', 'about', 'oz', 'was', 'its', 'brutality', 'and', 'unflinching', 'scenes', 'of', 'violence', ',', 'which', 'set', 'in', 'right', 'from', 'the', 'word', 'go', '.',

... (중략) ... ,

'not', 'just', 'violence', ',', 'but', 'injustice', '(', 'crooked', 'guards', 'who', "'ll", 'be', 'sold', 'out', 'for', 'a', 'nickel', ',', 'inmates', 'who', "'ll", 'kill', 'on', 'order', 'and', 'get', 'away', 'with', 'it', ',', 'well', 'mannered', ',', 'middle', 'class', 'inmates', 'being', 'turned', 'into', 'prison', 'bitches', 'due', 'to', 'their', 'lack', 'of', 'street', 'skills', 'or', 'prison', 'experience', ')', 'watching', 'oz', ',', 'you', 'may', 'become', 'comfortable', 'with', 'what', 'is', 'uncomfortable', 'viewing', '....', 'thats', 'if', 'you', 'can', 'get', 'in', 'touch', 'with', 'your', 'darker', 'side', '.']
```


<br>




지금부터 train 및 test 데이터에 대한 단어 토큰화를 진행하겠습니다.
```py
def tokenize(sentences):
    tokenized_sentences = []
    for sent in tqdm(sentences):
        tokenized_sent = word_tokenize(sent)
        tokenized_sent = [word.lower() for word in tokenized_sent]
        tokenized_sentences.append(tokenized_sent)
    return tokenized_sentences

tokenized_X_train = tokenize(X_train)
tokenized_X_test = tokenize(X_test)
```
```
[output]
100%|██████████| 25000/25000 [00:35<00:00, 713.44it/s]
100%|██████████| 25000/25000 [00:34<00:00, 722.73it/s]
```


<br>

```py
# 상위 샘플 2개 출력
for sent in tokenized_X_train[:2]:
    print(sent)
```
```
[output]
['life', 'is', 'too', 'short', 'to', 'waste', 'on', 'two', 'hours', 'of', 'hollywood', 'nonsense', 'like', 'this', ',', 'unless', 'you', "'re", 'a', 'clueless', 'naiive', '16', 'year', 'old', 'girl', 'with', 'no', 'sense', 'of', 'reality', 'and', 'nothing', 'better', 'to', 'do', '.', 'dull', 'characters', ',', 'poor', 'acting', '(', 'artificial', 'emotion', ')', ',', 'weak', 'story', ',', 'slow', 'pace', ',', 'and', 'most', 'important', 'to', 'this', 'films', 'flawed', 'existence-no', 'one', 'cares', 'about', 'the', 'overly', 'dramatic', 'relationship', '.']
['for', 'those', 'who', 'expect', 'documentaries', 'to', 'be', 'objective', 'creatures', ',', 'let', 'me', 'give', 'you', 'a', 'little', 'lesson', 'in', 'american', 'film-making.', '<', 'br', '/', '>', '<', 'br', '/', '>', 'documentaries', 'rely', 'heavily', 'on', 'casting', '.', 'you', 'pick', 'and', 'choose', 'characters', 'you', 'think', 'will', 'enhance', 'the', 'drama', 'and', 'entertainment', 'value', 'of', 'your', 'film.', '<', 'br', '/', '>', '<', 'br', '/', '>', 'after', 'you', 'have', 'shot', 'a', 'ton', 'of', 'footage', ',', 'you', 'splice', 'it', 'together', 'to', 'make', 'a', 'film', 'with', 'ups', 'and', 'downs', ',', 'turning', 'points', ',', 'climaxes', ',', 'etc', '.', 'if', 'you', 'have', 'trouble', 'with', 'existing', 'footage', ',', 'you', 'either', 'shoot', 'some', 'more', 'that', 'makes', 'sense', ',', 'find', 'some', 'stock', 'footage', ',', 'or', 'be', 'clever', 'with', 'your', 'narration.', '<', 'br', '/', '>', '<', 'br', '/', '>', 'the', 'allegation', 'that', 'the', 'filmmakers', 'used', 'footage', 'of', 'locales', 'not', 'part', 'of', 'the', 'movie', '(', 'favelas', 'next', 'to', 'beautiful', 'beaches', ')', 'does', 'not', 'detract', 'from', 'the', 'value', 'of', 'the', 'film', 'as', 'a', 'dramatic', 'piece', 'and', 'the', 'particular', 'image', 'is', 'one', 'that', 'resonates', 'enough', 'to', 'justify', 'its', 'not-quite-truthful', 'inclusion', '.', 'at', 'any', 'rate', ',', 'you', 'use', 'the', 'footage', 'you', 'can', '.', 'so', 'they', 'did', "n't", 'happen', 'to', 'have', 'police', 'violence', 'footage', 'for', 'that', 'particular', 'neighborhood', '.', 'does', 'this', 'mean', 'not', 'include', 'it', 'and', 'just', 'talk', 'about', 'it', 'or', 'maybe', 'put', 'in', 'some', 'cartoon', 'animation', 'so', 'the', 'audience', 'is', "n't", '``', 'duped', "''", '?', 'um', ',', 'no.', '<', 'br', '/', '>', '<', 'br', '/', '>', 'as', 'for', 'the', 'hopeful', 'ending', ',', 'why', 'not', '?', 'yes', ',', 'americans', 'made', 'it', '.', 'yes', ',', 'americans', 'are', 'optimistic', 'bastards', '.', 'but', 'why', 'end', 'on', 'a', 'down', 'note', '?', 'just', 'because', 'it', "'s", 'set', 'in', 'a', 'foreign', 'country', 'and', 'foreign', 'films', 'by', 'and', 'large', 'end', 'on', 'a', 'down', 'note', '?', 'let', 'foreigners', 'portray', 'the', 'dismal', 'outlook', 'of', 'life.', '<', 'br', '/', '>', '<', 'br', '/', '>', 'let', 'us', 'americans', 'think', 'there', 'may', 'be', 'a', 'happy', 'ending', 'looming', 'in', 'the', 'future', '.', 'there', 'just', 'may', 'be', 'one', '.']
```


<br>



train 및 test 데이터에 대한 단어 토큰화가 완료되었습니다. 총 단어의 개수 및 각 단어의 등장 횟수를 확인해보겠습니다.
```py
word_list = []
for sent in tokenized_X_train:
    for word in sent:
        word_list.append(word)

word_counts = Counter(word_list)
print('총 단어수 :', len(word_counts))
```
```
[output]
총 단어수 : 112946
```


<br>

```py
print(word_counts)
```
```
[output]
Counter({'the': 332140, ',': 271720, '.': 234036, 'and': 161143, 'a': 161005, 'of': 144426, 'to': 133327, 'is': 107917, ... (중략) ..., 'pseudo-film': 1, "'harris": 1, 'bridesmaid': 1, 'infatuations': 1, 'a.p': 1})

```


<br>

```py
print('훈련 데이터에서의 단어 the의 등장 횟수 :', word_counts['the'])
print('훈련 데이터에서의 단어 love의 등장 횟수 :', word_counts['love'])
```
```
[output]
훈련 데이터에서의 단어 the의 등장 횟수 : 332140
훈련 데이터에서의 단어 love의 등장 횟수 : 6260
```


<br>





# 5. 토큰화기반 통계적 전처리

단어사전에 대해 등장 빈도수 상위 10개 단어를 추출해보고 **통계적 수치** 를 계산하겠습니다.
```py
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print('등장 빈도수 상위 10개 단어')
print(vocab[:10])
```
```
[output]
등장 빈도수 상위 10개 단어
['the', ',', '.', 'and', 'a', 'of', 'to', 'is', '/', '>']
```


<br>


```py
threshold = 3
total_cnt = len(word_counts) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```
```
[output]
단어 집합(vocabulary)의 크기 : 112946
등장 빈도가 2번 이하인 희귀 단어의 수: 69670
단어 집합에서 희귀 단어의 비율: 61.68434473111044
전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 1.1946121064938062
```


<br>

등장 빈도가 threshold 값인 3회 미만. 즉, 2회 이하인 단어들은 단어 집합에서 무려 69% 이상을 차지합니다. 하지만, 실제로 훈련 데이터에서 등장 빈도로 차지하는 비중은 상대적으로 매우 적은 수치인 1.19%밖에 되지 않습니다. 아무래도 등장 빈도가 2회 이하인 단어들은 자연어 처리에서 별로 중요하지 않을 듯 합니다. 그래서 이 단어들은 정수 인코딩 과정에서 배제시키겠습니다.


등장 빈도수가 2이하인 단어들의 수를 제외한 **단어의 개수를 단어 집합의 최대 크기로 제한** 하겠습니다.
```py
# 전체 단어 개수 중 빈도수 1이하인 단어는 제거.
vocab_size = total_cnt - rare_cnt
vocab = vocab[:vocab_size]
print('단어 집합의 크기 :', len(vocab))
```
```
[output]
단어 집합의 크기 : 43276
```


<br>


```py
print(vocab)
```
```
[output]
['the', ',', '.', 'and', 'a', 'of', 'to', 'is', ... (중략) ..., '-atlantis-', 'middlemarch', 'lollo']
```


<br>


단어 집합의 크기는 112946개에서 등장 빈도수가 2이하인 단어들의 수를 제외한 뒤 확인해보니 43276개 였습니다. 정제된 단어 집합에 특수 토큰인 **PAD**, **UNK** 를 삽입하고 최종 단어 집합을 구성하겠습니다.
```py
word_to_index = {}
word_to_index['<PAD>'] = 0
word_to_index['<UNK>'] = 1

for index, word in enumerate(vocab):
    word_to_index[word] = index + 2

print(word_to_index)
```
```
[output]
{'<PAD>': 0, '<UNK>': 1, 'the': 2, ',': 3, '.': 4, 'and': 5, 'a': 6, 'of': 7, 'to': 8, 'is': 9, ... (중략) ..., '-atlantis-': 43277, 'middlemarch': 43278, 'lollo': 43279}
```


<br>


```py
vocab_size = len(word_to_index)
print('패딩 토큰과 UNK 토큰을 고려한 단어 집합의 크기 :', vocab_size)
```
```
[output]
패딩 토큰과 UNK 토큰을 고려한 단어 집합의 크기 : 43278
```


<br>



최종적으로 전처리된 단어 집합의 크기는 특수 토큰인 PAD, UNK를 포함한 개수인 43278개 입니다. 단어에 대한 index를 조회해보겠습니다.
```py
print('단어 <PAD>와 맵핑되는 정수 :', word_to_index['<PAD>'])
print('단어 <UNK>와 맵핑되는 정수 :', word_to_index['<UNK>'])
print('단어 the와 맵핑되는 정수 :', word_to_index['the'])
```
```
[output]
단어 <PAD>와 맵핑되는 정수 : 0
단어 <UNK>와 맵핑되는 정수 : 1
단어 the와 맵핑되는 정수 : 2
```


<br>


```py
word_to_index['bridesmaid']
```
```
[output]
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-108-a8fb05ad2c9a> in <module>
----> 1 word_to_index['bridesmaid']

KeyError: 'bridesmaid'
```


<br>

단어 사전에 없는 단어인 'bridesmaid'를 조회하면 오류가 납니다.

<br>





# 6. 정수 인코딩

지금부터 완성된 단어 사전에 대해 **정수 인코딩** 을 진행하겠습니다.
```py
def texts_to_sequences(tokenized_X_data, word_to_index):
    encoded_X_data = []
    for sent in tokenized_X_data:
        index_sequences = []
        for word in sent:
            try:
                index_sequences.append(word_to_index[word])
            except KeyError:
                index_sequences.append(word_to_index['<UNK>'])
        encoded_X_data.append(index_sequences)
        
    return encoded_X_data

encoded_X_train = texts_to_sequences(X_train, word_to_index)
encoded_X_test = texts_to_sequences(X_test, word_to_index)

print('토큰화 전 원본 문장 :', X_train[42043])
print('정수 인코딩 전 토큰화 :', tokenized_X_train[0])
print('정수 인코딩 결과 :', encoded_X_train[0])
```
```
[output]
토큰화 전 원본 문장 : Life is too short to waste on two hours of Hollywood nonsense like this, unless you're a clueless naiive 16 year old girl with no sense of reality and nothing better to do. Dull characters, poor acting (artificial emotion), weak story, slow pace, and most important to this films flawed existence-no one cares about the overly dramatic relationship.
정수 인코딩 전 토큰화 : ['life', 'is', 'too', 'short', 'to', 'waste', 'on', 'two', 'hours', 'of', 'hollywood', 'nonsense', 'like', 'this', ',', 'unless', 'you', "'re", 'a', 'clueless', 'naiive', '16', 'year', 'old', 'girl', 'with', 'no', 'sense', 'of', 'reality', 'and', 'nothing', 'better', 'to', 'do', '.', 'dull', 'characters', ',', 'poor', 'acting', '(', 'artificial', 'emotion', ')', ',', 'weak', 'story', ',', 'slow', 'pace', ',', 'and', 'most', 'important', 'to', 'this', 'films', 'flawed', 'existence-no', 'one', 'cares', 'about', 'the', 'overly', 'dramatic', 'relationship', '.']
정수 인코딩 결과 : [139, 9, 117, 353, 8, 459, 30, 129, 635, 7, 360, 1934, 50, 17, 3, 898, 29, 192, 6, 5485, 1, 4041, 346, 188, 261, 22, 72, 307, 7, 605, 5, 176, 143, 8, 54, 4, 772, 119, 3, 351, 132, 28, 4786, 1386, 27, 3, 838, 81, 3, 617, 1057, 3, 5, 104, 681, 8, 17, 123, 2958, 1, 40, 1994, 57, 2, 2346, 950, 632, 4]
```


<br>

정수 인코딩이 완료되었습니다. 정수 디코딩을 하여 기존 첫번째 샘플을 복원해보겠습니다.
```py
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

decoded_sample = [index_to_word[word] for word in encoded_X_train[0]]
print('기존의 첫번째 샘플 :', tokenized_X_train[0])
print('복원된 첫번째 샘플 :', decoded_sample)
```
```
[output]
기존의 첫번째 샘플 : ['life', 'is', 'too', 'short', 'to', 'waste', 'on', 'two', 'hours', 'of', 'hollywood', 'nonsense', 'like', 'this', ',', 'unless', 'you', "'re", 'a', 'clueless', 'naiive', '16', 'year', 'old', 'girl', 'with', 'no', 'sense', 'of', 'reality', 'and', 'nothing', 'better', 'to', 'do', '.', 'dull', 'characters', ',', 'poor', 'acting', '(', 'artificial', 'emotion', ')', ',', 'weak', 'story', ',', 'slow', 'pace', ',', 'and', 'most', 'important', 'to', 'this', 'films', 'flawed', 'existence-no', 'one', 'cares', 'about', 'the', 'overly', 'dramatic', 'relationship', '.']
복원된 첫번째 샘플 : ['life', 'is', 'too', 'short', 'to', 'waste', 'on', 'two', 'hours', 'of', 'hollywood', 'nonsense', 'like', 'this', ',', 'unless', 'you', "'re", 'a', 'clueless', '<UNK>', '16', 'year', 'old', 'girl', 'with', 'no', 'sense', 'of', 'reality', 'and', 'nothing', 'better', 'to', 'do', '.', 'dull', 'characters', ',', 'poor', 'acting', '(', 'artificial', 'emotion', ')', ',', 'weak', 'story', ',', 'slow', 'pace', ',', 'and', 'most', 'important', 'to', 'this', 'films', 'flawed', '<UNK>', 'one', 'cares', 'about', 'the', 'overly', 'dramatic', 'relationship', '.']
```


<br>

실제로 인코딩을 의미하는 **word_to_index** 와 디코딩을 의미하는 **index_to_word** 를 살펴보면 다음과 같습니다.
```py
print(word_to_index)
print(index_to_word)
```
```
[output]
{'<PAD>': 0, '<UNK>': 1, 'the': 2, ',': 3, '.': 4, 'and': 5, 'a': 6, 'of': 7, 'to': 8, 'is': 9, ... (중략) ..., middlemarch: 43276', lollo: '43277'}
{0: '<PAD>', 1: '<UNK>', 2: 'the', 3: ',', 4: '.', 5: 'and', 6: 'a', 7: 'of', 8: 'to', 9: 'is', ... (중략) ..., 43276: middlemarch', 43277: 'lollo'}
```


<br>






# 7. 패딩

정수 인코딩이 완료된 리뷰 데이터의 최대 길이와 평균 길이를 구해보겠습니다.
```py
print('리뷰의 최대 길이 :', max(len(review) for review in encoded_X_train))
print('리뷰의 평균 길이 :', sum(map(len, encoded_X_train))/len(encoded_X_train))
plt.hist([len(review) for review in encoded_X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```
```
[output]
리뷰의 최대 길이 : 2818
리뷰의 평균 길이 : 279.0998
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/a84c2c9d-9d54-4fc3-9e4f-aa470299d07a">
</p>

<br>


전체 샘플 중 길이가 **maxlen** 이하인 샘플의 비율을 확인하고 **패딩** 을 진행하겠습니다.
```py
def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1

    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 500
below_threshold_len(max_len, encoded_X_train)
```
```
[output]
전체 샘플 중 길이가 500 이하인 샘플의 비율: 87.836
```


<br>



전체 샘플 중 길이가 500 이하인 샘플의 비율은 87.836% 입니다. 패딩을 진행하겠습니다.
```py
def pad_sequences(sentences, max_len):
    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[index, :len(sentence)] = np.array(sentence)[:max_len]

    return features

padded_X_train = pad_sequences(encoded_X_train, max_len=max_len)
padded_X_test = pad_sequences(encoded_X_test, max_len=max_len)

print('훈련 데이터의 크기 :', padded_X_train.shape)
print('테스트 데이터의 크기 :', padded_X_test.shape)
```
```
[output]
훈련 데이터의 크기 : (25000, 500)
테스트 데이터의 크기 : (25000, 500)
```


<br>


정상적으로 패딩이 완료되었는지 확인하겠습니다.
```py
padded_X_train[:5]
```
```
[output]
array([[ 139,    9,  117, ...,    0,    0,    0],
       [  23,  162,   47, ...,    0,    0,    0],
       [  23,   40,  347, ...,    0,    0,    0],
       [  17,   24,   20, ...,    0,    0,    0],
       [1827,   92,  134, ...,    0,    0,    0]])
```


<br>




구체적으로 1개의 데이터에 대한 패딩확인을 진행하겠습니다.
```py
len(padded_X_train[0])
```
```
[output]
500
```


<br>

```py
print(padded_X_train[0])
```
```
[output]
[ 139    9  117  353    8  459   30  129  635    7  360 1934   50   17
    3  898   29  192    6 5485    1 4041  346  188  261   22   72  307
    7  605    5  176  143    8   54    4  772  119    3  351  132   28
 4786 1386   27    3  838   81    3  617 1057    3    5  104  681    8
   17  123 2958    1   40 1994   57    2 2346  950  632    4    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0]
```


<br>



정상적으로 모든 전처리가 완료되었습니다.





