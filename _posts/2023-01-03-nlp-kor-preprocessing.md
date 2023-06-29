---
layout: post
title: 
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





네이버 영화 리뷰 데이터를 이용하여 전처리 및 간단한 EDA와 정수 인코딩, 그리고 패딩 과정까지 진행해보겠습니다. 네이버 영화 리뷰 데이터는 총 200,000개 리뷰로 구성된 데이터로 영화 리뷰에 대한 텍스트와 해당 리뷰가 긍정인 경우 1, 부정인 경우 0을 표시한 레이블로 구성되어져 있습니다.

- [데이터 다운로드 링크](https://github.com/e9t/nsmc/)

<br>



실습을 진행하기 전에 간단하게 자연어 전처리 과정을 확인하고나서 실습을 진행하겠습니다.

<br>



1. Text Preprocessing
2. 네이버 영화 리뷰 데이터 다운로드
3. 데이터 개수, 중복값, Null 확인
4. 정규표현식을 이용한 전처리
5. 단어 토큰화
6. 토큰화기반 통계적 전처리
7. 정수 인코딩
8. 패딩

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
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from collections import Counter

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
```


<br>

**ratings_train.txt** 와 **ratings_test.txt** 가 다운로드 된 것을 볼 수 있습니다.

<br>





# 3. 데이터 개수, 중복값, Null 확인

데이터를 읽고 **간단한 EDA** 를 진행하겠습니다. 이때 주의해서 볼 것은 **전처리 과정** 입니다.
```py
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

train_data.head()
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/59b1ba3f-1332-4a9b-87da-2184dbe367ff">
</p>

<br>


데이터를 간략하게 파악할 수 있습니다. 훈련용 리뷰 데이터 개수를 출력하고, **document 열 기준으로 중복값을 제거** 해보겠습니다.
```py
print('훈련용 리뷰 개수 :', len(train_data)) # 훈련용 리뷰 개수 출력

# document 열의 중복 제거
train_data.drop_duplicates(subset=['document'], inplace=True)

print('총 샘플의 수 :', len(train_data))
```
```
[output]
훈련용 리뷰 개수 : 150000
총 샘플의 수 : 146183
```


<br>

중복값 제거가 완료되었습니다. 이제 label의 분포와 개수를 확인해보겠습니다.
```py
train_data['label'].value_counts().plot(kind='bar')
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/fc9ecd5d-3171-46c3-8cf9-7b269abeccbf">
</p>

<br>

```py
print(train_data.groupby('label').size().reset_index(name='count'))
```
```
[output]
   label  count
0      0  73342
1      1  72841
```


<br>


label의 분포는 비슷하며, label 0은 73342개, label 1은 72841개를 가지고 있습니다. 이제 document 열 기준으로 **Null값이 있는지** 확인해보겠습니다.
```py
print(train_data.loc[train_data.document.isnull()])
```
```
[output]
            id document  label
25857  2172111      NaN      1
```


<br>


document 열 기준으로 NULL값을 제거하고 개수를 재확인 하겠습니다.
```py
train_data = train_data.dropna(how='any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
```
```
[output]
False
```


<br>

```py
len(train_data)
```
```
[output]
146182
```


<br>


**ratings_train.txt** 데이터의 총 리뷰 개수는 150000개였으며, document 열 기준으로 중복을 제거했을때 146183개이며, Null값을 제거했을때 146182개가 되었습니다.

<br>




# 4. 정규표현식을 이용한 전처리

정규표현식을 사용하여 문자열에서 한글과 공백을 제외한 모든 문자를 제거하겠습니다.
```py
# 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data[:5]
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/0b1e293b-b4be-4427-8e11-187bf6155459">
</p>


<br>

위 코드는 정규표현식을 사용하여 문자열에서 한글과 공백을 제외한 모든 문자를 제거하는 역할을 합니다.

- "^": 이 문자는 대괄호 내에 사용되었을 때 부정(negation)을 의미합니다. 즉, 대괄호 안의 문자들을 제외한 문자를 선택하도록 합니다.
- "ㄱ-ㅎ": 자음을 의미합니다. 한글 자모 중 초성 자음을 나타내는 범위입니다.
- "ㅏ-ㅣ": 모음을 의미합니다. 한글 자모 중 중성 모음을 나타내는 범위입니다.
- "가-힣": 한글의 전체 범위를 나타냅니다.
- " ": 공백 문자입니다.

<br>

따라서, **'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]'** 는 한글 자모, 한글 문자, 그리고 공백을 제외한 모든 문자를 선택합니다. **'replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")'** 는 선택된 문자들을 빈 문자열로 대체하여 제거하는 역할을 합니다.

예를 들어 다음과 같이 입력하면, 원본 문자열에서 영문, 숫자, 특수문자 등은 모두 제거되고, 한글과 공백만 남게 됩니다.
```py
import re

text = "Hello, 안녕하세요! 123ABC"

cleaned_text = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text)

print(cleaned_text)
```
```
[output]
안녕하세요 
```


<br>


또한 정규표현식을 사용하여 문자열에서 시작 부분에 있는 하나 이상의 공백을 제거하겠습니다.
```py
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
```


<br>

위 코드는 정규표현식을 사용하여 문자열에서 시작 부분에 있는 하나 이상의 공백을 제거하는 역할을 합니다.

- "^": 이 문자는 정규표현식에서 문자열의 시작 부분을 나타냅니다.
- "+": 이 문자는 바로 앞의 패턴이 하나 이상의 반복을 의미합니다.
- " ": 공백 문자입니다.

<br>

따라서, **'^ +'** 는 문자열의 시작 부분에 있는 하나 이상의 공백을 선택합니다. **'replace("^ +", "")'** 는 선택된 공백을 빈 문자열로 대체하여 제거하는 역할을 합니다.

예를 들어 다음과 같이 입력하면, 원본 문자열에서 시작 부분에 있는 공백이 제거되어 "Hello, 안녕하세요!"가 남게 됩니다.
```py
import re

text = "    Hello, 안녕하세요!"

cleaned_text = re.sub("^ +", "", text)

print(cleaned_text)
```
```
[output]
Hello, 안녕하세요!
```


<br>


이후 빈 문자열을 numpy 라이브러리의 NaN 값으로 대체하겠습니다.
```py

train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())
```
```
[output]
id            0
document    789
label         0
dtype: int64
```


<br>

NaN값을 한번 확인해보겠습니다.
```py
print(train_data.loc[train_data.document.isnull()][:5])
```
```
[output]
           id document  label
404   4221289      NaN      0
412   9509970      NaN      1
470  10147571      NaN      1
584   7117896      NaN      0
593   6478189      NaN      0
```


<br>

NaN값을 제거한 뒤, 데이터의 개수를 확인하겠습니다.
```py
train_data = train_data.dropna(how='any')
print(len(train_data))
```
```
[output]
145393
```


<br>


최종적으로 **ratings_train.txt** 데이터의 총 리뷰 개수는 150000개였으며, document 열 기준으로 중복을 제거했을때 146183개이며, Null값을 제거했을때 146182개가 되었으며, 정규표현식 전처리를 통해 145393개가 되었습니다. 이제 **ratings_test.txt** 데이터에 그대로 적용하겠습니다.
```py
print('테스트용 샘플의 개수 :', len(test_data)) # 훈련용 리뷰 개수 출력
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :', len(test_data))
```
```
[output]
테스트용 샘플의 개수 : 50000
전처리 후 테스트용 샘플의 개수 : 48852
```


<br>





# 5. 단어 토큰화

전처리된 데이터를 바탕으로 **Ok** 를 이용해서 **단어 토큰화** 를 진행하겠습니다. 앞서 불용어를 정의하고 Okt를 불러오겠습니다.
```py
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
```


<br>

```py
okt = Okt()
okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔')
```
```
[output]
['와', '이런', '것', '도', '영화', '라고', '차라리', '뮤직비디오', '를', '만드는', '게', '나을', '뻔']
```


<br>


지금부터 train 및 test 데이터에 대한 단어 토큰화를 진행하겠습니다.
```py
X_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)
```
```
[output]
100%|██████████| 145393/145393 [09:19<00:00, 259.68it/s]
```


<br>

```py
X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(stopwords_removed_sentence)
```
```
[output]
100%|██████████| 48852/48852 [03:19<00:00, 244.43it/s]
```


<br>


train 및 test 데이터에 대한 단어 토큰화가 완료되었습니다. 총 단어의 개수 및 각 단어의 등장 횟수를 확인해보겠습니다.
```py
word_list = []
for sent in X_train:
    for word in sent:
        word_list.append(word)

word_counts = Counter(word_list)
print('총 단어수 :', len(word_counts))
```
```
[output]
총 단어수 : 100004
```


<br>

```py
print(word_counts)
```
```
[output]
Counter({'영화': 50367, '을': 23208, '너무': 11124, ... (중략) ..., '들어나': 1, '찎었': 1, '디케이드': 1, '수간': 1})
```


<br>

```py
print('훈련 데이터에서의 단어 영화의 등장 횟수 :', word_counts['영화'])
print('훈련 데이터에서의 단어 송강호의 등장 횟수 :', word_counts['송강호'])
print('훈련 데이터에서의 단어 열외의 등장 횟수 :', word_counts['열외'])
```
```
[output]
훈련 데이터에서의 단어 영화의 등장 횟수 : 50367
훈련 데이터에서의 단어 송강호의 등장 횟수 : 74
훈련 데이터에서의 단어 열외의 등장 횟수 : 0
```


<br>



# 6. 토큰화기반 통계적 전처리

단어사전에 대해 등장 빈도수 상위 10개 단어를 추출해보고 **통계적 수치** 를 계산하겠습니다.
```py
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print('등장 빈도수 상위 10개 단어')
print(vocab[:10])
```
```
[output]
등장 빈도수 상위 10개 단어
['영화', '을', '너무', '다', '정말', '적', '만', '진짜', '로', '점']
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

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```
```
[output]
단어 집합(vocabulary)의 크기 : 100004
등장 빈도가 2번 이하인 희귀 단어의 수: 67691
단어 집합에서 희귀 단어의 비율: 67.68829246830127
전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 4.949305773454659
```


<br>

등장 빈도가 threshold 값인 3회 미만. 즉, 2회 이하인 단어들은 단어 집합에서 무려 67% 이상을 차지합니다. 하지만, 실제로 훈련 데이터에서 등장 빈도로 차지하는 비중은 상대적으로 매우 적은 수치인 4.94%밖에 되지 않습니다. 아무래도 등장 빈도가 2회 이하인 단어들은 자연어 처리에서 별로 중요하지 않을 듯 합니다. 그래서 이 단어들은 정수 인코딩 과정에서 배제시키겠습니다.


등장 빈도수가 2이하인 단어들의 수를 제외한 **단어의 개수를 단어 집합의 최대 크기로 제한** 하겠습니다.
```py
# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
vocab_size = total_cnt - rare_cnt
vocab = vocab[:vocab_size]
print('단어 집합의 크기 :', len(vocab))
```
```
[output]
단어 집합의 크기 : 32313
```


<br>


```py
print(vocab)
```
```
[output]
['영화', '을', '너무', '다', ... (중략) ..., '황홀하게', '쥬다이', '라쿠']
```


<br>


단어 집합의 크기는 100004개에서 등장 빈도수가 2이하인 단어들의 수를 제외한 뒤 확인해보니 32313개 였습니다. 정제된 단어 집합에 특수 토큰인 **PAD**, **UNK** 를 삽입하고 최종 단어 집합을 구성하겠습니다.
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
{'<PAD>': 0, '<UNK>': 1, '영화': 2, '을': 3, '너무': 4, '다': 5, ... (중략) ..., '황홀하게': 32312, '쥬다이': 32313, '라쿠': 32314}
```


<br>


```py
vocab_size = len(word_to_index)
print('패딩 토큰과 UNK 토큰을 고려한 단어 집합의 크기 :', vocab_size)
```
```
[output]
패딩 토큰과 UNK 토큰을 고려한 단어 집합의 크기 : 32315
```


<br>



최종적으로 전처리된 단어 집합의 크기는 특수 토큰인 PAD, UNK를 포함한 개수인 32315개 입니다. 단어에 대한 index를 조회해보겠습니다.
```py
word_to_index['영화']
```
```
[output]
2
```


<br>


```py
word_to_index['송강호']
```
```
[output]
2314
```


<br>

```py
word_to_index['열외']
```
```
[output]
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-50-a70a2828dec6> in <module>
----> 1 word_to_index['열외']

KeyError: '열외'
```


<br>

단어 사전에 없는 단어인 '열외'를 조회하면 오류가 납니다.

<br>





# 7. 정수 인코딩

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

print('토큰화 전 원본 문장 :', train_data['document'][0])
print('정수 인코딩 전 토큰화 :', X_train[0])
print('정수 인코딩 결과 :', encoded_X_train[0])
```
```
[output]
토큰화 전 원본 문장 : 아 더빙 진짜 짜증나네요 목소리
정수 인코딩 전 토큰화 : ['아', '더빙', '진짜', '짜증나네요', '목소리']
정수 인코딩 결과 : [41, 418, 9, 6599, 625]
```


<br>

정수 인코딩이 완료되었습니다. 정수 디코딩을 하여 기존 첫번째 샘플을 복원해보겠습니다.
```py
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

decoded_sample = [index_to_word[word] for word in encoded_X_train[0]]
print('기존의 첫번째 샘플 :', X_train[0])
print('복원된 첫번째 샘플 :', decoded_sample)
```
```
[output]
기존의 첫번째 샘플 : ['아', '더빙', '진짜', '짜증나네요', '목소리']
복원된 첫번째 샘플 : ['아', '더빙', '진짜', '짜증나네요', '목소리']
```


<br>





# 8. 패딩

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
리뷰의 최대 길이 : 72
리뷰의 평균 길이 : 11.222988727105156
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/9d34dcd9-a693-4961-a79e-609ffb6c768b">
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

max_len = 70
below_threshold_len(max_len, encoded_X_train)
```
```
[output]
전체 샘플 중 길이가 70 이하인 샘플의 비율: 99.99931220897842
```


<br>



전체 샘플 중 길이가 70 이하인 샘플의 비율은 99.999% 입니다. 패딩을 진행하겠습니다.
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
훈련 데이터의 크기 : (145393, 70)
테스트 데이터의 크기 : (48852, 70)
```


<br>


정상적으로 패딩이 완료되었는지 확인하겠습니다.
```py
padded_X_train[:5]
```
```
[output]
array([[   41,   418,     9,  6599,   625,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0],
       [  906,   420,    32,   568,     2,   183,  1522,    13,   940,
         6037, 25785,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0],
       [  358,  2814,     1,  2647,  7327, 12029,   190,     5,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0],
       [ 8582,    90, 11211,   206,    47,    65,    15,  4338,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0],
       [ 1006,     1,    18,     1,    13,  6410,     2,  2971,    12,
         5281,     1,   441, 21848,     1,  1071,  3610,  4527,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0]])
```


<br>





정상적으로 모든 전처리가 완료되었습니다.





