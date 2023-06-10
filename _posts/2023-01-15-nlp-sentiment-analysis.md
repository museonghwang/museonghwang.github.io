---
layout: post
title: 영화리뷰 텍스트 감성분석(sentiment analysis)하기
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





자연어 처리에 사용되는 기본 아키텍처인 **RNN(Recurrent Neural Network)** 과 컴퓨터 비전에서 주로 사용하는 **CNN(Convolutional Neural Network)** 구조를 학습하고 이를 활용하여 IMDB 영화 리뷰 평점 데이터를 토대로 영화리뷰에 대한 **<span style="color:red">감성분석(sentiment analysis)</span>** 를 진행해 보도록 하겠습니다.

<br>





# 1. 텍스트 데이터의 특징

인공지능 모델을 입력과 출력이 정해진 함수라고 생각해 봅시다. 예를 들어 MNIST 숫자 분류기 모델이라면 이미지 파일을 읽어 들인 매트릭스가 입력이 되고, 이미지 파일에 쓰여 있는 실제 숫자 값이 출력이 되는 함수가 될 것입니다.



이제 텍스트 문장을 입력으로 받아서 그 의미가 긍정이면 1, 부정이면 0을 출력하는 인공지능 모델을 만든다고 생각해 봅시다. 이 모델을 만들기 위해서는 숫자 분류기를 만들 때는 생각할 필요가 없었던 2가지 문제가 생깁니다.

- **텍스트를 어떻게 숫자 행렬로 표현할 수 있을까?**
- **텍스트에는 순서가 중요한데, 입력 데이터의 순서를 인공지능 모델에 어떻게 반영해야 하는가?**

<br>

인공지능 모델의 입력이 될 수 있는 것은 0과 1의 비트로 표현 가능한 숫자만으로 이루어진 매트릭스일 뿐입니다. 아주 단순히, A=0, B=1, ..., Z=25 라고 숫자를 임의로 부여한다고 해봅시다. 그러면 의미적으로 A와 B는 1만큼 멀고, A와 Z는 25만큼 멀까요? 그렇지 않습니다. **<u>텍스트의 중요한 특징은 그 자체로는 기호일 뿐이며, 텍스트가 내포하는 의미를 기호가 직접 내포하지 않는다는 점입니다.</u>**

<br>





# 2. 텍스트 데이터의 특징 (1) 텍스트를 숫자로 표현하는 방법


우선 단어 사전을 만들어 볼 수는 있습니다. 우리가 사용하는 국어, 영어 사전에는 단어와 그 의미 설명이 짝지어져 있습니다. 우리가 하려는 것은 **<span style="color:red">단어와 그 단어의 의미를 나타내는 벡터 를 짝지어 보려고 하는 것</span>** 입니다. 그런데 그 벡터는 어디서 가져올까요? 그렇습니다. 우리는 **딥러닝을 통해 그 벡터를 만들어 낼 수 있습니다.**

아래와 같이 단 3개의 짧은 문장으로 이루어진 텍스트 데이터를 처리하는 간단한 예제를 생각해 보겠습니다.
```
i feel hungry
i eat lunch
now i feel happy
```

<br>

```py
# 처리해야 할 문장을 파이썬 리스트에 옮겨 담았습니다.
sentences = ['i feel hungry', 'i eat lunch', 'now i feel happy']

# 파이썬 split() 메소드를 이용해 단어 단위로 문장을 쪼개 봅니다.
word_list = 'i feel hungry'.split()
print(word_list)
```
```
[output]
['i', 'feel', 'hungry']
```


<br>



**<u>텍스트 데이터로부터 사전을 만들기 위해</u>** 모든 문장을 단어 단위로 쪼갠 후에 파이썬 **딕셔너리(dict) 자료구조** 로 표현해 보겠습니다.
```py
index_to_word = {}  # 빈 딕셔너리를 만들어서

# 단어들을 하나씩 채워 봅니다. 채우는 순서는 일단 임의로 하였습니다. 그러나 사실 순서는 중요하지 않습니다. 
# <BOS>, <PAD>, <UNK>는 관례적으로 딕셔너리 맨 앞에 넣어줍니다. 
index_to_word[0] = '<PAD>'  # 패딩용 단어
index_to_word[1] = '<BOS>'  # 문장의 시작지점
index_to_word[2] = '<UNK>'  # 사전에 없는(Unknown) 단어
index_to_word[3] = 'i'
index_to_word[4] = 'feel'
index_to_word[5] = 'hungry'
index_to_word[6] = 'eat'
index_to_word[7] = 'lunch'
index_to_word[8] = 'now'
index_to_word[9] = 'happy'

print(index_to_word)
```
```
[output]
{0: '<PAD>', 1: '<BOS>', 2: '<UNK>', 3: 'i', 4: 'feel', 5: 'hungry', 6: 'eat', 7: 'lunch', 8: 'now', 9: 'happy'}
```


<br>


단어 10개짜리 작은 딕셔너리가 만들어졌습니다. 하지만 우리가 가진 **<u>텍스트 데이터를 숫자로 바꿔</u>** 보려고 하는데, 텍스트를 숫자로 바꾸려면 위의 딕셔너리가 **{텍스트:인덱스}** 구조여야 합니다.
```py
word_to_index = {word:index for index, word in index_to_word.items()}
print(word_to_index)
```
```
[output]
{'<PAD>': 0, '<BOS>': 1, '<UNK>': 2, 'i': 3, 'feel': 4, 'hungry': 5, 'eat': 6, 'lunch': 7, 'now': 8, 'happy': 9}
```


<br>


이 딕셔너리는 단어를 주면 그 단어의 인덱스를 반환하는 방식으로 사용할 수 있습니다.
```py
print(word_to_index['feel'])  # 단어 'feel'은 숫자 인덱스 4로 바뀝니다.
```
```
[output]
4
```


<br>


이제 우리가 가진 텍스트 데이터를 숫자로 바꿔 표현, 즉 **<span style="color:red">encode</span>** 해 봅시다.
```py
# 문장 1개를 활용할 딕셔너리와 함께 주면, 단어 인덱스 리스트로 변환해 주는 함수를 만들어 봅시다.
# 단, 모든 문장은 <BOS>로 시작하는 것으로 합니다. 
def get_encoded_sentence(sentence, word_to_index):
    return [word_to_index['<BOS>']] + [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]

print(get_encoded_sentence('i eat lunch', word_to_index))
```
```
[output]
[1, 3, 6, 7]
```


<br>

**get_encoded_sentence** 함수를 통해 아래와 같이 매핑된 것이 확인할 수 있습니다.

- **<BOS>** -> 1
- **i** -> 3
- **eat** -> 6
- **lunch** -> 7


<br>


```py
# 여러 개의 문장 리스트를 한꺼번에 숫자 텐서로 encode해 주는 함수입니다. 
def get_encoded_sentences(sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

# sentences=['i feel hungry', 'i eat lunch', 'now i feel happy'] 가 아래와 같이 변환됩니다. 
encoded_sentences = get_encoded_sentences(sentences, word_to_index)
print(encoded_sentences)
```
```
[output]
[[1, 3, 4, 5], [1, 3, 6, 7], [1, 8, 3, 4, 9]]
```


<br>



반대로, **encode** 된 벡터를 **<span style="color:red">decode</span>** 하여 다시 원래 텍스트 데이터로 복구할 수도 있습니다.
```py
# 숫자 벡터로 encode된 문장을 원래대로 decode하는 함수입니다. 
def get_decoded_sentence(encoded_sentence, index_to_word):
    # [1:]를 통해 <BOS>를 제외
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])

print(get_decoded_sentence([1, 3, 4, 5], index_to_word))
```
```
[output]
i feel hungry
```


<br>

```py
# 여러 개의 숫자 벡터로 encode된 문장을 한꺼번에 원래대로 decode하는 함수입니다. 
def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]

# encoded_sentences=[[1, 3, 4, 5], [1, 3, 6, 7], [1, 8, 3, 4, 9]] 가 아래와 같이 변환됩니다.
print(get_decoded_sentences(encoded_sentences, index_to_word))
```
```
[output]
['i feel hungry', 'i eat lunch', 'now i feel happy']
```


<br>


여기서 정의된 함수들은 이후 스텝들에서 반복해서 활용됩니다.


<br>





# 3. 텍스트 데이터의 특징 (2) Embedding 레이어의 등장

텍스트가 숫자로 변환되어 인공지능 모델의 입력으로 사용될 수 있게 되었지만, 이것으로 충분하지는 않습니다. **'i feel hungry'** 가 **1, 3, 4, 5** 로 변환되었지만 **<span style="color:red">이 벡터는 텍스트에 담긴 언어의 의미와 대응되는 벡터가 아니라 임의로 부여된 단어의 순서에 불과</span>** 합니다. **<span style="color:red">우리가 하려는 것은 단어와 그 단어의 의미를 나타내는 벡터를 짝짓는 것</span>** 이었습니다. 그래서 단어의 의미를 나타내는 벡터를 훈련 가능한 파라미터로 놓고 이를 딥러닝을 통해 학습해서 최적화하게 됩니다. Tensorflow, Pytorch 등의 딥러닝 프레임워크들은 이러한 의미 벡터 파라미터를 구현한 **<span style="color:red">Embedding 레이어</span>** 를 제공합니다.


<br>


자연어 처리(Natural Language Processing)분야에서 **<span style="color:red">임베딩(Embedding)</span>** 은 사람이 쓰는 자연어를 기계가 이해할 수 있는 숫자형태인 vector로 바꾼 결과 혹은 그 일련의 과정 전체를 의미하며, 가장 간단한 형태의 임베딩은 단어의 빈도를 그대로 벡터로 사용하는 것입니다. 임베딩은 다른 딥러닝 모델의 입력값으로 자주 쓰이고, 품질 좋은 임베딩을 쓸수록 모델의 성능이 좋아집니다.


<br>




<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/ab094e70-e0d4-4bf4-bac6-149a6a88731e">
</p>

<br>


위 그림에서 **word_to_index('great')** 는 **1918** 입니다. 그러면 **'great'** 라는 단어의 의미 공간상의 워드 벡터(word vector)는 Lookup Table 형태로 구성된 Embedding 레이어의 1919번째 벡터가 됩니다. 위 그림에서는 **1.2, 0.7, 1.9, 1.5** 가 됩니다.

**Embedding** 레이어를 활용하여 이전 스텝의 텍스트 데이터를 **<span style="color:red">워드 벡터 텐서 형태로 다시 표현</span>** 해 보겠습니다.
```py
# 아래 코드는 그대로 실행하시면 에러가 발생할 것입니다. 

import numpy as np
import tensorflow as tf
import os

vocab_size = len(word_to_index)  # 위 예시에서 딕셔너리에 포함된 단어 개수는 10
word_vector_dim = 4    # 위 그림과 같이 4차원의 워드 벡터를 가정합니다. 

embedding = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=word_vector_dim,
    mask_zero=True
)

# 숫자로 변환된 텍스트 데이터 [[1, 3, 4, 5], [1, 3, 6, 7], [1, 8, 3, 4, 9]] 에 Embedding 레이어를 적용합니다. 
raw_inputs = np.array(get_encoded_sentences(sentences, word_to_index), dtype='object')
output = embedding(raw_inputs)
print(output)
```
```
[output]
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/root/DeepLearningProject/Exploration-01/1. 영화리뷰 텍스트 감성분석하기.ipynb 셀 34 in 1
     12 # 숫자로 변환된 텍스트 데이터 [[1, 3, 4, 5], [1, 3, 6, 7], [1, 8, 3, 4, 9]] 에 Embedding 레이어를 적용합니다. 
     13 raw_inputs = np.array(get_encoded_sentences(sentences, word_to_index), dtype='object')
---> 14 output = embedding(raw_inputs)
     15 print(output)

.
.
.

ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type list).
```


<br>


실행해 보니 에러가 발생합니다. 왜 그럴까요? 주의해야 할 점이 있는데, **<span style="color:red">Embedding 레이어의 input이 되는 문장 벡터는 그 길이가 일정 해야 합니다.</span>** **raw_inputs** 의 **3개 벡터의 길이** 는 각각 **4, 4, 5** 입니다. Tensorflow에서는 **tf.keras.preprocessing.sequence.pad_sequences** 라는 편리한 함수를 통해 문장 벡터 뒤에 **<span style="color:red">패딩(<PAD>)</span>** 을 추가하여 **<u>길이를 일정하게 맞춰주는 기능을 제공</u>** 합니다.
```py
raw_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs,
    value=word_to_index['<PAD>'],
    padding='post',
    maxlen=5
)
print(raw_inputs)
```
```
[output]
[[1 3 4 5 0]
 [1 3 6 7 0]
 [1 8 3 4 9]]
```


<br>



짧은 문장 뒤쪽이 0으로 채워지는 것을 확인할 수 있습니다. **<PAD>** 가 0에 매핑되어 있다는 걸 기억하세요. 그러면 위에 시도했던 **output = embedding(raw_inputs)** 을 다시 시도해보겠습니다.
```py
import numpy as np
import tensorflow as tf
import os

vocab_size = len(word_to_index)  # 위 예시에서 딕셔너리에 포함된 단어 개수는 10
word_vector_dim = 4    # 그림과 같이 4차원의 워드 벡터를 가정합니다.

# tf.keras.preprocessing.sequence.pad_sequences를 통해 word vector를 모두 일정 길이로 맞춰주어야 
# embedding 레이어의 input이 될 수 있음에 주의해 주세요. 
raw_inputs = np.array(get_encoded_sentences(sentences, word_to_index), dtype=object)
raw_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs,
    value=word_to_index['<PAD>'],
    padding='post',
    maxlen=5
)

embedding = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=word_vector_dim,
    mask_zero=True
)

output = embedding(raw_inputs)
print(output)
```
```
[output]
tf.Tensor(
[[[ 0.01395203  0.01275567 -0.02401054 -0.02090124]
  [-0.04068121 -0.01742718 -0.00815885 -0.02921033]
  [ 0.01333598 -0.02777994 -0.03375378 -0.03576127]
  [ 0.03633847  0.01739961 -0.00358608 -0.04881281]
  [ 0.01803286 -0.0018087  -0.00501338 -0.00896306]]

 [[ 0.01395203  0.01275567 -0.02401054 -0.02090124]
  [-0.04068121 -0.01742718 -0.00815885 -0.02921033]
  [-0.03603647  0.04998456 -0.00863974  0.02029398]
  [-0.02421701 -0.00824345 -0.0313128  -0.03255729]
  [ 0.01803286 -0.0018087  -0.00501338 -0.00896306]]

 [[ 0.01395203  0.01275567 -0.02401054 -0.02090124]
  [-0.01857213  0.01194413  0.02710113  0.04482098]
  [-0.04068121 -0.01742718 -0.00815885 -0.02921033]
  [ 0.01333598 -0.02777994 -0.03375378 -0.03576127]
  [-0.048308    0.00933223  0.04508836 -0.04411023]]], shape=(3, 5, 4), dtype=float32)
```


<br>

여기서 **output** 의 **shape=(3, 5, 4)** 에서 3, 5, 4의 의미는 다음과 같습니다.

- **3** : 입력문장 개수
- **5** : 입력문장의 최대 길이
- **4** : 워드 벡터의 차원 수



<br>





# 4. 시퀀스 데이터를 다루는 RNN

텍스트 데이터를 다루는 데 주로 사용되는 딥러닝 모델은 바로 **<span style="color:red">Recurrent Neural Network(RNN)</span>** 입니다. **<u>RNN은 시퀀스(Sequence) 형태의 데이터를 처리하기에 최적인 모델</u>** 로 알려져 있습니다.

텍스트 데이터도 시퀀스 데이터라는 관점으로 해석할 수 있습니다만, 시퀀스 데이터의 정의에 가장 잘 어울리는 것은 음성 데이터가 아닐까 합니다. **시퀀스 데이터란 바로 입력이 시간 축을 따라 발생하는 데이터** 입니다. 예를 들어 이전 스텝의 'i feel hungry'라는 문장을 누군가가 초당 한 단어씩, 3초에 걸쳐 이 문장을 발음했다고 합시다.
```
at time=0s : 듣는이의 귀에 들어온 input='i'
at time=1s : 듣는이의 귀에 들어온 input='feel'
at time=2s : 듣는이의 귀에 들어온 input='hungry'
```

<br>


**time=1s** 인 시점에서 입력으로 받은 문장은 **'i feel'** 까지입니다. 그다음에 **'hungry'** 가 올지, **'happy'** 가 올지 알 수 없는 상황입니다. **RNN**은 그런 상황을 묘사하기에 가장 적당한 모델 구조를 가지고 있습니다. 왜냐하면 **RNN** 은 시간의 흐름에 따라 새롭게 들어오는 입력에 따라 변하는 현재 상태를 묘사하는 **state machine** 으로 설계되었기 때문입니다.
```py
vocab_size = 10  # 어휘 사전의 크기입니다(10개의 단어)
word_vector_dim = 4  # 단어 하나를 표현하는 임베딩 벡터의 차원수입니다. 

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(tf.keras.layers.LSTM(8))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경 가능)
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
```
```
[output]
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_5 (Embedding)     (None, None, 4)           40        
                                                                 
 lstm (LSTM)                 (None, 8)                 416       
                                                                 
 dense (Dense)               (None, 8)                 72        
                                                                 
 dense_1 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 537
Trainable params: 537
Non-trainable params: 0
_________________________________________________________________
```


<br>





# 5. 1-D Convolution Neural Network


텍스트를 처리하기 위해 **RNN** 이 아니라 **<span style="color:red">1-D Convolution Neural Network(1-D CNN)</span>** 를 사용할 수도 있습니다. **1-D CNN** 은 **<u>문장 전체를 한꺼번에 한 방향으로 길이 n짜리 필터로 스캐닝 하면서 n단어 이내에서 발견되는 특징을 추출하여 그것으로 문장을 분류하는 방식으로 사용</u>** 됩니다. 이 방식도 텍스트를 처리하는 데 RNN 못지않은 효율을 보여줍니다. 그리고 CNN 계열은 RNN 계열보다 병렬처리가 효율적이기 때문에 학습 속도도 훨씬 빠르게 진행된다는 장점이 있습니다.
```py
vocab_size = 10  # 어휘 사전의 크기입니다(10개의 단어)
word_vector_dim = 4   # 단어 하나를 표현하는 임베딩 벡터의 차원 수입니다. 

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(tf.keras.layers.Conv1D(16, 7, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(5))
model.add(tf.keras.layers.Conv1D(16, 7, activation='relu'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
```
```
[output]
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_6 (Embedding)     (None, None, 4)           40        
                                                                 
 conv1d (Conv1D)             (None, None, 16)          464       
                                                                 
 max_pooling1d (MaxPooling1D  (None, None, 16)         0         
 )                                                               
                                                                 
 conv1d_1 (Conv1D)           (None, None, 16)          1808      
                                                                 
 global_max_pooling1d (Globa  (None, 16)               0         
 lMaxPooling1D)                                                  
                                                                 
 dense_2 (Dense)             (None, 8)                 136       
                                                                 
 dense_3 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 2,457
Trainable params: 2,457
Non-trainable params: 0
_________________________________________________________________
```


<br>


아주 간단히는 **GlobalMaxPooling1D()** 레이어 하나만 사용하는 방법도 생각해 볼 수 있습니다. 이 방식은 전체 문장 중에서 단 하나의 가장 중요한 단어만 피처로 추출하여 그것으로 문장의 긍정/부정을 평가하는 방식이라고 생각할 수 있는데, 의외로 성능이 잘 나올 수도 있습니다.
```py
vocab_size = 10  # 어휘 사전의 크기입니다(10개의 단어)
word_vector_dim = 4   # 단어 하나를 표현하는 임베딩 벡터의 차원 수입니다. 

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
```
```
[output]
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_7 (Embedding)     (None, None, 4)           40        
                                                                 
 global_max_pooling1d_1 (Glo  (None, 4)                0         
 balMaxPooling1D)                                                
                                                                 
 dense_4 (Dense)             (None, 8)                 40        
                                                                 
 dense_5 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 89
Trainable params: 89
Non-trainable params: 0
_________________________________________________________________
```


<br>



# 6. IMDB 영화리뷰 감성분석 (1) IMDB 데이터셋 분석

이제 본격적으로 IMDB 영화리뷰 감성분석 태스크에 도전해 보겠습니다. IMDb Large Movie Dataset은 50000개의 영어로 작성된 영화 리뷰 텍스트로 구성되어 있으며, 긍정은 1, 부정은 0의 라벨이 달려 있습니다. 2011년 [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015.pdf) 논문에서 이 데이터셋을 소개하였습니다.

50000개의 리뷰 중 절반인 25000개가 훈련용 데이터, 나머지 25000개를 테스트용 데이터로 사용하도록 지정되어 있습니다. 이 데이터셋은 tensorflow Keras 데이터셋 안에 포함되어 있어서 손쉽게 다운로드하여 사용할 수 있습니다.
```py
imdb = tf.keras.datasets.imdb

# IMDb 데이터셋 다운로드 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(f"훈련 샘플 개수: {len(x_train)}, 테스트 개수: {len(x_test)}")
```
```
[output]
훈련 샘플 개수: 25000, 테스트 개수: 25000
```


<br>


**imdb.load_data()** 호출 시 단어사전에 등재할 단어의 개수(num_words)를 10000으로 지정하면, 그 개수만큼의 **word_to_index** 딕셔너리까지 생성된 형태로 데이터셋이 생성됩니다. 다운로드한 데이터 실제 예시를 확인해 보겠습니다.
```py
print(x_train[0])  # 1번째 리뷰데이터
print('라벨: ', y_train[0])  # 1번째 리뷰데이터의 라벨
print('1번째 리뷰 문장 길이: ', len(x_train[0]))
print('2번째 리뷰 문장 길이: ', len(x_train[1]))
```
```
[output]
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
라벨:  1
1번째 리뷰 문장 길이:  218
2번째 리뷰 문장 길이:  189
```


<br>


텍스트 데이터가 아니라 이미 숫자로 **encode** 된 텍스트 데이터를 다운로드했음을 확인할 수 있습니다. 이미 텍스트가 **encode** 되었으므로 IMDb 데이터셋에는 **encode** 에 사용한 딕셔너리까지 함께 제공합니다.
```py
word_to_index = imdb.get_word_index()
index_to_word = {index:word for word, index in word_to_index.items()}
print(index_to_word[1])     # 'the' 가 출력됩니다. 
print(word_to_index['the']) # 1 이 출력됩니다.
```
```
[output]
the
1
```


<br>


여기서 주의할 점이 있습니다. IMDb 데이터셋의 텍스트 인코딩을 위한 **word_to_index**, **index_to_word** 는 보정이 필요합니다. 예를 들어 다음 코드를 실행시켜보면 보정이 되지 않은 상태라 문장이 이상함을 확인하실 겁니다.
```py
# 보정 전 x_train[0] 데이터
print(get_decoded_sentence(x_train[0], index_to_word))
```
```
[output]
as you with out themselves powerful lets loves their becomes reaching had journalist of lot from anyone to have after out atmosphere never more room and it so heart shows to years of every never going and help moments or of every chest visual movie except her was several of enough more with is now current film as you of mine potentially unfortunately of you than him that with out themselves her get for was camp of you movie sometimes movie that with scary but and to story wonderful that in seeing in character to of 70s musicians with heart had shadows they of here that with her serious to have does when from why what have critics they is you that isn't one will very to as itself with other and in of seen over landed for anyone of and br show's to whether from than out themselves history he name half some br of and odd was two most of mean for 1 any an boat she he should is thought frog but of script you not while history he heart to real at barrel but when from one bit then have two of script their with her nobody most that with wasn't to with armed acting watch an for with heartfelt film want an
```


<br>


그럼 매핑 보정 작업을 해보겠습니다. **word_to_index** 는 IMDb 텍스트 데이터셋의 단어 출현 빈도 기준으로 내림차수 정렬되어 있습니다.
```py
#실제 인코딩 인덱스는 제공된 word_to_index에서 index 기준으로 3씩 뒤로 밀려 있습니다.  
word_to_index = {k:(v+3) for k,v in word_to_index.items()}

# 처음 몇 개 인덱스는 사전에 정의되어 있습니다.
word_to_index["<PAD>"] = 0
word_to_index["<BOS>"] = 1
word_to_index["<UNK>"] = 2  # unknown
word_to_index["<UNUSED>"] = 3

index_to_word = {index:word for word, index in word_to_index.items()}

print(index_to_word[1])     # '<BOS>' 가 출력됩니다. 
print(word_to_index['the'])  # 4 이 출력됩니다. 
print(index_to_word[4])     # 'the' 가 출력됩니다.

# 보정 후 x_train[0] 데이터
print(get_decoded_sentence(x_train[0], index_to_word))
```
```
[output]
<BOS>
4
the
this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert <UNK> is an amazing actor and now the same being director <UNK> father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for <UNK> and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also <UNK> to the two little boy's that played the <UNK> of norman and paul they were just brilliant children are often left out of the <UNK> list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all
```


<br>


다운로드한 데이터셋이 확인되었습니다. 보정 후 **x_train[0]** 데이터도 자연스러운 문장으로 바뀌었습니다. 마지막으로, **encode** 된 텍스트가 정상적으로 **decode** 되는지 확인해 보겠습니다.
```py
print(get_decoded_sentence(x_train[0], index_to_word))
print('라벨: ', y_train[0])  # 1번째 리뷰데이터의 라벨
```
```
[output]
this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert <UNK> is an amazing actor and now the same being director <UNK> father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for <UNK> and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also <UNK> to the two little boy's that played the <UNK> of norman and paul they were just brilliant children are often left out of the <UNK> list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all
라벨:  1
```


<br>


**pad_sequences** 를 통해 데이터셋 상의 문장의 길이를 통일하는 것을 잊어서는 안됩니다. 문장 최대 길이 **maxlen** 의 값 설정도 전체 모델 성능에 영향을 미치게 됩니다. 이 길이도 적절한 값을 찾기 위해서는 전체 데이터셋의 분포를 확인해 보는 것이 좋습니다.
```py
total_data_text = list(x_train) + list(x_test)

# 텍스트데이터 문장길이의 리스트를 생성한 후
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)

# 문장길이의 평균값, 최대값, 표준편차를 계산해 본다. 
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))

# 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,  
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print('pad_sequences maxlen : ', maxlen)
print(f'전체 문장의 {np.sum(num_tokens < max_tokens) / len(num_tokens)}%가 maxlen 설정값 이내에 포함됩니다. ')
```
```
[output]
문장길이 평균 :  234.75892
문장길이 최대 :  2494
문장길이 표준편차 :  172.91149458735703
pad_sequences maxlen :  580
전체 문장의 0.94536%가 maxlen 설정값 이내에 포함됩니다. 
```


<br>


위의 경우에는 **maxlen=580** 이 됩니다. 또 한 가지 유의해야 하는 것은 **<u>padding 방식을 문장 뒤쪽('post')과 앞쪽('pre') 중 어느 쪽으로 하느냐에 따라 RNN을 이용한 딥러닝 적용 시 성능 차이가 발생한다는 점</u>** 입니다.

**RNN** 활용 시 **pad_sequences** 의 **padding** 방식은 **'post'** 와 **'pre'** 중 어느 것이 유리할까요? **RNN** 은 입력데이터가 순차적으로 처리되어, 가장 마지막 입력이 최종 **state** 값에 가장 영향을 많이 미치게 됩니다. **<u>그러므로 마지막 입력이 무의미한 padding으로 채워지는 것은 비효율적</u>** 입니다. **<span style="color:red">따라서 'pre'가 훨씬 유리</span>** 하며, 10% 이상의 테스트 성능 차이를 보이게 됩니다.
```py
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train,
    value=word_to_index["<PAD>"],
    padding='post', # 혹은 'pre'
    maxlen=maxlen
)

x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test,
    value=word_to_index["<PAD>"],
    padding='post', # 혹은 'pre'
    maxlen=maxlen
)

print(x_train.shape)
```
```
[output]
(25000, 580)
```


<br>





# 7. IMDB 영화리뷰 감성분석 (2) 딥러닝 모델 설계와 훈련

**model** 훈련 전에, 훈련용 데이터셋 25000건 중 10000건을 분리하여 ***검증셋(validation set)** 으로 사용하도록 합니다. 적절한 validation 데이터는 몇 개가 좋을지 고민해 봅시다.
```py
# validation set 10000건 분리
x_val = x_train[:10000]   
y_val = y_train[:10000]

# validation set을 제외한 나머지 15000건
partial_x_train = x_train[10000:]  
partial_y_train = y_train[10000:]

print(x_val.shape)
print(y_val.shape)
print(partial_x_train.shape)
print(partial_y_train.shape)
```
```
[output]
(10000, 580)
(10000,)
(15000, 580)
(15000,)
```


<br>



RNN 모델을 직접 설계해 보겠습니다. 참고로 여러가지 모델을 사용할 수 있습니다.
```py
vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 16  # 워드 벡터의 차원 수 (변경 가능한 하이퍼파라미터)

# model 설계 - 딥러닝 모델 코드를 직접 작성해 주세요.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(tf.keras.layers.LSTM(8))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경 가능)
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
```
```
[output]
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_4 (Embedding)     (None, None, 16)          160000    
                                                                 
 lstm_1 (LSTM)               (None, 8)                 800       
                                                                 
 dense_6 (Dense)             (None, 8)                 72        
                                                                 
 dense_7 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 160,881
Trainable params: 160,881
Non-trainable params: 0
_________________________________________________________________
```


<br>


model 학습을 시작해 봅시다.
```py
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
              
epochs=20  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=epochs,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)
```
```
[output]
Epoch 1/20
30/30 [==============================] - 3s 34ms/step - loss: 0.6931 - accuracy: 0.5082 - val_loss: 0.6932 - val_accuracy: 0.5019
Epoch 2/20
30/30 [==============================] - 1s 28ms/step - loss: 0.6926 - accuracy: 0.5107 - val_loss: 0.6928 - val_accuracy: 0.5010
Epoch 3/20
30/30 [==============================] - 1s 30ms/step - loss: 0.6918 - accuracy: 0.5015 - val_loss: 0.6923 - val_accuracy: 0.5026
Epoch 4/20
30/30 [==============================] - 1s 29ms/step - loss: 0.6875 - accuracy: 0.5169 - val_loss: 0.6954 - val_accuracy: 0.5043
Epoch 5/20
30/30 [==============================] - 1s 28ms/step - loss: 0.6793 - accuracy: 0.5285 - val_loss: 0.6885 - val_accuracy: 0.5084
Epoch 6/20
30/30 [==============================] - 1s 26ms/step - loss: 0.6741 - accuracy: 0.5335 - val_loss: 0.6910 - val_accuracy: 0.5074
Epoch 7/20
30/30 [==============================] - 1s 32ms/step - loss: 0.6734 - accuracy: 0.5317 - val_loss: 0.6915 - val_accuracy: 0.5067
Epoch 8/20
30/30 [==============================] - 1s 30ms/step - loss: 0.6707 - accuracy: 0.5365 - val_loss: 0.6957 - val_accuracy: 0.5068
Epoch 9/20
30/30 [==============================] - 1s 29ms/step - loss: 0.6687 - accuracy: 0.5377 - val_loss: 0.6981 - val_accuracy: 0.5070
Epoch 10/20
30/30 [==============================] - 1s 28ms/step - loss: 0.6669 - accuracy: 0.5381 - val_loss: 0.6977 - val_accuracy: 0.5084
Epoch 11/20
30/30 [==============================] - 1s 26ms/step - loss: 0.6649 - accuracy: 0.5388 - val_loss: 0.7020 - val_accuracy: 0.5076
Epoch 12/20
30/30 [==============================] - 1s 28ms/step - loss: 0.6624 - accuracy: 0.5389 - val_loss: 0.7012 - val_accuracy: 0.5093
Epoch 13/20
30/30 [==============================] - 1s 31ms/step - loss: 0.6597 - accuracy: 0.5389 - val_loss: 0.6996 - val_accuracy: 0.5106
Epoch 14/20
30/30 [==============================] - 1s 31ms/step - loss: 0.6573 - accuracy: 0.5391 - val_loss: 0.7016 - val_accuracy: 0.5117
Epoch 15/20
30/30 [==============================] - 1s 33ms/step - loss: 0.6558 - accuracy: 0.5388 - val_loss: 0.7139 - val_accuracy: 0.5102
Epoch 16/20
30/30 [==============================] - 1s 27ms/step - loss: 0.6540 - accuracy: 0.5390 - val_loss: 0.7021 - val_accuracy: 0.5127
Epoch 17/20
30/30 [==============================] - 1s 26ms/step - loss: 0.6527 - accuracy: 0.5395 - val_loss: 0.6975 - val_accuracy: 0.5131
Epoch 18/20
30/30 [==============================] - 1s 33ms/step - loss: 0.6519 - accuracy: 0.5392 - val_loss: 0.7030 - val_accuracy: 0.5134
Epoch 19/20
30/30 [==============================] - 1s 27ms/step - loss: 0.6516 - accuracy: 0.5317 - val_loss: 0.7096 - val_accuracy: 0.5130
Epoch 20/20
30/30 [==============================] - 1s 30ms/step - loss: 0.6507 - accuracy: 0.5402 - val_loss: 0.7019 - val_accuracy: 0.5136
```


<br>


학습이 끝난 모델을 테스트셋으로 평가해 봅니다.
```py
results = model.evaluate(x_test, y_test, verbose=2)

print(results)
```
```
[output]
782/782 - 8s - loss: 0.6969 - accuracy: 0.5182 - 8s/epoch - 10ms/step
[0.6969093084335327, 0.5182399749755859]
```


<br>


**model.fit()** 과정 중의 **train/validation loss**, **accuracy** 등이 매 **epoch** 마다 **history** 변수에 저장되어 있습니다.

이 데이터를 그래프로 그려 보면, 수행했던 딥러닝 학습이 잘 진행되었는지, 오버피팅 혹은 언더피팅하지 않았는지, 성능을 개선할 수 있는 다양한 아이디어를 얻을 수 있는 좋은 자료가 됩니다.
```py
history_dict = history.history
print(history_dict.keys()) # epoch에 따른 그래프를 그려볼 수 있는 항목들
```
```
[output]
dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
```


<br>



```py
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/a8236430-17d8-47d2-95bb-68d01c6f2669">
</p>

<br>


**Training and validation loss** 를 그려 보면, 몇 **epoch** 까지의 트레이닝이 적절한지 최적점을 추정해 볼 수 있습니다. **validation loss** 의 그래프가 **train loss** 와의 이격이 발생하게 되면 더 이상의 트레이닝은 무의미해지게 마련입니다.
```py
plt.clf()   # 그림을 초기화합니다

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/d91b371d-fd66-47f4-b009-048033058d7f">
</p>

<br>



마찬가지로 **Training and validation accuracy** 를 그려 보아도 유사한 인사이트를 얻을 수 있습니다.

<br>





# 8. IMDB 영화리뷰 감성분석 (3) Word2Vec의 적용


이전 스텝에서 라벨링 비용이 많이 드는 머신러닝 기반 감성분석의 비용을 절감하면서 정확도를 크게 향상시킬 수 있는 자연어처리 기법으로 단어의 특성을 저차원 벡터값으로 표현할 수 있는 **<span style="color:red">워드 임베딩(word embedding)</span>** 기법이 있다는 언급을 한 바 있습니다.


우리는 이미 이전 스텝에서 워드 임베딩을 사용했습니다. 사용했던 **model** 의 첫 번째 레이어는 바로 **Embedding** 레이어였습니다. 이 레이어는 우리가 가진 **사전의 단어 개수 × 워드 벡터 사이즈** 만큼의 크기를 가진 학습 파라미터였습니다. 만약 우리의 감성 분류 모델이 학습이 잘 되었다면, **Embedding** 레이어에 학습된 우리의 워드 벡터들도 의미 공간상에 유의미한 형태로 학습되었을 것입니다. 한번 확인해 봅시다.
```py
embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]
print(weights.shape)    # shape: (vocab_size, embedding_dim)
```
```
[output]
(10000, 16)
```


<br>

```py
# 학습한 Embedding 파라미터를 파일에 써서 저장합니다. 
word2vec_file_path = '/data/word2vec.txt'
f = open(word2vec_file_path, 'w')
f.write('{} {}\n'.format(vocab_size-4, word_vector_dim))  # 몇개의 벡터를 얼마 사이즈로 기재할지 타이틀을 씁니다.

# 단어 개수(에서 특수문자 4개는 제외하고)만큼의 워드 벡터를 파일에 기록합니다. 
vectors = model.get_weights()[0]
for i in range(4,vocab_size):
    f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
f.close()
```
```
[output]

```


<br>



워드 벡터를 다루는데 유용한 **<span style="color:red">gensim</span>** 에서 제공하는 패키지를 이용해, 위에 남긴 임베딩 파라미터를 읽어서 **word vector** 로 활용할 수 있습니다.
```py
from gensim.models.keyedvectors import Word2VecKeyedVectors

word_vectors = Word2VecKeyedVectors.load_word2vec_format(word2vec_file_path, binary=False)
vector = word_vectors['computer']
vector
```
```
[output]
array([ 0.03693542,  0.09759595,  0.02630373,  0.00367015, -0.03850541,
        0.03551435,  0.00019023, -0.00949474, -0.07800166, -0.06625549,
        0.01207815, -0.0637126 ,  0.06162278, -0.05298196, -0.01577774,
       -0.04987772], dtype=float32)
```


<br>

위와 같이 얻은 워드 벡터를 가지고 재미있는 실험을 해볼 수 있습니다. **<span style="color:red">워드 벡터가 의미 벡터 공간상에 유의미하게 학습되었는지 확인하는 방법 중에, 단어를 하나 주고 그와 가장 유사한 단어와 그 유사도를 확인하는 방법</span>** 이 있습니다. **gensim** 을 사용하면 아래와 같이 해볼 수 있습니다.
```py
word_vectors.similar_by_word("love")
```
```
[output]
[('idiocy', 0.8063594102859497),
 ('sense', 0.7779642939567566),
 ('grasp', 0.7166379690170288),
 ('pie', 0.708033561706543),
 ('weather', 0.7012685537338257),
 ('work', 0.6996005177497864),
 ('cheaper', 0.698716402053833),
 ('ranger', 0.6964582800865173),
 ('encounters', 0.6961168050765991),
 ('incidentally', 0.6926853656768799)]
```


<br>


어떻습니까? **love** 라는 단어와 유사한 다른 단어를 그리 잘 찾았다고 느껴지지는 않습니다. 감성 분류 태스크를 잠깐 학습한 것만으로 워드 벡터가 유의미하게 학습되기는 어려운 것 같습니다. 우리가 다룬 정도의 훈련 데이터로는 워드 벡터를 정교하게 학습시키기 어렵습니다.

그래서 이번에는 구글에서 제공하는 **<span style="color:red">Word2Vec</span>** 이라는 **<u>사전학습된(Pretrained) 워드 임베딩 모델을 가져다 활용</u>** 해 보겠습니다. **Word2Vec** 은 무려 1억 개의 단어로 구성된 Google News dataset을 바탕으로 학습되었습니다. 총 300만 개의 단어를 각각 300차원의 벡터로 표현한 것입니다. **Word2Vec** 이 학습되는 원리에 대해서는 차후 깊이 있게 다루게 될 것입니다. 하지만 그렇게 해서 학습된 **Word2Vec** 이라는 것도 실은 방금 우리가 파일에 써본 **Embedding Layer** 와 원리는 동일합니다.


<br>

그러면 본격적으로 Google의 **Word2Vec** 모델을 가져와 적용해 봅시다.
```py
from gensim.models import KeyedVectors

word2vec_path = '/root/share/aiffel-data/sentiment_classification/data/GoogleNews-vectors-negative300.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=1000000)
vector = word2vec['computer']
vector     # 무려 300dim의 워드 벡터입니다.
```
```
[output]
array([ 1.07421875e-01, -2.01171875e-01,  1.23046875e-01,  2.11914062e-01,
       -9.13085938e-02,  2.16796875e-01, -1.31835938e-01,  8.30078125e-02,
        2.02148438e-01,  4.78515625e-02,  3.66210938e-02, -2.45361328e-02,
        2.39257812e-02, -1.60156250e-01, -2.61230469e-02,  9.71679688e-02,
       -6.34765625e-02,  1.84570312e-01,  1.70898438e-01, -1.63085938e-01,
       -1.09375000e-01,  1.49414062e-01, -4.65393066e-04,  9.61914062e-02,
        1.68945312e-01,  2.60925293e-03,  8.93554688e-02,  6.49414062e-02,
        3.56445312e-02, -6.93359375e-02, -1.46484375e-01, -1.21093750e-01,
       -2.27539062e-01,  2.45361328e-02, -1.24511719e-01, -3.18359375e-01,
       -2.20703125e-01,  1.30859375e-01,  3.66210938e-02, -3.63769531e-02,
       -1.13281250e-01,  1.95312500e-01,  9.76562500e-02,  1.26953125e-01,
        6.59179688e-02,  6.93359375e-02,  1.02539062e-02,  1.75781250e-01,
       -1.68945312e-01,  1.21307373e-03, -2.98828125e-01, -1.15234375e-01,
        5.66406250e-02, -1.77734375e-01, -2.08984375e-01,  1.76757812e-01,
        2.38037109e-02, -2.57812500e-01, -4.46777344e-02,  1.88476562e-01,
        5.51757812e-02,  5.02929688e-02, -1.06933594e-01,  1.89453125e-01,
       -1.16210938e-01,  8.49609375e-02, -1.71875000e-01,  2.45117188e-01,
       -1.73828125e-01, -8.30078125e-03,  4.56542969e-02, -1.61132812e-02,
        1.86523438e-01, -6.05468750e-02, -4.17480469e-02,  1.82617188e-01,
        2.20703125e-01, -1.22558594e-01, -2.55126953e-02, -3.08593750e-01,
        9.13085938e-02,  1.60156250e-01,  1.70898438e-01,  1.19628906e-01,
        7.08007812e-02, -2.64892578e-02, -3.08837891e-02,  4.06250000e-01,
       -1.01562500e-01,  5.71289062e-02, -7.26318359e-03, -9.17968750e-02,
       -1.50390625e-01, -2.55859375e-01,  2.16796875e-01, -3.63769531e-02,
        2.24609375e-01,  8.00781250e-02,  1.56250000e-01,  5.27343750e-02,
        1.50390625e-01, -1.14746094e-01, -8.64257812e-02,  1.19140625e-01,
       -7.17773438e-02,  2.73437500e-01, -1.64062500e-01,  7.29370117e-03,
        4.21875000e-01, -1.12792969e-01, -1.35742188e-01, -1.31835938e-01,
       -1.37695312e-01, -7.66601562e-02,  6.25000000e-02,  4.98046875e-02,
       -1.91406250e-01, -6.03027344e-02,  2.27539062e-01,  5.88378906e-02,
       -3.24218750e-01,  5.41992188e-02, -1.35742188e-01,  8.17871094e-03,
       -5.24902344e-02, -1.74713135e-03, -9.81445312e-02, -2.86865234e-02,
        3.61328125e-02,  2.15820312e-01,  5.98144531e-02, -3.08593750e-01,
       -2.27539062e-01,  2.61718750e-01,  9.86328125e-02, -5.07812500e-02,
        1.78222656e-02,  1.31835938e-01, -5.35156250e-01, -1.81640625e-01,
        1.38671875e-01, -3.10546875e-01, -9.71679688e-02,  1.31835938e-01,
       -1.16210938e-01,  7.03125000e-02,  2.85156250e-01,  3.51562500e-02,
       -1.01562500e-01, -3.75976562e-02,  1.41601562e-01,  1.42578125e-01,
       -5.68847656e-02,  2.65625000e-01, -2.09960938e-01,  9.64355469e-03,
       -6.68945312e-02, -4.83398438e-02, -6.10351562e-02,  2.45117188e-01,
       -9.66796875e-02,  1.78222656e-02, -1.27929688e-01, -4.78515625e-02,
       -7.26318359e-03,  1.79687500e-01,  2.78320312e-02, -2.10937500e-01,
       -1.43554688e-01, -1.27929688e-01,  1.73339844e-02, -3.60107422e-03,
       -2.04101562e-01,  3.63159180e-03, -1.19628906e-01, -6.15234375e-02,
        5.93261719e-02, -3.23486328e-03, -1.70898438e-01, -3.14941406e-02,
       -8.88671875e-02, -2.89062500e-01,  3.44238281e-02, -1.87500000e-01,
        2.94921875e-01,  1.58203125e-01, -1.19628906e-01,  7.61718750e-02,
        6.39648438e-02, -4.68750000e-02, -6.83593750e-02,  1.21459961e-02,
       -1.44531250e-01,  4.54101562e-02,  3.68652344e-02,  3.88671875e-01,
        1.45507812e-01, -2.55859375e-01, -4.46777344e-02, -1.33789062e-01,
       -1.38671875e-01,  6.59179688e-02,  1.37695312e-01,  1.14746094e-01,
        2.03125000e-01, -4.78515625e-02,  1.80664062e-02, -8.54492188e-02,
       -2.48046875e-01, -3.39843750e-01, -2.83203125e-02,  1.05468750e-01,
       -2.14843750e-01, -8.74023438e-02,  7.12890625e-02,  1.87500000e-01,
       -1.12304688e-01,  2.73437500e-01, -3.26171875e-01, -1.77734375e-01,
       -4.24804688e-02, -2.69531250e-01,  6.64062500e-02, -6.88476562e-02,
       -1.99218750e-01, -7.03125000e-02, -2.43164062e-01, -3.66210938e-02,
       -7.37304688e-02, -1.77734375e-01,  9.17968750e-02, -1.25000000e-01,
       -1.65039062e-01, -3.57421875e-01, -2.85156250e-01, -1.66992188e-01,
        1.97265625e-01, -1.53320312e-01,  2.31933594e-02,  2.06054688e-01,
        1.80664062e-01, -2.74658203e-02, -1.92382812e-01, -9.61914062e-02,
       -1.06811523e-02, -4.73632812e-02,  6.54296875e-02, -1.25732422e-02,
        1.78222656e-02, -8.00781250e-02, -2.59765625e-01,  9.37500000e-02,
       -7.81250000e-02,  4.68750000e-02, -2.22167969e-02,  1.86767578e-02,
        3.11279297e-02,  1.04980469e-02, -1.69921875e-01,  2.58789062e-02,
       -3.41796875e-02, -1.44042969e-02, -5.46875000e-02, -8.78906250e-02,
        1.96838379e-03,  2.23632812e-01, -1.36718750e-01,  1.75781250e-01,
       -1.63085938e-01,  1.87500000e-01,  3.44238281e-02, -5.63964844e-02,
       -2.27689743e-05,  4.27246094e-02,  5.81054688e-02, -1.07910156e-01,
       -3.88183594e-02, -2.69531250e-01,  3.34472656e-02,  9.81445312e-02,
        5.63964844e-02,  2.23632812e-01, -5.49316406e-02,  1.46484375e-01,
        5.93261719e-02, -2.19726562e-01,  6.39648438e-02,  1.66015625e-02,
        4.56542969e-02,  3.26171875e-01, -3.80859375e-01,  1.70898438e-01,
        5.66406250e-02, -1.04492188e-01,  1.38671875e-01, -1.57226562e-01,
        3.23486328e-03, -4.80957031e-02, -2.48046875e-01, -6.20117188e-02],
      dtype=float32)
```


<br>



300dim의 벡터로 이루어진 300만 개의 단어입니다. 이 단어 사전을 메모리에 모두 로딩하면 아주 높은 확률로 여러분의 실습환경에 메모리 에러가 날 것입니다. 그래서 **KeyedVectors.load_word2vec_format** 메서드로 워드 벡터를 로딩할 때 가장 많이 사용되는 상위 100만 개만 limt으로 조건을 주어 로딩했습니다.

메모리가 충분하다면 `limt=None` 으로 하시면 300만 개를 모두 로딩합니다.
```py
# 메모리를 다소 많이 소비하는 작업이니 유의해 주세요.
word2vec.similar_by_word("love")
```
```
[output]
[('loved', 0.6907791495323181),
 ('adore', 0.6816873550415039),
 ('loves', 0.661863386631012),
 ('passion', 0.6100708842277527),
 ('hate', 0.600395679473877),
 ('loving', 0.5886635780334473),
 ('Ilove', 0.5702950954437256),
 ('affection', 0.5664337873458862),
 ('undying_love', 0.5547304749488831),
 ('absolutely_adore', 0.5536840558052063)]
```


<br>


어떻습니까? **<span style="color:red">Word2Vec에서 제공하는 워드 임베딩 벡터들끼리는 의미적 유사도가 가까운 것이 서로 가깝게 제대로 학습된 것을 확인</span>** 할 수 있습니다. 이제 우리는 이전 스텝에서 학습했던 모델의 임베딩 레이어를 **Word2Vec** 의 것으로 교체하여 다시 학습시켜 볼 것입니다.
```py
vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 300  # 워드 벡터의 차원수
embedding_matrix = np.random.rand(vocab_size, word_vector_dim)
print(embedding_matrix.shape)

# embedding_matrix에 Word2Vec 워드 벡터를 단어 하나씩마다 차례차례 카피한다.
for i in range(4, vocab_size):
    if index_to_word[i] in word2vec:
        embedding_matrix[i] = word2vec[index_to_word[i]]
```
```
[output]
(10000, 300)
```


<br>

```py
from tensorflow.keras.initializers import Constant

vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 300  # 워드 벡터의 차원 수 

# 모델 구성
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Embedding(
        vocab_size,
        word_vector_dim,
        embeddings_initializer=Constant(embedding_matrix),  # 카피한 임베딩을 여기서 활용
        input_length=maxlen,
        trainable=True
    )
)   # trainable을 True로 주면 Fine-tuning
model.add(tf.keras.layers.Conv1D(16, 7, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(5))
model.add(tf.keras.layers.Conv1D(16, 7, activation='relu'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 

model.summary()
```
```
[output]
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_5 (Embedding)     (None, 580, 300)          3000000   
                                                                 
 conv1d_2 (Conv1D)           (None, 574, 16)           33616     
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 114, 16)          0         
 1D)                                                             
                                                                 
 conv1d_3 (Conv1D)           (None, 108, 16)           1808      
                                                                 
 global_max_pooling1d_2 (Glo  (None, 16)               0         
 balMaxPooling1D)                                                
                                                                 
 dense_8 (Dense)             (None, 8)                 136       
                                                                 
 dense_9 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 3,035,569
Trainable params: 3,035,569
Non-trainable params: 0
_________________________________________________________________
```


<br>

```py
# 학습의 진행
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
              
epochs=20  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=epochs,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)
```
```
[output]

```


<br>


```py
# 테스트셋을 통한 모델 평가
results = model.evaluate(x_test,  y_test, verbose=2)
print(results)
```
```
[output]
782/782 - 3s - loss: 0.5146 - accuracy: 0.8620 - 3s/epoch - 4ms/step
[0.5145856142044067, 0.8619999885559082]
```


<br>


```py
import matplotlib.pyplot as plt

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/279284a0-2e0e-4c4b-b1e7-d2b4b31b88ab">
</p>

<br>


```py
plt.clf()   # 그림을 초기화합니다

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/49a8f2c9-da82-4ad0-bb75-6838558cefed">
</p>

<br>




**Word2Vec** 을 정상적으로 잘 활용하면 그렇지 않은 경우보다 약 30% 이상의 성능 향상이 발생합니다.






