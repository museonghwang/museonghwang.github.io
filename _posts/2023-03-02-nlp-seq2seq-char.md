---
layout: post
title: Seq2seq Character‐Level 번역기(NMT) 만들기
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





**seq2seq** 를 이용해서 기계 번역기를 만들어보겠습니다. 실제 성능이 좋은 기계 번역기를 구현하려면 정말 방대한 데이터가 필요하므로 여기서는 **seq2seq** 를 실습해보는 수준에서 아주 간단한 기계 번역기를 구축해보겠습니다. 기계 번역기를 훈련시키기 위해서는 훈련 데이터로 **<span style="color:red">병렬 코퍼스(parallel corpus) 데이터</span>** 가 필요합니다. 병렬 코퍼스란, **<u>두 개 이상의 언어가 병렬적으로 구성된 코퍼스를 의미</u>** 합니다.

- [다운로드 링크](http://www.manythings.org/anki)

<br>

본 실습에서는 프랑스-영어 병렬 코퍼스인 **fra-eng.zip** 파일을 사용하겠습니다. 위 링크에서 해당 파일을 다운받으면 됩니다. 해당 파일의 압축을 풀면 **fra.txt** 라는 파일이 있는데 이 파일이 이번 실습에서 사용할 파일입니다.

<br>
<br>





# 1. 병렬 코퍼스 데이터에 대한 이해와 전처리

우선 **병렬 코퍼스 데이터** 에 대한 이해를 해보겠습니다. 태깅 작업의 병렬 데이터는 쌍이 되는 모든 데이터가 길이가 같았지만 여기서는 쌍이 된다고 해서 길이가 같지않습니다. 실제 번역기를 생각해보면 구글 번역기에 '나는 학생이다.'라는 토큰의 개수가 2인 문장을 넣었을 때 'I am a student.'라는 토큰의 개수가 4인 문장이 나오는 것과 같은 이치입니다.

**seq2seq는 기본적으로 입력 시퀀스와 출력 시퀀스의 길이가 다를 수 있다고 가정합니다.** 지금은 기계 번역기가 예제지만 seq2seq의 또 다른 유명한 예제 중 하나인 챗봇을 만든다고 가정해보면, 대답의 길이가 질문의 길이와 항상 똑같아야 한다고하면 그 또한 이상합니다.
```
Watch me.   Regardez-moi !
```

<br>


여기서 사용할 **fra.txt** 데이터는 위와 같이 왼쪽의 영어 문장과 오른쪽의 프랑스어 문장 사이에 탭으로 구분되는 구조가 하나의 샘플입니다. 그리고 이와 같은 형식의 약 21만개의 병렬 문장 샘플을 포함하고 있습니다. 해당 데이터를 다운받고, 읽고, 전처리를 진행해보겠습니다.
```py
import os
import urllib3
import zipfile
import shutil

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)


lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
print('전체 샘플의 개수 :',len(lines))
```
```
[output]
전체 샘플의 개수 : 217975
```


<br>



해당 데이터는 약 21만 7천개의 병렬 문장 샘플로 구성되어있지만 여기서는 간단히 60,000개의 샘플만 가지고 기계 번역기를 구축해보도록 하겠습니다. 우선 전체 데이터 중 60,000개의 샘플만 저장하고 현재 데이터가 어떤 구성이 되었는지 확인해보겠습니다.
```py
lines = lines.loc[:, 'src':'tar']
lines = lines[0:60000] # 6만개만 저장
lines.sample(10)
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/8347b281-35f4-4461-90b6-9601d26db095">
</p>

<br>


위의 테이블은 랜덤으로 선택된 10개의 샘플을 보여줍니다. **번역 문장에 해당되는 프랑스어 데이터** 는 시작을 의미하는 심볼 **\<sos\>** 과 종료를 의미하는 심볼 **\<eos\>** 을 넣어주어야 합니다. 여기서는 Character‐Level 이므로 **\<sos\>** 와 **\<eos\>** 대신 **'\t'** 를 시작 심볼, **'\n'** 을 종료 심볼로 간주하여 추가하고 다시 데이터를 출력해보겠습니다.
```py
lines.tar = lines.tar.apply(lambda x : '\t ' + x + ' \n')
lines.sample(10)
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/5ee7de62-13ad-4b64-a101-ef2250f02772">
</p>

<br>


**프랑스어 데이터에서 시작 심볼과 종료 심볼이 추가된 것** 을 볼 수 있습니다. **<span style="color:red">문자 집합을 생성</span>** 하고 **<span style="color:red">문자 집합의 크기</span>** 를 보겠습니다. 단어 집합이 아니라 문자 집합이라고 하는 이유는 **<u>토큰 단위가 단어가 아니라 문자이기 때문</u>** 입니다.
```py
# 글자 집합 구축
src_vocab = set()
for line in lines.src:  # 1줄씩 읽음
    for char in line:   # 1개의 글자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab) + 1
tar_vocab_size = len(tar_vocab) + 1
print('source 문장의 char 집합 :', src_vocab_size)
print('target 문장의 char 집합 :', tar_vocab_size)
```
```
[output]
source 문장의 char 집합 : 80
target 문장의 char 집합 : 103
```


<br>

**영어와 프랑스어는 각각 80개와 103개의 문자가 존재** 합니다. 이 중에서 인덱스를 임의로 부여하여 일부만 출력하겠습니다. set() 함수 안에 문자들이 있으므로 현 상태에서 인덱스를 사용하려고하면 에러가 납니다. 하지만 정렬하여 순서를 정해준 뒤에 인덱스를 사용하여 출력해주면 됩니다.
```py
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
print(src_vocab[45:75])
print(tar_vocab[45:75])
```
```
[output]
['U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x']
['Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
```


<br>



문자 집합에 문자 단위로 저장된 것을 확인할 수 있습니다. **각 문자에 인덱스를 부여** 하겠습니다.
```py
src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
print(src_to_index)
print(tar_to_index)
```
```
[output]
{' ': 1, '!': 2, '"': 3, '$': 4, '%': 5, ... 중략 ..., '—': 85, '‘': 86, '’': 87, '₂': 88, '€': 89}
{'\t': 1, '\n': 2, ' ': 3, '!': 4, '"': 5, ... 중략 ..., '’': 111, '…': 112, '\u202f': 113, '‽': 114, '₂': 115}
```


<br>



**<span style="color:red">인덱스가 부여된 문자 집합으로부터 갖고있는 훈련 데이터에 정수 인코딩을 수행</span>** 합니다. 우선 인코더의 입력이 될 영어 문장 샘플에 대해서 정수 인코딩을 수행해보고, 5개의 샘플을 출력해보겠습니다. 또한 확인차 훈련 데이터 5개도 출력하여 매칭되는지 확인하겠습니다.
```py
lines.src[:5]
```
```
[output]
0    Go.
1    Go.
2    Go.
3    Go.
4    Hi.
```


<br>

```py
encoder_input = []

# 1개의 문장
for line in lines.src:
    encoded_line = []
    # 각 줄에서 1개의 char
    for char in line:
        # 각 char을 정수로 변환
        encoded_line.append(src_to_index[char])
    encoder_input.append(encoded_line)

print('source 문장의 정수 인코딩 :', encoder_input[:5])
```
```
[output]
source 문장의 정수 인코딩 : [[32, 66, 11], [32, 66, 11], [32, 66, 11], [32, 66, 11], [33, 60, 11]]
```


<br>


정수 인코딩이 수행된 것을 볼 수 있습니다. **<span style="color:red">디코더의 입력이 될 프랑스어 데이터에 대해서 정수 인코딩을 수행</span>** 해보겠습니다.
```py
lines.tar[:5]
```
```
[output]
0          \t Va ! \n
1       \t Marche. \n
2    \t En route ! \n
3       \t Bouge ! \n
4       \t Salut ! \n
```


<br>

```py
decoder_input = []

for line in lines.tar:
    encoded_line = []
    for char in line:
        encoded_line.append(tar_to_index[char])
    decoder_input.append(encoded_line)

print('target 문장의 정수 인코딩 :', decoder_input[:5])
```
```
[output]
target 문장의 정수 인코딩 : [[1, 3, 51, 56, 3, 4, 3, 2], [1, 3, 42, 56, 73, 58, 63, 60, 15, 3, 2], [1, 3, 34, 69, 3, 73, 70, 76, 75, 60, 3, 4, 3, 2], [1, 3, 31, 70, 76, 62, 60, 3, 4, 3, 2], [1, 3, 48, 56, 67, 76, 75, 3, 4, 3, 2]]
```


<br>



정상적으로 정수 인코딩이 수행된 것을 볼 수 있습니다. 아직 정수 인코딩을 수행해야 할 데이터가 하나 더 남았습니다. **<span style="color:red">디코더의 예측값과 비교하기 위한 실제값이 필요합니다. 하지만 이 실제값에는 시작 심볼에 해당되는 \<sos\> 가 있을 필요가 없습니다.</span>** 그래서 이번에는 정수 인코딩 과정에서 **\<sos\>** 를 제거합니다. **<span style="color:red">즉, 모든 프랑스어 문장의 맨 앞에 붙어있는 '\t'를 제거</span>** 하도록 합니다.
```py
decoder_target = []

for line in lines.tar:
    timestep = 0
    encoded_line = []
    for char in line:
        if timestep > 0:
            encoded_line.append(tar_to_index[char])
        timestep = timestep + 1
    decoder_target.append(encoded_line)
    
print('target 문장 레이블의 정수 인코딩 :', decoder_target[:5])
```
```
[output]
target 문장 레이블의 정수 인코딩 : [[3, 51, 56, 3, 4, 3, 2], [3, 42, 56, 73, 58, 63, 60, 15, 3, 2], [3, 34, 69, 3, 73, 70, 76, 75, 60, 3, 4, 3, 2], [3, 31, 70, 76, 62, 60, 3, 4, 3, 2], [3, 48, 56, 67, 76, 75, 3, 4, 3, 2]]
```


<br>


앞서 먼저 만들었던 디코더의 입력값에 해당되는 **decoder_input** 데이터와 비교하면 **<u>decoder_input에서는 모든 문장의 앞에 붙어있던 숫자 1이 decoder_target에서는 제거된 것을 볼 수 있습니다.</u>** **'\t'** 가 인덱스가 1이므로 정상적으로 제거된 것입니다.

모든 데이터에 대해서 정수 인덱스로 변경하였으니 **<span style="color:red">패딩 작업을 수행</span>** 합니다. 패딩을 위해서 영어 문장과 프랑스어 문장 각각에 대해서 가장 길이가 긴 샘플의 길이를 확인합니다.
```py
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print('source 문장의 최대 길이 :', max_src_len)
print('target 문장의 최대 길이 :', max_tar_len)
```
```
[output]
source 문장의 최대 길이 : 22
target 문장의 최대 길이 : 76
```


<br>

각각 22와 76의 길이를 가집니다. **<u>이번 병렬 데이터는 영어와 프랑스어의 길이는 하나의 쌍이라고 하더라도 전부 다르므로 패딩을 할 때도 이 두 개의 데이터의 길이를 전부 동일하게 맞춰줄 필요는 없습니다.</u>** **<span style="color:red">영어 데이터는 영어 샘플들끼리, 프랑스어는 프랑스어 샘플들끼리 길이를 맞추어서 패딩</span>** 하면 됩니다. 여기서는 가장 긴 샘플의 길이에 맞춰서 영어 데이터의 샘플은 전부 길이가 22이 되도록 패딩하고, 프랑스어 데이터의 샘플은 전부 길이가 76이 되도록 패딩합니다.
```py
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

print('shape of encoder_input : ', encoder_input.shape)
print('shape of decoder_input : ', decoder_input.shape)
print('shape of decoder_target : ', decoder_target.shape)
```
```
[output]
shape of encoder_input :  (60000, 22)
shape of decoder_input :  (60000, 76)
shape of decoder_target :  (60000, 76)
```


<br>



모든 값에 대해서 **<span style="color:red">원-핫 인코딩을 수행</span>**합니다. **<u>문자 단위 번역기므로 워드 임베딩은 별도로 사용되지 않으며, 예측값과의 오차 측정에 사용되는 실제값뿐만 아니라 입력값도 원-핫 벡터를 사용</u>** 하겠습니다.
```py
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

print('shape of encoder_input : ', encoder_input.shape)
print('shape of decoder_input : ', decoder_input.shape)
print('shape of decoder_target : ', decoder_target.shape)
```
```
[output]
shape of encoder_input :  (60000, 22, 80)
shape of decoder_input :  (60000, 76, 103)
shape of decoder_target :  (60000, 76, 103)
```


<br>



데이터에 대한 전처리가 모두 끝났습니다. 본격적으로 **seq2seq 모델을 설계** 해보겠습니다.

<br>





# 2. 교사 강요(Teacher forcing)

모델 설계에 있어서 **<span style="color:red">decoder_input이 필요한 이유</span>** 를 살펴보겠습니다.


테스트 과정에서 현재 시점의 디코더 셀의 입력은 오직 이전 디코더 셀의 출력을 입력으로 받습니다. 하지만 **<span style="color:red">훈련 과정</span>** 에서는 **<u>이전 시점의 디코더 셀의 출력을 현재 시점의 디코더 셀의 입력으로 넣어주지 않고</u>**, **<span style="color:red">이전 시점의 실제값을 현재 시점의 디코더 셀의 입력값으로 하는 방법을 사용</span>** 합니다.

<br>

그 **<span style="color:red">이유</span>** 는 **<span style="background-color: #fff5b1">이전 시점의 디코더 셀의 예측이 틀렸는데 이를 현재 시점의 디코더 셀의 입력으로 사용하면 현재 시점의 디코더 셀의 예측도 잘못될 가능성이 높고 이는 연쇄 작용으로 디코더 전체의 예측을 어렵게 합니다.</span>** 이런 상황이 반복되면 훈련 시간이 느려집니다.

만약 이 상황을 원하지 않는다면 이전 시점의 디코더 셀의 예측값 대신 실제값을 현재 시점의 디코더 셀의 입력으로 사용하는 방법을 사용할 수 있습니다. 이와 같이 **<span style="color:red">RNN의 모든 시점에 대해서 이전 시점의 예측값 대신 실제값을 입력으로 주는 방법</span>** 을 **<span style="color:red">교사 강요</span>** 라고 합니다.


<br>




# 3. seq2seq 기계 번역기 훈련시키기

**seq2seq 모델을 설계** 하고 **교사 강요를 사용** 하여 훈련시켜보도록 하겠습니다.
```py
import numpy as np
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# src_vocab_size=80
# shape of encoder_inputs : (None, None, 80)
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)

# encoder_outputs은 여기서는 불필요
# shape of encoder_outputs : (None, 256)
# shape of state_h : (None, 256)
# shape of state_c : (None, 256)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
encoder_states = [state_h, state_c]
```


<br>


인코더를 보면 기본 **LSTM** 설계와 크게 다르지는 않습니다. 우선 **LSTM** 의 은닉 상태 크기는 256으로 선택하였으며, **<u>인코더의 내부 상태를 디코더로 넘겨주어야 하기 때문에 return_state=True로 설정</u>** 합니다.

<br>

**인코더(encoder_lstm)** 에 입력을 넣으면 내부 상태, 즉 **LSTM** 에서 **state_h**, **state_c** 를 리턴받는데, 이는 각각 **은닉 상태** 와 **셀 상태** 에 해당됩니다. **<span style="color:red">은닉 상태</span>** 와 **<span style="color:red">셀 상태</span>** 두 가지를 전달한다고 생각하면 됩니다. **<u>이 두 가지 상태를 encoder_states에 저장</u>** 합니다.

**<u>encoder_states를 디코더에 전달하므로서 이 두 가지 상태 모두를 디코더로 전달</u>** 합니다. 이것이 **<span style="color:red">컨텍스트 벡터</span>** 입니다.
```py
# tar_vocab_size=103
# shape of decoder_inputs : (None, None, 103)
decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달.
# shape of decoder_outputs : (None, None, 256)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# shape of decoder_outputs : (None, None, 103)
decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)
```


<br>


**<span style="color:red">디코더는 인코더의 마지막 은닉 상태를 초기 은닉 상태로 사용</span>** 합니다. 위에서 **initial_state** 의 인자값으로 **encoder_states** 를 주는 코드가 이에 해당됩니다. 또한 동일하게 디코더의 은닉 상태 크기도 256으로 주었습니다.

**<u>디코더도 은닉 상태, 셀 상태를 리턴하기는 하지만 훈련 과정에서는 사용하지 않습니다.</u>** 그 후 출력층에 프랑스어의 **단어 집합의 크기만큼 뉴런을 배치한 후 소프트맥스 함수를 사용하여 실제값과의 오차를 구합니다.**
```py
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy"
)

model.fit(
    x=[encoder_input, decoder_input],
    y=decoder_target,
    batch_size=64,
    epochs=50,
    validation_split=0.2
)
```
```
[output]
Epoch 1/50
750/750 [==============================] - 17s 20ms/step - loss: 0.7379 - val_loss: 0.6436
Epoch 2/50
750/750 [==============================] - 16s 21ms/step - loss: 0.4455 - val_loss: 0.5092
Epoch 3/50
750/750 [==============================] - 15s 20ms/step - loss: 0.3706 - val_loss: 0.4474
Epoch 4/50
750/750 [==============================] - 15s 21ms/step - loss: 0.3287 - val_loss: 0.4096
.
.
.
Epoch 47/50
750/750 [==============================] - 15s 19ms/step - loss: 0.1198 - val_loss: 0.4011
Epoch 48/50
750/750 [==============================] - 16s 21ms/step - loss: 0.1186 - val_loss: 0.4044
Epoch 49/50
750/750 [==============================] - 15s 21ms/step - loss: 0.1175 - val_loss: 0.4063
Epoch 50/50
750/750 [==============================] - 15s 20ms/step - loss: 0.1163 - val_loss: 0.4115
```


<br>



**<span style="color:red">입력</span>** 으로는 **<span style="color:red">인코더 입력</span>** 과 **<span style="color:red">디코더 입력</span>** 이 들어가고, **<span style="color:red">디코더의 실제값인 decoder_target</span>** 도 필요합니다. 배치 크기는 64로 하였으며 총 50 에포크를 학습합니다.

위에서 설정한 은닉 상태의 크기와 에포크 수는 실제로는 훈련 데이터에 과적합 상태를 불러오지만, 여기서는 우선 seq2seq의 메커니즘과 짧은 문장과 긴 문장에 대한 성능 차이에 대한 확인을 중점으로 두고 훈련 데이터에 과적합 된 상태로 동작 단계로 넘어갑니다.

<br>





# 4. seq2seq 기계 번역기 동작시키기

앞서 seq2seq는 훈련할 때와 동작할 때의 방식이 다르다고 언급한 바 있습니다. 이번에는 **입력한 문장에 대해서 기계 번역을 하도록 모델을 조정하고 동작** 시켜보도록 하겠습니다.

전체적인 번역 동작 단계를 정리하면 아래와 같습니다.

1. 번역하고자 하는 입력 문장이 인코더에 들어가서 은닉 상태와 셀 상태를 얻습니다.
2. 상태와 **\<SOS\>** 에 해당하는 **'\t'** 를 디코더로 보냅니다.
3. 디코더가 **\<EOS\>** 에 해당하는 **'\n'** 이 나올 때까지 다음 문자를 예측하는 행동을 반복합니다.

<br>

우선 **<span style="color:red">인코더를 정의</span>**합니다. **encoder_inputs** 와 **encoder_states** 는 훈련 과정에서 이미 정의한 것들을 재사용하는 것입니다.
```py
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
encoder_model.summary()
```
```
[output]
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, None, 80)]        0         
                                                                 
 lstm (LSTM)                 [(None, 256),             345088    
                              (None, 256),                       
                              (None, 256)]                       
                                                                 
=================================================================
Total params: 345,088
Trainable params: 345,088
Non-trainable params: 0
_________________________________________________________________
```


<br>



**<span style="color:red">디코더를 설계</span>** 해보겠습니다.
```py
# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용.
# 뒤의 함수 decode_sequence()에 동작을 구현 예정
# shape of decoder_inputs : (None, None, 103)
# shape of decoder_outputs : (None, None, 256)
# shape of state_h : (None, 256)
# shape of state_c : (None, 256)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태를 버리지 않음.
# shape of decoder_outputs : (None, None, 103)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)

decoder_model = Model(
    inputs=[decoder_inputs] + decoder_states_inputs,
    outputs=[decoder_outputs] + decoder_states
)
```
```py
decoder_model.summary()
```
```
[output]
Model: "model_2"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_2 (InputLayer)           [(None, None, 103)]  0           []                               
                                                                                                  
 input_3 (InputLayer)           [(None, 256)]        0           []                               
                                                                                                  
 input_4 (InputLayer)           [(None, 256)]        0           []                               
                                                                                                  
 lstm_1 (LSTM)                  [(None, None, 256),  368640      ['input_2[0][0]',                
                                 (None, 256),                     'input_3[0][0]',                
                                 (None, 256)]                     'input_4[0][0]']                
                                                                                                  
 dense (Dense)                  (None, None, 103)    26471       ['lstm_1[1][0]']                 
                                                                                                  
==================================================================================================
Total params: 395,111
Trainable params: 395,111
Non-trainable params: 0
__________________________________________________________________________________________________
```


<br>


```py
index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())
```


<br>





단어로부터 인덱스를 얻는 것이 아니라 인덱스로부터 단어를 얻을 수 있는 **index_to_src** 와 **index_to_tar** 를 만들었습니다.
```py
def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    states_value = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ""

    # stop_condition이 True가 될 때까지 루프 반복
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # 현재 시점의 예측 문자를 예측 문장에 추가
        decoded_sentence += sampled_char

        # <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_char == '\n' or
            len(decoded_sentence) > max_tar_len):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1, 1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence
```
```py
for seq_index in [3, 50, 100, 300, 1001]: # 입력 문장의 인덱스
    input_seq = encoder_input[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    
    print(35 * "-")
    print('입력 문장:', lines.src[seq_index])
    print('정답 문장:', lines.tar[seq_index][2:len(lines.tar[seq_index])-1]) # '\t'와 '\n'을 빼고 출력
    print('번역 문장:', decoded_sentence[1:len(decoded_sentence)-1]) # '\n'을 빼고 출력
```
```
[output]
-----------------------------------
입력 문장: Go.
정답 문장: Bouge ! 
번역 문장: Décampe ! 
-----------------------------------
입력 문장: Hello!
정답 문장: Bonjour ! 
번역 문장: Bonjour ! 
-----------------------------------
입력 문장: Got it!
정답 문장: Compris ! 
번역 문장: Compris ! 
-----------------------------------
입력 문장: Goodbye.
정답 문장: Au revoir. 
번역 문장: Casse-toi. 
-----------------------------------
입력 문장: Hands off.
정답 문장: Pas touche ! 
번역 문장: Va ! 
```


<br>

지금까지 문자 단위의 seq2seq를 구현하였습니다.





