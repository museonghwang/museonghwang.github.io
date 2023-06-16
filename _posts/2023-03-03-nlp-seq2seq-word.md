---
layout: post
title: Seq2seq Word‐Level 번역기(NMT) 만들기
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


여기서 사용할 **fra.txt** 데이터는 위와 같이 왼쪽의 영어 문장과 오른쪽의 프랑스어 문장 사이에 탭으로 구분되는 구조가 하나의 샘플입니다. 그리고 이와 같은 형식의 약 21만개의 병렬 문장 샘플을 포함하고 있습니다. 해당 데이터를 다운받고, 읽고, 전처리를 진행해보겠습니다. **fra-eng.zip** 파일을 다운로드하고 압축을 풀겠습니다.
```py
import re
import os
import unicodedata
import urllib3
import zipfile
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)
```


<br>



본 실습에서는 약 21만개의 데이터 중 **33,000개의 샘플** 만을 사용하겠습니다.
```py
num_samples = 33000
```


<br>


**전처리 함수** 들을 구현하겠습니다. 구두점 등을 제거하거나 단어와 구분해주기 위한 전처리입니다.
```py
def unicode_to_ascii(s):
    # 프랑스어 악센트(accent) 삭제
    # 예시 : 'déjà diné' -> deja dine
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sent):
    # 악센트 삭제 함수 호출
    sent = unicode_to_ascii(sent.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sent = re.sub(r"([?.!,¿])", r" \1", sent)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

    # 다수 개의 공백을 하나의 공백으로 치환
    sent = re.sub(r"\s+", " ", sent)
    
    return sent
```


<br>




구현한 전처리 함수들을 임의의 문장을 입력으로 테스트해보겠습니다.
```py
# 전처리 테스트
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous déjà diné?"

print('전처리 전 영어 문장 :', en_sent)
print('전처리 후 영어 문장 :', preprocess_sentence(en_sent))
print('전처리 전 프랑스어 문장 :', fr_sent)
print('전처리 후 프랑스어 문장 :', preprocess_sentence(fr_sent))
```
```
[output]
전처리 전 영어 문장 : Have you had dinner?
전처리 후 영어 문장 : have you had dinner ?
전처리 전 프랑스어 문장 : Avez-vous déjà diné?
전처리 후 프랑스어 문장 : avez vous deja dine ?
```


<br>



**전체 데이터에서 33,000개의 샘플에 대해서 전처리를 수행** 합니다. 또한 훈련 과정에서 **교사 강요(Teacher Forcing)** 을 사용할 예정이므로, **<u>훈련 시 사용할 디코더의 입력 시퀀스와 실제값. 즉, 레이블에 해당되는 출력 시퀀스를 따로 분리하여 저장</u>** 합니다. **<span style="color:red">입력 시퀀스</span>** 에는 시작을 의미하는 토큰인 **\<sos\>** 를 추가하고, **<span style="color:red">출력 시퀀스</span>** 에는 종료를 의미하는 토큰인 **\<eos\>** 를 추가합니다.
```py
def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []

    with open("fra.txt", "r") as lines:
        for i, line in enumerate(lines):
            # source 데이터와 target 데이터 분리
            src_line, tar_line, _ = line.strip().split('\t')

            # source 데이터 전처리
            src_line = [w for w in preprocess_sentence(src_line).split()]

            # target 데이터 전처리
            tar_line = preprocess_sentence(tar_line)
            tar_line_in = [w for w in ("<sos> " + tar_line).split()]
            tar_line_out = [w for w in (tar_line + " <eos>").split()]

            encoder_input.append(src_line)
            decoder_input.append(tar_line_in)
            decoder_target.append(tar_line_out)

            if i == num_samples - 1:
                break
                
    return encoder_input, decoder_input, decoder_target
```


<br>




이렇게 얻은 3개의 데이터셋 **<span style="color:red">인코더의 입력</span>**, **<span style="color:red">디코더의 입력</span>**, **<span style="color:red">디코더의 레이블</span>** 을 상위 5개 샘플만 출력해보겠습니다.
```py
sents_en_in, sents_fra_in, sents_fra_out = load_preprocessed_data()

print('인코더의 입력 :', sents_en_in[:5])
print('디코더의 입력 :', sents_fra_in[:5])
print('디코더의 레이블 :', sents_fra_out[:5])
```
```
[output]
인코더의 입력 : [['go', '.'], ['go', '.'], ['go', '.'], ['go', '.'], ['hi', '.']]
디코더의 입력 : [['<sos>', 'va', '!'], ['<sos>', 'marche', '.'], ['<sos>', 'en', 'route', '!'], ['<sos>', 'bouge', '!'], ['<sos>', 'salut', '!']]
디코더의 레이블 : [['va', '!', '<eos>'], ['marche', '.', '<eos>'], ['en', 'route', '!', '<eos>'], ['bouge', '!', '<eos>'], ['salut', '!', '<eos>']]
```


<br>



모델 설계에 있어서 디코더의 입력에 해당하는 데이터인 **<span style="color:red">sents_fra_in이 필요한 이유</span>** 를 살펴보겠습니다.


테스트 과정에서 현재 시점의 디코더 셀의 입력은 오직 이전 디코더 셀의 출력을 입력으로 받습니다. 하지만 **<span style="color:red">훈련 과정</span>** 에서는 **<u>이전 시점의 디코더 셀의 출력을 현재 시점의 디코더 셀의 입력으로 넣어주지 않고</u>**, **<span style="color:red">이전 시점의 실제값을 현재 시점의 디코더 셀의 입력값으로 하는 방법을 사용</span>** 합니다.

<br>

그 **<span style="color:red">이유</span>** 는 **<span style="background-color: #fff5b1">이전 시점의 디코더 셀의 예측이 틀렸는데 이를 현재 시점의 디코더 셀의 입력으로 사용하면 현재 시점의 디코더 셀의 예측도 잘못될 가능성이 높고 이는 연쇄 작용으로 디코더 전체의 예측을 어렵게 합니다.</span>** 이런 상황이 반복되면 훈련 시간이 느려집니다.

만약 이 상황을 원하지 않는다면 이전 시점의 디코더 셀의 예측값 대신 실제값을 현재 시점의 디코더 셀의 입력으로 사용하는 방법을 사용할 수 있습니다. 이와 같이 **<span style="color:red">RNN의 모든 시점에 대해서 이전 시점의 예측값 대신 실제값을 입력으로 주는 방법</span>** 을 **<span style="color:red">교사 강요</span>** 라고 합니다.

<br>




케라스 토크나이저를 통해 **<span style="color:red">단어 집합을 생성</span>**, **<span style="color:red">정수 인코딩</span>** 을 진행 후 이어서 **<span style="color:red">패딩</span>** 을 진행하겠습니다.
```py
tokenizer_en = Tokenizer(filters="", lower=False)
tokenizer_en.fit_on_texts(sents_en_in)
encoder_input = tokenizer_en.texts_to_sequences(sents_en_in)
encoder_input = pad_sequences(encoder_input, padding="post")


tokenizer_fra = Tokenizer(filters="", lower=False)
tokenizer_fra.fit_on_texts(sents_fra_in)
tokenizer_fra.fit_on_texts(sents_fra_out)


decoder_input = tokenizer_fra.texts_to_sequences(sents_fra_in)
decoder_input = pad_sequences(decoder_input, padding="post")

decoder_target = tokenizer_fra.texts_to_sequences(sents_fra_out)
decoder_target = pad_sequences(decoder_target, padding="post")
```
```py
print('인코더의 입력의 크기(shape) :', encoder_input.shape)
print('디코더의 입력의 크기(shape) :', decoder_input.shape)
print('디코더의 레이블의 크기(shape) :', decoder_target.shape)
```
```
[output]
인코더의 입력의 크기(shape) : (33000, 7)
디코더의 입력의 크기(shape) : (33000, 16)
디코더의 레이블의 크기(shape) : (33000, 16)
```


<br>


데이터의 크기(shape)를 확인했을때 샘플은 총 33,000개 존재하며 영어 문장의 길이는 7, 프랑스어 문장의 길이는 16입니다. **<span style="color:red">단어 집합의 크기를 정의</span>** 합니다.
```py
src_vocab_size = len(tokenizer_en.word_index) + 1
tar_vocab_size = len(tokenizer_fra.word_index) + 1
print("영어 단어 집합의 크기 : {:d}, 프랑스어 단어 집합의 크기 : {:d}".format(src_vocab_size, tar_vocab_size))
```
```
[output]
영어 단어 집합의 크기 : 4516, 프랑스어 단어 집합의 크기 : 7907
```


<br>


단어 집합의 크기는 각각 4,516개와 7,907개입니다. **단어로부터 정수를 얻는 딕셔너리** 와 **정수로부터 단어를 얻는 딕셔너리** 를 각각 만들어줍니다. **<u>이들은 훈련을 마치고 예측값과 실제값을 비교하는 단계에서 사용됩니다.</u>**
```py
src_to_index = tokenizer_en.word_index
index_to_src = tokenizer_en.index_word
tar_to_index = tokenizer_fra.word_index
index_to_tar = tokenizer_fra.index_word
```


<br>

테스트 데이터를 분리하기 전 데이터를 섞어줍니다. 이를 위해서 **순서가 섞인 정수 시퀀스 리스트** 를 만듭니다.
```py
indices = np.arange(encoder_input.shape[0]) # 33000
np.random.shuffle(indices)
print('랜덤 시퀀스 :', indices)
```
```
[output]
랜덤 시퀀스 : [24985  5677 24649 ... 19502 14537  9821]
```


<br>



이를 데이터셋의 순서로 지정해주면 샘플들이 기존 순서와 다른 순서로 섞이게 됩니다.
```py
encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]
```


<br>



임의로 30,997번째 샘플을 출력해보겠습니다. 이때 **decoder_input과 decoder_target은 데이터의 구조상으로 앞에 붙은 \<sos\> 토큰과 뒤에 붙은 \<eos\> 을 제외하면 동일한 정수 시퀀스를 가져야 합니다.**
```py
encoder_input[30997]
```
```
[output]
array([ 2, 97,  3,  1,  0,  0,  0], dtype=int32)
```
```py
decoder_input[30997]
```
```
[output]
array([  2,   4,  54, 757,   1,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0], dtype=int32)
```
```py
decoder_target[30997]
```
```
[output]
array([  4,  54, 757,   1,   3,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0], dtype=int32)
```
<br>



4, 54, 757, 173, 1이라는 동일 시퀀스를 확인했습니다. 이제 훈련 데이터의 10%를 테스트 데이터로 분리하겠습니다.
```py
n_of_val = int(33000 * 0.1)
print('검증 데이터의 개수 :', n_of_val)
```
```
[output]
검증 데이터의 개수 : 3300
```


<br>


33,000개의 10%에 해당되는 3,300개의 데이터를 테스트 데이터로 사용합니다.
```py
encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]
```


<br>


훈련 데이터와 테스트 데이터의 크기(shape)를 출력해보겠습니다.
```py
print('훈련 source 데이터의 크기 :', encoder_input_train.shape)
print('훈련 target 데이터의 크기 :', decoder_input_train.shape)
print('훈련 target 레이블의 크기 :', decoder_target_train.shape)

print('\n테스트 source 데이터의 크기 :', encoder_input_test.shape)
print('테스트 target 데이터의 크기 :', decoder_input_test.shape)
print('테스트 target 레이블의 크기 :', decoder_target_test.shape)
```
```
[output]
훈련 source 데이터의 크기 : (29700, 7)
훈련 target 데이터의 크기 : (29700, 16)
훈련 target 레이블의 크기 : (29700, 16)

테스트 source 데이터의 크기 : (3300, 7)
테스트 target 데이터의 크기 : (3300, 16)
테스트 target 레이블의 크기 : (3300, 16)
```


<br>


훈련 데이터의 샘플은 29,700개, 테스트 데이터의 샘플은 3,300개가 존재합니다. 이제 모델을 설계합니다.

<br>




# 2. 기계 번역기 만들기


우선 **임베딩 벡터의 차원** 과 **LSTM의 은닉 상태의 크기** 를 64로 사용합니다.
```py
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model

embedding_dim = 64
hidden_units = 64
```


<br>




**<span style="color:red">인코더</span>** 를 살펴보면, 우선 **Masking** 은 패딩 토큰인 숫자 0의 경우에는 연산을 제외하는 역할을 수행합니다. 인코더의 내부 상태를 디코더로 넘겨주어야 하기 때문에 **LSTM** 의 **return_state=True** 로 설정합니다. 인코더에 입력을 넣으면 내부 상태를 리턴합니다.

**LSTM** 에서 **state_h**, **state_c** 를 리턴받는데, 이는 각각 **은닉 상태** 와 **셀 상태** 에 해당됩니다. 이 두 가지 상태를 **encoder_states** 에 저장합니다. **encoder_states** 를 디코더에 전달하므로서 이 두 가지 상태 모두를 디코더로 전달할 예정입니다. 이것이 **<span style="color:red">컨텍스트 벡터</span>** 입니다.
```py
# 인코더
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(src_vocab_size, embedding_dim)(encoder_inputs) # 임베딩 층
enc_masking = Masking(mask_value=0.0)(enc_emb) # 패딩 0은 연산에서 제외

# 상태값 리턴을 위해 return_state는 True
encoder_lstm = LSTM(hidden_units, return_state=True)

encoder_outputs, state_h, state_c = encoder_lstm(enc_masking) # 은닉 상태와 셀 상태를 리턴

# 인코더의 은닉 상태와 셀 상태를 저장
encoder_states = [state_h, state_c]
```


<br>




**<span style="color:red">디코더</span>** 를 살펴보면, 디코더는 인코더의 마지막 은닉 상태로부터 초기 은닉 상태를 얻습니다. **initial_state** 의 인자값으로 **encoder_states** 를 주는 코드가 이에 해당됩니다. **<u>디코더도 은닉 상태, 셀 상태를 리턴하기는 하지만 훈련 과정에서는 사용하지 않습니다.</u>** seq2seq의 디코더는 기본적으로 각 시점마다 다중 클래스 분류 문제를 풀고있습니다. **매 시점마다 프랑스어 단어 집합의 크기(tar_vocab_size)의 선택지에서 단어를 1개 선택하여 이를 이번 시점에서 예측한 단어로 택합니다.** 다중 클래스 분류 문제이므로 출력층으로 소프트맥스 함수와 손실 함수를 크로스 엔트로피 함수를 사용합니다.

categorical_crossentropy를 사용하려면 레이블은 원-핫 인코딩이 된 상태여야 합니다. 그런데 현재 decoder_outputs의 경우에는 원-핫 인코딩을 하지 않은 상태입니다. 원-핫 인코딩을 하지 않은 상태로 정수 레이블에 대해서 다중 클래스 분류 문제를 풀고자 하는 경우에는 categorical_crossentropy가 아니라 **sparse_categorical_crossentropy** 를 사용하면 됩니다.
```py
# 디코더
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(tar_vocab_size, hidden_units) # 임베딩 층
dec_emb = dec_emb_layer(decoder_inputs)
dec_masking = Masking(mask_value=0.0)(dec_emb) # 패딩 0은 연산에서 제외

# 상태값 리턴을 위해 return_state는 True, 모든 시점에 대해서 단어를 예측하기 위해 return_sequences는 True
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True) 

# 인코더의 은닉 상태를 초기 은닉 상태(initial_state)로 사용
decoder_outputs, _, _ = decoder_lstm(dec_masking, initial_state=encoder_states)

# 모든 시점의 결과에 대해서 소프트맥스 함수를 사용한 출력층을 통해 단어 예측
decoder_dense = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
```
```py
# 모델의 입력과 출력을 정의.
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
```
```
[output]
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 input_2 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 embedding (Embedding)          (None, None, 64)     289024      ['input_1[0][0]']                
                                                                                                  
 embedding_1 (Embedding)        (None, None, 64)     506048      ['input_2[0][0]']                
                                                                                                  
 masking (Masking)              (None, None, 64)     0           ['embedding[0][0]']              
                                                                                                  
 masking_1 (Masking)            (None, None, 64)     0           ['embedding_1[0][0]']            
                                                                                                  
 lstm (LSTM)                    [(None, 64),         33024       ['masking[0][0]']                
                                 (None, 64),                                                      
                                 (None, 64)]                                                      
                                                                                                  
 lstm_1 (LSTM)                  [(None, None, 64),   33024       ['masking_1[0][0]',              
                                 (None, 64),                      'lstm[0][1]',                   
                                 (None, 64)]                      'lstm[0][2]']                   
                                                                                                  
 dense (Dense)                  (None, None, 7907)   513955      ['lstm_1[0][0]']                 
                                                                                                  
==================================================================================================
Total params: 1,375,075
Trainable params: 1,375,075
Non-trainable params: 0
__________________________________________________________________________________________________
```


<br>




모델을 훈련합니다. 128개의 배치 크기로 총 50 에포크 학습합니다. 테스트 데이터를 검증 데이터로 사용하여 훈련이 제대로 되고있는지 모니터링하겠습니다.
```py
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.fit(
    x=[encoder_input_train, decoder_input_train],
    y=decoder_target_train,
    validation_data=(
        [encoder_input_test, decoder_input_test],
        decoder_target_test
    ),
    batch_size=128,
    epochs=50
)
```
```
[output]
Epoch 1/50
233/233 [==============================] - 12s 33ms/step - loss: 3.4443 - acc: 0.6133 - val_loss: 2.1003 - val_acc: 0.6172
Epoch 2/50
233/233 [==============================] - 7s 29ms/step - loss: 1.9005 - acc: 0.6512 - val_loss: 1.7903 - val_acc: 0.7075
Epoch 3/50
233/233 [==============================] - 7s 29ms/step - loss: 1.6843 - acc: 0.7355 - val_loss: 1.6207 - val_acc: 0.7490
Epoch 4/50
233/233 [==============================] - 7s 29ms/step - loss: 1.5351 - acc: 0.7559 - val_loss: 1.4994 - val_acc: 0.7611
.
.
.
Epoch 46/50
233/233 [==============================] - 7s 29ms/step - loss: 0.3713 - acc: 0.9116 - val_loss: 0.7621 - val_acc: 0.8646
Epoch 47/50
233/233 [==============================] - 7s 29ms/step - loss: 0.3635 - acc: 0.9133 - val_loss: 0.7616 - val_acc: 0.8651
Epoch 48/50
233/233 [==============================] - 7s 29ms/step - loss: 0.3559 - acc: 0.9147 - val_loss: 0.7601 - val_acc: 0.8660
Epoch 49/50
233/233 [==============================] - 7s 29ms/step - loss: 0.3485 - acc: 0.9158 - val_loss: 0.7568 - val_acc: 0.8660
Epoch 50/50
233/233 [==============================] - 7s 29ms/step - loss: 0.3403 - acc: 0.9176 - val_loss: 0.7584 - val_acc: 0.8671
```


<br>






# 3. seq2seq 기계 번역기 동작시키기

seq2seq는 훈련 과정(교사 강요)과 테스트 과정에서의 동작 방식이 다릅니다. 그래서 테스트 과정을 위해 모델을 다시 설계해주어야하며, **특히 디코더를 수정** 해야 합니다. 이번에는 번역 단계를 위해 모델을 수정하고 동작시켜보겠습니다.

전체적인 번역 단계를 정리하면 아래와 같습니다.

1. 번역하고자 하는 입력 문장이 인코더로 입력되어 인코더의 마지막 시점의 은닉 상태와 셀 상태를 얻습니다.
2. 인코더의 은닉 상태와 셀 상태, 그리고 토큰 **\<sos\>** 를 디코더로 보냅니다.
3. 디코더가 토큰 **\<eos\>** 가 나올 때까지 다음 단어를 예측하는 행동을 반복합니다.

<br>

인코더의 입, 출력으로 사용하는 **encoder_inputs** 와 **encoder_states** 는 훈련 과정에서 이미 정의한 것들을 재사용합니다. 이렇게 되면 훈련 단계에 **encoder_inputs** 와 **encoder_states** 사이에 있는 모든 층까지 전부 불러오게 되므로 결과적으로 훈련 단계에서 사용한 인코더를 그대로 재사용하게 됩니다.
```py
# 인코더
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()
```
```
[output]
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, None)]            0         
                                                                 
 embedding (Embedding)       (None, None, 64)          289024    
                                                                 
 masking (Masking)           (None, None, 64)          0         
                                                                 
 lstm (LSTM)                 [(None, 64),              33024     
                              (None, 64),                        
                              (None, 64)]                        
                                                                 
=================================================================
Total params: 322,048
Trainable params: 322,048
Non-trainable params: 0
_________________________________________________________________
```


<br>

이어서 디코더를 설계합니다. 테스트 단계에서는 디코더를 매 시점 별로 컨트롤 할 예정으로, 이를 위해서 이전 시점의 상태를 저장할 텐서인 **decoder_state_input_h**, **decoder_state_input_c** 를 정의합니다. 매 시점 별로 디코더를 컨트롤하는 함수는 뒤에서 정의할 **decode_sequence()** 로 해당 함수를 자세히 살펴봐야 합니다.
```py
# 디코더 설계 시작
# 이전 시점의 상태를 보관할 텐서
decoder_state_input_h = Input(shape=(hidden_units,))
decoder_state_input_c = Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 훈련 때 사용했던 임베딩 층을 재사용
dec_emb2 = dec_emb_layer(decoder_inputs)

# 다음 단어 예측을 위해 이전 시점의 상태를 현 시점의 초기 상태로 사용
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

# 모든 시점에 대해서 단어 예측
decoder_outputs2 = decoder_dense(decoder_outputs2)

# 수정된 디코더
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

decoder_model.summary()
```
```
[output]
Model: "model_2"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_2 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 embedding_1 (Embedding)        (None, None, 64)     506048      ['input_2[0][0]']                
                                                                                                  
 input_3 (InputLayer)           [(None, 64)]         0           []                               
                                                                                                  
 input_4 (InputLayer)           [(None, 64)]         0           []                               
                                                                                                  
 lstm_1 (LSTM)                  [(None, None, 64),   33024       ['embedding_1[1][0]',            
                                 (None, 64),                      'input_3[0][0]',                
                                 (None, 64)]                      'input_4[0][0]']                
                                                                                                  
 dense (Dense)                  (None, None, 7907)   513955      ['lstm_1[1][0]']                 
                                                                                                  
==================================================================================================
Total params: 1,053,027
Trainable params: 1,053,027
Non-trainable params: 0
__________________________________________________________________________________________________
```

<br>


테스트 단계에서의 동작을 위한 **decode_sequence** 함수를 구현합니다. 입력 문장이 들어오면 인코더는 마지막 시점까지 전개하여 마지막 시점의 은닉 상태와 셀 상태를 리턴합니다. 이 두 개의 값을 **states_value** 에 저장합니다. 그리고 디코더의 초기 입력으로 **\<sos\>** 를 준비합니다. 이를 **target_seq** 에 저장합니다. 이 두 가지 입력을 가지고 while문 안으로 진입하여 이 두 가지를 디코더의 입력으로 사용합니다.

이제 디코더는 현재 시점에 대해서 예측을 하게 되는데, 현재 시점의 예측 벡터가 output_tokens, 현재 시점의 은닉 상태가 h, 현재 시점의 셀 상태가 c입니다. 예측 벡터로부터 현재 시점의 예측 단어인 **target_seq** 를 얻고, h와 c 이 두 개의 값은 **states_value** 에 저장합니다. 그리고 while문의 다음 루프. 즉, 두번째 시점의 디코더의 입력으로 다시 **target_seq** 와 **states_value** 를 사용합니다. 이를 현재 시점의 예측 단어로 **\<eos\>** 를 예측하거나 번역 문장의 길이가 50이 넘는 순간까지 반복합니다. 각 시점마다 번역된 단어는 **decoded_sentence** 에 누적하여 저장하였다가 최종 번역 시퀀스로 리턴합니다.
```py
def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    states_value = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 정수 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tar_to_index['<sos>']

    stop_condition = False
    decoded_sentence = ''

    # stop_condition이 True가 될 때까지 루프 반복
    # 구현의 간소화를 위해서 이 함수는 배치 크기를 1로 가정합니다.
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 단어로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # 현재 시점의 예측 단어를 예측 문장에 추가
        decoded_sentence += ' '+sampled_char

        # <eos>에 도달하거나 정해진 길이를 넘으면 중단.
        if (sampled_char == '<eos>' or
            len(decoded_sentence) > 50):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence
```


<br>



결과 확인을 위한 함수를 만듭니다. **seq_to_src** 함수는 영어 문장에 해당하는 정수 시퀀스를 입력받으면 정수로부터 영어 단어를 리턴하는 **index_to_src** 를 통해 영어 문장으로 변환합니다. **seq_to_tar** 은 프랑스어에 해당하는 정수 시퀀스를 입력받으면 정수로부터 프랑스어 단어를 리턴하는 **index_to_tar** 을 통해 프랑스어 문장으로 변환합니다.
```py
# 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq_to_src(input_seq):
    sentence = ''
    for encoded_word in input_seq:
        if(encoded_word!=0):
            sentence = sentence + index_to_src[encoded_word] + ' '
    return sentence

# 번역문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq_to_tar(input_seq):
    sentence = ''
    for encoded_word in input_seq:
        if(encoded_word!=0 and encoded_word!=tar_to_index['<sos>'] and encoded_word!=tar_to_index['<eos>']):
            sentence = sentence + index_to_tar[encoded_word] + ' '
    return sentence
```


<br>


훈련 데이터에 대해서 임의로 선택한 인덱스의 샘플의 결과를 출력해봅시다.
```py
for seq_index in [3, 50, 100, 300, 1001]:
    input_seq = encoder_input_train[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    
    print("입력문장 :", seq_to_src(encoder_input_train[seq_index]))
    print("정답문장 :", seq_to_tar(decoder_input_train[seq_index]))
    print("번역문장 :", decoded_sentence[1:-5])
    print("-"*50)
```
```
[output]
입력문장 : i retired in . 
정답문장 : j ai pris ma retraite en . 
번역문장 : je suis tombe a la retraite . 
--------------------------------------------------
입력문장 : i found you . 
정답문장 : je t ai trouve . 
번역문장 : je vous ai trouve . 
--------------------------------------------------
입력문장 : i have many discs . 
정답문장 : j ai beaucoup de disques . 
번역문장 : j ai beaucoup de disques . 
--------------------------------------------------
입력문장 : i m shivering . 
정답문장 : je tremble . 
번역문장 : je tremble . 
--------------------------------------------------
입력문장 : i often hiccup . 
정답문장 : j ai souvent le hoquet . 
번역문장 : je fais que j ai gagne . 
--------------------------------------------------
```


<br>



테스트 데이터에 대해서 임의로 선택한 인덱스의 샘플의 결과를 출력해봅시다.
```py
for seq_index in [3, 50, 100, 300, 1001]:
    input_seq = encoder_input_test[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    
    print("입력문장 :", seq_to_src(encoder_input_test[seq_index]))
    print("정답문장 :", seq_to_tar(decoder_input_test[seq_index]))
    print("번역문장 :", decoded_sentence[1:-5])
    print("-"*50)
```
```
[output]
입력문장 : tom is childish . 
정답문장 : tom est immature . 
번역문장 : tom est en train de danser . 
--------------------------------------------------
입력문장 : i have needs . 
정답문장 : j ai des besoins . 
번역문장 : il me faut que j en arriere . 
--------------------------------------------------
입력문장 : do as you want . 
정답문장 : fais comme ca te chante . 
번역문장 : fais comme tu veux . 
--------------------------------------------------
입력문장 : brace yourselves . 
정답문장 : accrochez vous . 
번역문장 : preparez vous . 
--------------------------------------------------
입력문장 : tom hates french . 
정답문장 : tom deteste le francais . 
번역문장 : tom deteste le francais . 
--------------------------------------------------
```




