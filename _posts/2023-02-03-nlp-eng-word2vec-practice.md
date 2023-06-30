---
layout: post
title: 영어 및 한국어 Word2Vec 실습
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---




1. gensim 패키지 버전확인
2. 영어 Word2Vec 만들기
- 2.1 Libaray Import
- 2.2 훈련 데이터 이해
- 2.3 훈련 데이터 전처리
- 2.4 Word2Vec 훈련
- 2.5 Word2Vec 모델 저장 및 로드
3. 영어 Word2Vec 임베딩 벡터의 시각화(Embedding Visualization)
3.1 워드 임베딩 모델로부터 2개의 tsv 파일 생성하기
3.2 임베딩 프로젝터를 사용하여 시각화하기
4. 사전 훈련된 Word2Vec 임베딩(Pre-trained Word2Vec embedding) 소개



<br>

gensim 패키지에서 제공하는 이미 구현된 Word2Vec을 사용하여 영어 데이터를 학습하겠습니다.


<br>
<br>




# 1. gensim 패키지 버전확인

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

만약 **gensim** 패키지가 없다면 설치해줍니다.
```
pip install gensim
```

<br>





# 2. 영어 Word2Vec 만들기

영어로 된 코퍼스를 다운받아 전처리를 수행하고, 전처리한 데이터를 바탕으로 **Word2Vec** 작업을 진행하겠습니다.

<br>



## 2.1 Libaray Import

우선 필요 라이브러리를 불러옵니다.
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




## 2.2 훈련 데이터 이해

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




## 2.3 훈련 데이터 전처리
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






## 2.4 Word2Vec 훈련

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
model_result = model.wv.most_similar("man")
print(model_result)
```
```
[output]
[('woman', 0.8425011038780212), ('guy', 0.8098958134651184), ('lady', 0.781208872795105), ('boy', 0.7525790929794312), ('girl', 0.7469564080238342), ('gentleman', 0.7207304835319519), ('soldier', 0.7077170014381409), ('kid', 0.6885204315185547), ('david', 0.6624912023544312), ('friend', 0.6530545353889465)]
```


<br>


man과 유사한 단어로 woman, guy, boy, lady, girl, gentleman, soldier, kid 등을 출력하는 것을 볼 수 있습니다. **<span style="color:red">Word2Vec를 통해 단어의 유사도를 계산할 수 있게 되었습니다.</span>**

각 단어의 임베딩 값을 확인할 수 있습니다.
```py
model.wv["man"]
```
```
[output]
array([ 0.54919225, -2.5828376 , -0.05627956, -0.72883856,  0.5080902 ,
       -0.588747  ,  1.1527569 ,  0.6861413 ,  0.20908435,  0.5790621 ,
       -0.76411045, -1.2080296 , -0.9166982 ,  0.6161433 , -0.32686922,
        0.3346195 ,  0.47164342, -0.30977565,  0.360217  , -0.6516018 ,
       -0.06280681,  0.9388923 ,  0.6213905 , -0.4060864 ,  0.8803398 ,
        0.4036564 , -1.8721576 , -0.5711301 ,  0.92875475, -1.4228262 ,
        0.76451683, -0.4689635 ,  1.478043  , -0.3736253 ,  0.24919653,
       -1.2209562 , -2.0871649 , -0.64423513, -1.8315326 , -1.0469043 ,
        1.3488007 , -2.40771   , -0.8882299 ,  1.0518845 ,  0.3505911 ,
       -0.5359099 , -0.11452804, -1.7889714 , -0.50420225,  0.13257498,
        0.46635804, -1.5578051 , -0.40210238,  0.41704193, -0.16498177,
       -1.7667351 , -0.42030132, -0.89286804, -1.9498727 ,  0.31205317,
        0.1363872 , -0.32287887,  0.83032966,  1.0676957 , -2.0881174 ,
        0.5390724 , -1.3501817 , -0.15355064, -0.42196646,  0.5385719 ,
        0.7717964 ,  0.42193443,  2.9974504 , -1.1239656 , -0.8758551 ,
       -1.6787865 , -0.30246603,  0.6885682 , -0.5081502 ,  1.6597394 ,
        0.7549413 ,  0.6027066 , -1.0214967 , -0.23701903, -0.37534398,
        1.4510185 ,  0.13622098,  0.79067725, -0.89343023, -0.14235029,
        0.707251  ,  0.40881404,  0.00797209, -0.5443254 ,  2.3957598 ,
       -0.40322962,  0.37388444,  0.57005996, -2.089544  ,  2.3334754 ],
      dtype=float32)
```


<br>





## 2.5 Word2Vec 모델 저장 및 로드

학습한 모델을 언제든 나중에 다시 사용할 수 있도록 컴퓨터 파일로 저장하고 다시 로드해보겠습니다.
```py
from gensim.models import KeyedVectors

model.wv.save_word2vec_format('eng_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드
```



<br>


**eng_w2v** 라는 모델 파일이 저장되었으며, 로드한 모델에 대해서 다시 man과 유사한 단어를 출력해보겠습니다.
```py
model_result = loaded_model.most_similar("man")
print(model_result)
```
```
[output]
[('woman', 0.845443844795227), ('guy', 0.8228113651275635), ('lady', 0.7647513151168823), ('boy', 0.7602432370185852), ('gentleman', 0.7566214799880981), ('girl', 0.7488241195678711), ('soldier', 0.7224786877632141), ('kid', 0.7164652347564697), ('king', 0.6694438457489014), ('poet', 0.6539930105209351)]
```


<br>





# 3. 영어 Word2Vec 임베딩 벡터의 시각화(Embedding Visualization)

구글은 **임베딩 프로젝터(embedding projector)** 라는 데이터 시각화 도구를 지원합니다. 학습한 임베딩 벡터들을 시각화해보겠습니다.

<br>




# 3.1 워드 임베딩 모델로부터 2개의 tsv 파일 생성하기

학습한 임베딩 벡터들을 시각화해보겠습니다. 시각화를 위해서는 이미 모델을 학습하고, 파일로 저장되어져 있어야 합니다. 모델이 저장되어져 있다면 아래 커맨드를 통해 시각화에 필요한 파일들을 생성할 수 있습니다.
```py
!python -m gensim.scripts.word2vec2tensor --input 모델이름 --output 모델이름
```

<br>


여기서는 위에서 실습한 영어 **Word2Vec** 모델인 **'eng_w2v'** 를 사용하겠습니다. **eng_w2v** 라는 **Word2Vec** 모델이 이미 존재한다는 가정 하에 아래 커맨드를 수행합니다.
```py
!python -m gensim.scripts.word2vec2tensor --input eng_w2v --output eng_w2v
```


<br>

위 명령를 수행하면 기존에 있던 **eng_w2v** 외에도 두 개의 파일이 생깁니다. 새로 생긴 **eng_w2v_metadata.tsv** 와 **eng_w2v_tensor.tsv** 이 두 개 파일이 임베딩 벡터 시각화를 위해 사용할 파일입니다. 만약 **eng_w2v** 모델 파일이 아니라 다른 모델 파일 이름으로 실습을 진행하고 있다면, **'모델 이름_metadata.tsv'** 와 **'모델 이름_tensor.tsv'** 라는 파일이 생성됩니다.

<br>





# 3.2 임베딩 프로젝터를 사용하여 시각화하기

구글의 임베딩 프로젝터를 사용해서 워드 임베딩 모델을 시각화해보겠습니다. 아래의 링크에 접속합니다.

- [구글 임베딩 프로젝터](https://projector.tensorflow.org/)

<br>

사이트에 접속해서 좌측 상단을 보면 Load라는 버튼이 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/fd89edc3-a64b-4e69-b92f-023b0b57a578">
</p>

<br>


Load라는 버튼을 누르면 아래와 같은 창이 뜨는데 총 두 개의 Choose file 버튼이 있습니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/e05b2721-7990-4a54-9006-2d79c582bf6e">
</p>

<br>


위에 있는 Choose file 버튼을 누르고 **eng_w2v_tensor.tsv** 파일을 업로드하고, 아래에 있는 Choose file 버튼을 누르고 **eng_w2v_metadata.tsv** 파일을 업로드합니다. 두 파일을 업로드하면 임베딩 프로젝터에 학습했던 워드 임베딩 모델이 시각화됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/4f6120a9-0547-4fe5-b840-495a4ac8dbdc">
</p>

<br>


그 후에는 임베딩 프로젝터의 다양한 기능을 사용할 수 있습니다. 예를 들어 임베딩 프로젝터는 복잡한 데이터를 차원을 축소하여 시각화 할 수 있도록 도와주는 PCA, t-SNE 등을 제공합니다. 위의 그림은 'man' 이라는 단어를 선택하고, 코사인 유사도를 기준으로 가장 유사한 상위 10개 벡터들을 표시해봤습니다.




<br>





# 4. 사전 훈련된 Word2Vec 임베딩(Pre-trained Word2Vec embedding) 소개

자연어 처리 작업을 할때, 케라스의 **Embedding()** 를 사용하여 갖고 있는 훈련 데이터로부터 처음부터 임베딩 벡터를 훈련시키기도 하지만, 위키피디아 등의 방대한 데이터로 사전에 훈련된 워드 임베딩(pre-trained word embedding vector)를 가지고 와서 해당 벡터들의 값을 원하는 작업에 사용 할 수도 있습니다.

**예를 들어서 감성 분류 작업을 하는데 훈련 데이터의 양이 부족한 상황이라면, 다른 방대한 데이터를 Word2Vec이나 GloVe 등으로 사전에 학습시켜놓은 임베딩 벡터들을 가지고 와서 모델의 입력으로 사용하는 것이 때로는 더 좋은 성능을 얻을 수 있습니다.** 사전 훈련된 워드 임베딩을 가져와서 간단히 단어들의 유사도를 구해보는 실습을 해보겠습니다.

구글이 제공하는 사전 훈련된(미리 학습되어져 있는) Word2Vec 모델을 사용하는 방법에 대해서 알아보겠습니다. 구글은 사전 훈련된 3백만 개의 Word2Vec 단어 벡터들을 제공합니다. 각 임베딩 벡터의 차원은 300입니다. **gensim** 을 통해서 이 모델을 불러오는 건 매우 간단합니다. 이 모델을 다운로드하고 파일 경로를 기재하면 됩니다.

- [모델 다운로드 경로](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

<br>


압축 파일의 용량은 약 1.5GB이지만, 파일의 압축을 풀면 약 3.3GB의 파일이 나옵니다.

```py
import gensim
import urllib.request

# 구글의 사전 훈련된 Word2Vec 모델 다운로드.
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Av37IVBQAAntSe1X3MOAl5gvowQzd2_j' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Av37IVBQAAntSe1X3MOAl5gvowQzd2_j" -O GoogleNews-vectors-negative300.bin.gz && rm -rf /tmp/cookies.txt

# 구글의 사전 훈련된 Word2vec 모델을 로드합니다.
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True) 
```


<br>

모델의 크기(shape)를 확인해보겠습니다.
```py
print(word2vec_model.vectors.shape)
```
```
[output]
(3000000, 300)
```


<br>


모델의 크기는 $3,000,000 × 300$ 입니다. 즉, 3백만 개의 단어와 각 단어의 차원은 300입니다. 파일의 크기가 3기가가 넘는 이유를 계산해보면 아래와 같습니다.

- 3 million words * 300 features * 4bytes/feature = ~3.35GB

<br>



사전 훈련된 임베딩을 사용하여 두 단어의 유사도를 계산해보겠습니다.
```py
print(word2vec_model.similarity('this', 'is'))
print(word2vec_model.similarity('post', 'book'))
```
```
[output]
0.40797037
0.057204384
```


<br>


단어 'book'의 벡터를 출력해보겠습니다.
```py
print(word2vec_model['book'])
```
```
[output]
[ 0.11279297 -0.02612305 -0.04492188  0.06982422  0.140625    0.03039551
 -0.04370117  0.24511719  0.08740234 -0.05053711  0.23144531 -0.07470703
  0.21875     0.03466797 -0.14550781  0.05761719  0.00671387 -0.00701904
  0.13183594 -0.25390625  0.14355469 -0.140625   -0.03564453 -0.21289062
 -0.24804688  0.04980469 -0.09082031  0.14453125  0.05712891 -0.10400391
 -0.19628906 -0.20507812 -0.27539062  0.03063965  0.20117188  0.17382812
  0.09130859 -0.10107422  0.22851562 -0.04077148  0.02709961 -0.00106049
  0.02709961  0.34179688 -0.13183594 -0.078125    0.02197266 -0.18847656
 -0.17480469 -0.05566406 -0.20898438  0.04858398 -0.07617188 -0.15625
 -0.05419922  0.01672363 -0.02722168 -0.11132812 -0.03588867 -0.18359375
  0.28710938  0.01757812  0.02185059 -0.05664062 -0.01251221  0.01708984
 -0.21777344 -0.06787109  0.04711914 -0.00668335  0.08544922 -0.02209473
  0.31835938  0.01794434 -0.02246094 -0.03051758 -0.09570312  0.24414062
  0.20507812  0.05419922  0.29101562  0.03637695  0.04956055 -0.06689453
  0.09277344 -0.10595703 -0.04370117  0.19726562 -0.03015137  0.05615234
  0.08544922 -0.09863281 -0.02392578 -0.08691406 -0.22460938 -0.16894531
  0.09521484 -0.0612793  -0.03015137 -0.265625   -0.13378906  0.00139618
  0.01794434  0.10107422  0.13964844  0.06445312 -0.09765625 -0.11376953
 -0.24511719 -0.15722656  0.00457764  0.12988281 -0.03540039 -0.08105469
  0.18652344  0.03125    -0.09326172 -0.04760742  0.23730469  0.11083984
  0.08691406  0.01916504  0.21386719 -0.0065918  -0.08984375 -0.02502441
 -0.09863281 -0.05639648 -0.26757812  0.19335938 -0.08886719 -0.25976562
  0.05957031 -0.10742188  0.09863281  0.1484375   0.04101562  0.00340271
 -0.06591797 -0.02941895  0.20019531 -0.00521851  0.02355957 -0.13671875
 -0.12597656 -0.10791016  0.0067749   0.15917969  0.0145874  -0.15136719
  0.07519531 -0.02905273  0.01843262  0.20800781  0.25195312 -0.11523438
 -0.23535156  0.04101562 -0.11035156  0.02905273  0.22460938 -0.04272461
  0.09667969  0.11865234  0.08007812  0.07958984  0.3125     -0.14941406
 -0.234375    0.06079102  0.06982422 -0.14355469 -0.05834961 -0.36914062
 -0.10595703  0.00738525  0.24023438 -0.10400391 -0.02124023  0.05712891
 -0.11621094 -0.16894531 -0.06396484 -0.12060547  0.08105469 -0.13769531
 -0.08447266  0.12792969 -0.15429688  0.17871094  0.2421875  -0.06884766
  0.03320312  0.04394531 -0.04589844  0.03686523 -0.07421875 -0.01635742
 -0.24121094 -0.08203125 -0.01733398  0.0291748   0.10742188  0.11279297
  0.12890625  0.01416016 -0.28710938  0.16503906 -0.25585938  0.2109375
 -0.19238281  0.22363281  0.04541016  0.00872803  0.11376953  0.375
  0.09765625  0.06201172  0.12109375 -0.24316406  0.203125    0.12158203
  0.08642578  0.01782227  0.17382812  0.01855469  0.03613281 -0.02124023
 -0.02905273 -0.04541016  0.1796875   0.06494141 -0.13378906 -0.09228516
  0.02172852  0.02099609  0.07226562  0.3046875  -0.27539062 -0.30078125
  0.08691406 -0.22949219  0.0546875  -0.34179688 -0.00680542 -0.0291748
 -0.03222656  0.16210938  0.01141357  0.23339844 -0.0859375  -0.06494141
  0.15039062  0.17675781  0.08251953 -0.26757812 -0.11669922  0.01330566
  0.01818848  0.10009766 -0.09570312  0.109375   -0.16992188 -0.23046875
 -0.22070312  0.0625      0.03662109 -0.125       0.05151367 -0.18847656
  0.22949219  0.26367188 -0.09814453  0.06176758  0.11669922  0.23046875
  0.32617188  0.02038574 -0.03735352 -0.12255859  0.296875   -0.25
 -0.08544922 -0.03149414  0.38085938  0.02929688 -0.265625    0.42382812
 -0.1484375   0.14355469 -0.03125     0.00717163 -0.16601562 -0.15820312
  0.03637695 -0.16796875 -0.01483154  0.09667969 -0.05761719 -0.00515747]
```


<br>



참고로, **Word2vec** 모델은 자연어 처리에서 단어를 밀집 벡터로 만들어주는 단어 임베딩 방법론이지만 최근에 들어서는 자연어 처리를 넘어서 추천 시스템에도 사용되고 있는 모델입니다. 적당하게 데이터를 나열해주면 **Word2vec** 은 위치가 근접한 데이터를 유사도가 높은 벡터를 만들어준다는 점에서 착안된 아이디어 **item2vec** 방법론도 있습니다.





