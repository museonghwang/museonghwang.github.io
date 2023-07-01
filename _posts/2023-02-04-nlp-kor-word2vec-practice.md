---
layout: post
title: 한국어 Word2Vec 실습
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---




1. gensim 패키지 버전확인
2. 한국어 Word2Vec 만들기
    - 2.1 Libaray Import
    - 2.2 훈련 데이터 이해
    - 2.3 훈련 데이터 전처리
    - 2.4 Word2Vec 훈련
    - 2.5 Word2Vec 모델 저장 및 로드
3. 한국어 Word2Vec 임베딩 벡터의 시각화(Embedding Visualization)
    - 3.1 워드 임베딩 모델로부터 2개의 tsv 파일 생성하기
    - 3.2 임베딩 프로젝터를 사용하여 시각화하기



<br>

**gensim** 패키지에서 제공하는 이미 구현된 **Word2Vec** 을 사용하여 한국어 데이터를 학습하겠습니다.


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





# 2. 한국어 Word2Vec 만들기

한국어로 된 코퍼스를 다운받아 전처리를 수행하고, 전처리한 데이터를 바탕으로 **Word2Vec** 작업을 진행하겠습니다.

<br>



## 2.1 Libaray Import

우선 **KoNLPy** 의 **OKT** 등은 형태소 분석 속도가 너무 느립니다. 그래서 **Mecab** 을 설치하겠습니다. 단, **Mecab** 은 형태소 분석 속도는 빠르지만 설치하는데 시간이 좀 걸립니다.
```
!pip install konlpy
!pip install mecab-python
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

<br>


필요 라이브러리를 불러옵니다.
```py
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tqdm import tqdm
from konlpy.tag import Mecab
from gensim.models.word2vec import Word2Vec
```


<br>




## 2.2 훈련 데이터 다운로드

**Word2Vec** 을 학습하기 위해서 데이터를 다운로드합니다. 사용할 훈련 데이터는, 네이버 영화 리뷰 데이터입니다.
```py
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
```
```
[output]
('ratings.txt', <http.client.HTTPMessage at 0x7f17bdc04700>)
```


<br>




## 2.3 훈련 데이터 전처리

데이터를 읽고 상위 5개를 출력하겠습니다.
```py
train_data = pd.read_table('ratings.txt')
train_data[:5] # 상위 5개 출력
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/c94ca887-2a7b-4eb4-869d-76ba1180f687">
</p>

<br>



이후 리뷰 개수를 확인해보고 **Null** 값이 있으면 제거하겠습니다.
```py
print('리뷰 개수 :', len(train_data)) # 리뷰 개수 출력

# NULL 값 존재 유무
print('NULL 값 존재 유무 :', train_data.isnull().values.any())

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print('NULL 값 존재 유무 :', train_data.isnull().values.any()) # Null 값이 존재하는지 확인

print('리뷰 개수 :', len(train_data)) # 리뷰 개수 출력
```
```
[output]
리뷰 개수 : 200000
NULL 값 존재 유무 : True
NULL 값 존재 유무 : False
리뷰 개수 : 199992
```


<br>



또한 정규표현식으로 한글 외 문자들읊 제거하겠습니다.
```py
# 정규 표현식을 통한 한글 외 문자 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data[:5] # 상위 5개 출력
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/f65969b8-7195-4270-89bf-35b092e1547f">
</p>

<br>



이제 **mecab** 을 이용하여 토큰화 작업을 수행하겠습니다.
```py
# 불용어 정의
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']

# 형태소 분석기 mecab을 사용한 토큰화 작업 (다소 시간 소요)
mecab = Mecab()

tokenized_data = []
for sentence in tqdm(train_data['document']):
    temp_X = mecab.morphs(sentence) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_X)
```
```
[output]
100%|██████████| 199992/199992 [00:23<00:00, 8652.42it/s]
```


<br>


```py
print(tokenized_data[:3])
```
```
[output]
[['어릴', '때', '보', '지금', '다시', '봐도', '재밌', '어요', 'ㅋㅋ'],
 ['디자인', '배우', '학생', '으로', '외국', '디자이너', '그', '일군', '전통', '통해', '발전', '해', '문화', '산업', '부러웠', '는데', '사실', '우리', '나라', '에서', '그', '어려운', '시절', '끝', '까지', '열정', '지킨', '노라노', '같', '전통', '있', '어', '저', '같', '사람', '꿈', '꾸', '이뤄나갈', '수', '있', '다는', '것', '감사', '합니다'], 
 ['폴리스', '스토리', '시리즈', '부터', '뉴', '까지', '버릴', '께', '하나', '없', '음', '최고']]
```


<br>



상위 3개 문장만 출력해보았는데 토큰화가 잘 수행되었음을 볼 수 있습니다. 리뷰 길이 분포도 한번 확인해보겠습니다.
```py
# 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```
```
[output]
리뷰의 최대 길이 : 74
리뷰의 평균 길이 : 11.996394855794232
```

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/4b268b68-0e27-4f75-b155-1aa9670639e8">
</p>

<br>



이제 **Word2Vec** 모델에 텍스트 데이터를 훈련시킵니다.

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
    sentences=tokenized_data,
    vector_size=100,
    window=5,
    min_count=5,
    workers =4,
    sg=0)
```


<br>



**Word2Vec** 에 대해서 학습을 진행하였습니다. 완성된 임베딩 매트릭스의 크기를 확인해보겠습니다.
```py
# 완성된 임베딩 매트릭스의 크기 확인
print('완성된 임베딩 매트릭스의 크기 확인 :', model.wv.vectors.shape)
```
```
[output]
완성된 임베딩 매트릭스의 크기 확인 : (18134, 100)
```


<br>


**Word2Vec** 는 입력한 단어에 대해서 가장 유사한 단어들을 출력하는 **model.wv.most_similar**을 지원합니다. 특정 단어와 가장 유사한 단어들을 추출해보겠습니다. 이때 코사인 유사도라는 것을 유사도 메트릭으로 사용하며, 값의 범위는 **-1 ~ 1** 입니다.
```py
print(model.wv.most_similar("최민식"))
```
```
[output]
[('한석규', 0.8795050978660583), ('드니로', 0.8788982629776001), ('안성기', 0.8662292957305908), ('박중훈', 0.8494662046432495), ('조재현', 0.844645082950592), ('송강호', 0.8443252444267273), ('주진모', 0.8288795351982117), ('채민서', 0.826660692691803), ('설경구', 0.8246625661849976), ('혼신', 0.8212024569511414)]
```


<br>

```py
print(model.wv.most_similar("히어로"))
```
```
[output]
[('호러', 0.8805227279663086), ('슬래셔', 0.8623908758163452), ('무비', 0.8325807452201843), ('고어', 0.8201742768287659), ('느와르', 0.8061329126358032), ('무협', 0.7969907522201538), ('정통', 0.7941304445266724), ('블록버스터', 0.7905427813529968), ('괴수', 0.7877203822135925), ('로코', 0.7815171480178833)]
```


<br>

```py
print(model.wv.most_similar("발연기"))
```
```
[output]
[('조연', 0.7612935304641724), ('사투리', 0.7608969807624817), ('발음', 0.7429847121238708), ('아역', 0.742458701133728), ('연기력', 0.7332570552825928), ('신하균', 0.7288188338279724), ('주연', 0.7250414490699768), ('연기', 0.7174587845802307), ('김민준', 0.7035930752754211), ('연기파', 0.6986960172653198)]

```


<br>


**<span style="color:red">Word2Vec를 통해 단어의 유사도를 계산할 수 있게 되었습니다.</span>**

각 단어의 임베딩 값을 확인할 수 있습니다.
```py
model.wv['최민식']
```
```
[output]
array([ 0.12769549,  0.07311668, -0.09119831, -0.23128814, -0.20396759,
       -0.38492376,  0.15919358,  0.18515116, -0.05406173,  0.02143147,
        0.0847561 , -0.35062462, -0.25053436,  0.063058  ,  0.10669395,
        0.10934699, -0.02529504, -0.01488628,  0.21295786, -0.3048591 ,
        0.10559981, -0.13670704,  0.1679281 ,  0.38849354, -0.10775245,
        0.26664627,  0.18594052, -0.25520116, -0.08447041,  0.16291223,
        0.2878455 , -0.01453895,  0.44054905,  0.0901166 ,  0.06951199,
        0.36122474,  0.27994072,  0.07385039, -0.40010566, -0.52587914,
       -0.02900855, -0.4807919 ,  0.2886666 ,  0.06037587,  0.39145902,
       -0.32478467, -0.19919494, -0.2002729 ,  0.18941337, -0.06913789,
        0.06830232,  0.15130404, -0.05708817,  0.01178154, -0.15022099,
       -0.2102085 , -0.06560393,  0.08470158,  0.23514558, -0.12844469,
        0.24766013,  0.3250303 , -0.4092613 ,  0.13039684, -0.31366074,
        0.18714847,  0.06172501,  0.15533729, -0.4577842 ,  0.4986381 ,
       -0.46604767, -0.1577858 ,  0.31474996,  0.03983723,  0.12968569,
        0.41637075,  0.2854629 , -0.07649355,  0.01544307, -0.0667455 ,
       -0.10259806, -0.25724038,  0.07584251,  0.2599289 , -0.16078846,
       -0.26536906,  0.02234764,  0.23640175,  0.11873386,  0.00365566,
        0.00673907,  0.06015598, -0.05594924,  0.23348501,  0.5042513 ,
        0.05479302, -0.06083839,  0.18337795, -0.14395966,  0.03753674],
      dtype=float32)
```


<br>





## 2.5 Word2Vec 모델 저장 및 로드

학습한 모델을 언제든 나중에 다시 사용할 수 있도록 컴퓨터 파일로 저장하고 다시 로드해보겠습니다.
```py
from gensim.models import KeyedVectors

model.wv.save_word2vec_format('kor_w2v') # 모델 저장

# loaded_model = KeyedVectors.load_word2vec_format("kor_w2v") # 모델 로드
```



<br>




# 3. 한국어 Word2Vec 임베딩 벡터의 시각화(Embedding Visualization)

구글은 **임베딩 프로젝터(embedding projector)** 라는 데이터 시각화 도구를 지원합니다. 학습한 임베딩 벡터들을 시각화해보겠습니다.

<br>




# 3.1 워드 임베딩 모델로부터 2개의 tsv 파일 생성하기

학습한 임베딩 벡터들을 시각화해보겠습니다. 시각화를 위해서는 이미 모델을 학습하고, 파일로 저장되어져 있어야 합니다. 모델이 저장되어져 있다면 아래 커맨드를 통해 시각화에 필요한 파일들을 생성할 수 있습니다.
```py
!python -m gensim.scripts.word2vec2tensor --input 모델이름 --output 모델이름
```

<br>


여기서는 위에서 실습한 한국어 **Word2Vec** 모델인 **'kor_w2v'** 를 사용하겠습니다. **kor_w2v** 라는 **Word2Vec** 모델이 이미 존재한다는 가정 하에 아래 커맨드를 수행합니다.
```py
!python -m gensim.scripts.word2vec2tensor --input kor_w2v --output kor_w2v
```


<br>

위 명령를 수행하면 기존에 있던 **kor_w2v** 외에도 두 개의 파일이 생깁니다. 새로 생긴 **kor_w2v_metadata.tsv** 와 **kor_w2v_tensor.tsv** 이 두 개 파일이 임베딩 벡터 시각화를 위해 사용할 파일입니다. 만약 **kor_w2v** 모델 파일이 아니라 다른 모델 파일 이름으로 실습을 진행하고 있다면, **'모델 이름_metadata.tsv'** 와 **'모델 이름_tensor.tsv'** 라는 파일이 생성됩니다.

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


위에 있는 Choose file 버튼을 누르고 **kor_w2v_tensor.tsv** 파일을 업로드하고, 아래에 있는 Choose file 버튼을 누르고 **kor_w2v_metadata.tsv** 파일을 업로드합니다. 두 파일을 업로드하면 임베딩 프로젝터에 학습했던 워드 임베딩 모델이 시각화됩니다.

<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/3504ea8e-edd1-4f92-9ab5-4a602a8fb6db">
</p>





