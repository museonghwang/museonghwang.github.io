---
layout: post
title: Text Analytics을 위한 텍스트 전처리 개요
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---




# 텍스트 분석(Text Analytics)


**NLP(National Language Processing)** 와 **텍스트 분석(Text Analytics, TA)** 을 구분하는 것이 큰 의미는 없어 보이지만, 굳이 구분하자면 **<span style="color:red">NLP</span>** 는 **<u>머신이 인간의 언어를 이해하고 해석하는 데 더 중점</u>** 을 두며, 텍스트 마이닝(Text Mining)이라고도 불리는 **<span style="color:red">텍스트 분석</span>** 은 **<u>비정형 텍스트에서 의미 있는 정보를 추출하는 것에 좀 더 중점</u>** 을 두고 있습니다.


**텍스트 분석** 은 머신러닝, 언어 이해, 통계 등을 활용해 모델을 수립하고 정보를 추출해 **<u>비즈니스 인텔리전스(Business Intelligence)나 예측 분석 등의 분석 작업을 주로 수행</u>** 합니다. 주로 다음과 같은 기술 영역에 집중합니다.

- **텍스트 분류(Text Classification)**
    - 문서가 특정 분류 또는 카테고리에 속하는 것을 예측하는 기법을 통칭합니다.
    - 예를 들어 특정 신문 기사 내용이 연예/정치/사회/문화 중 어떤 카테고리에 속하는지 자동으로 분류하거나 스팸 메일 검출 같은 프로그램이 이에 속합니다.
    - 지도학습을 적용합니다.
- **감성 분석(Sentiment Analysis)**
    - 텍스트에서 나타나는 감정/판단/믿음/의견/기분 등의 주관적인 요소를 분석하는 기법을 총칭합니다.
    - 소셜 미디어 감정 분석, 영화나 제품에 대한 긍정 또는 리뷰, 여론조사 의견 분석 등의 다양한 영역에서 활용됩니다.
    - 지도학습 방법뿐만 아니라 비지도학습을 이용해 적용할 수 있습니다.
- **텍스트 요약(Summarization)**
    - 텍스트 내에서 중요한 주제나 중심 사상을 추출하는 기법을 말합니다.
    - 대표적으로 토픽 모델링(Topic Modeling)이 있습니다.
- **텍스트 군집화(Clustering)와 유사도 측정**
    - 비슷한 유형의 문서에 대해 군집화를 수행하는 기법을 말합니다.
    - 텍스트 분류를 비지도학습으로 수행하는 방법의 일환으로 사용될 수 있습니다.
    - 유사도 측정 역시 문서들간의 유사도를 측정해 비슷한 문서끼리 모을 수 있는 방법입니다.

<br>






# 텍스트 분석 이해

텍스트 분석은 비정형 데이터인 텍스트를 분석하는 것입니다. 머신러닝 알고리즘은 숫자형의 피처 기반 데이터만 입력받을 수 있기 때문에 **<u>텍스트를 머신러닝에 적용하기 위해서는 비정형 텍스트 데이터를 어떻게 피처 형태로 추출하고 추출된 피처에 의미 있는 값을 부여하는가 하는 것이 매우 중요한 요소</u>** 이며, 이렇게 텍스트를 변환하는 것을 **<span style="color:red">피처 벡터화(Feature Vectorization)</span>** 또는 **<span style="color:red">피처 추출(Feature Extraction)</span>** 이라고 합니다. **<span style="color:red">텍스트를 벡터값을 가지는 피처로 변환하는 것은 머신러닝 모델을 적용하기 전에 수행해야 할 매우 중요한 요소</span>** 입니다.

<br>






## 텍스트 분석 수행 프로세스

**<span style="color:red">머신러닝 기반의 텍스트 분석 프로세스</span>** 는 다음과 같은 프로세스 순으로 수행합니다.

- 1. **텍스트 사전 준비작업(텍스트 전처리)**
    - 텍스트를 피처로 만들기 전에 미리 클렌징, 대/소문자 변경, 특수문자 삭제등의 클렌징 작업, 단어(Word) 등의 토큰화 작업, 의미 없는 단어(Stop word) 제거 작업, 어근 추출(Stemming/Lemmatization) 등의 텍스트 정규화 작업을 수행하는 것을 통칭합니다.
- 2. **피처 벡터화/추출**
    - 사전 준비 작업으로 가공된 텍스트에서 피처를 추출하고 여기에 벡터 값을 할당합니다.
    - 대표적인 방법은 BOW와 Word2Vec이 있으며, BOW는 대표적으로 Count 기반과 TF-IDF 기반 벡터화가 있습니다.
- 3. **ML 모델 수립 및 학습/예측/평가**
    - 피처 벡터화된 데이터 세트에 ML 모델을 적용해 학습/예측 및 평가를 수행합니다.


<br>


<p align="center">
<img alt="image" src="https://github.com/museonghwang/museonghwang.github.io/assets/77891754/33852a01-df16-4530-a5ef-de1937811a0d">
</p>

<br>





# 텍스트 사전 준비 작업(텍스트 전처리) - 텍스트 정규화

텍스트 자체를 바로 피처로 만들 수는 없습니다. 이를 위해 **<u>사전에 텍스트를 가공하는 준비 작업이 필요</u>** 합니다. 텍스트 정규화는 텍스트를 머신러닝 알고리즘이나 NLP 애플리케이션에 입력 데이터로 사용하기 위해 클렌징, 정제, 토큰화, 어근화 등의 다양한 텍스트 데이터의 사전 작업을 수행하는 것을 의미합니다. **<span style="color:red">텍스트 분석은 이러한 텍스트 정규화 작업이 매우 중요</span>** 하며, 이러한 텍스트 정규화 작업은 크게 다음과 같이 분류할 수 있습니다.

- **클렌징(Cleansing)**
- **토큰화(Tokenization)**
- **필터링/스톱 워드 제거/철자 수정**
- **Stemming**
- **Lemmatization**

<br>


텍스트 정규화의 주요 작업을 **NLTK** 패키지를 이용해 실습해 보겠습니다.


<br>




## 1. 클렌징(Cleansing)

텍스트에서 **<span style="color:red">분석에 오히려 방해가 되는 불필요한 문자, 기호 등을 사전에 제거하는 작업</span>** 입니다. 예를들어 HTML, XML 태그나 특정 기호 등을 사전에 제거합니다.

<br>



## 2. 텍스트 토큰화(Tokenization)

토큰화의 유형은 문서에서 문장을 분리하는 **<span style="color:red">문장 토큰화</span>** 와 문장에서 단어를 토큰으로 분리하는 **<span style="color:red">단어 토큰화</span>** 로 나눌 수 있습니다.

<br>



### 문장 토큰화(sentence tokenization)

**<u>문장 토큰화는 문장의 마침표(.), 개행문자(\n) 등 문장의 마지막을 뜻하는 기호에 따라 분리하는 것이 일반적</u>** 입니다. 또한 정규 표현식에 따른 문장 토큰화도 가능합니다. **NTLK** 에서 일반적으로 많이 쓰이는 **sent_tokenize** 를 이용해 토큰화를 수행해 보겠습니다. 다음은 3개의 문장으로 이루어진 **텍스트 문서를 문장으로 각각 분리하는 예제** 입니다. **nltk.download('punkt')** 는 마침표, 개행 문자등의 데이터 세트를 다운로드합니다.
```py
from nltk import sent_tokenize
import nltk
nltk.download('punkt')

text_sample = 'The Matrix is everywhere its all around us, here even in this room. \
               You can see it out your window or on your television. \
               You feel it when you go to work, or go to church or pay your taxes.'

sentences = sent_tokenize(text=text_sample)
print(type(sentences), len(sentences))
print(sentences)
```
```
[output]
<class 'list'> 3
['The Matrix is everywhere its all around us, here even in this room.', 'You can see it out your window or on your television.', 'You feel it when you go to work, or go to church or pay your taxes.']
```


<br>


**sent_tokenize()** 가 반환하는 것은 각각의 문장으로 구성된 **list** 객체로, 3개의 문장으로 된 문자열을 가지고 있는 것을 알 수 있습니다.

<br>



### 단어 토큰화(Word Tokenization)

**<u>단어 토큰화는 문장을 단어로 토큰화하는 것</u>** 입니다. 기본적으로 공백, 콤마(,), 마침표(.), 개행문자 등으로 단어를 분리하지만, 정규표현식을 이용해 다양한 유형으로 토큰화를 수행할 수 있습니다.

**<span style="color:red">일반적으로 문장 토큰화는 각 문장이 가지는 시맨틱적인 의미가 중요한 요소로 사용될 때 사용</span>** 합니다. **Bag of Word** 와 같이 단어의 순서가 중요하지 않은 경우 문장 토큰화를 사용하지 않고 단어 토큰화만 사용해도 충분합니다. **NTLK** 에서 기본으로 제공하는 **word_tokenize()** 를 이용해 단어로 토큰화해 보겠습니다.
```py
from nltk import word_tokenize

sentence = "The Matrix is everywhere its all around us, here even in this room."
words = word_tokenize(sentence)
print(type(words), len(words))
print(words)
```
```
[output]
<class 'list'> 15
['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.']
```


<br>




이번에는 **sent_tokenize** 와 **word_tokenize** 를 조합해 문서에 대해서 모든 단어를 토큰화해 보겠습니다. 이전 예제에서 선언된 3개의 문장으로 된 **text_sample** 을 문장별로 단어 토큰화를 적용합니다. 이를 위해 문서를 먼저 문장으로 나누고, 개별 문장을 다시 단어로 토큰화하는 **tokenize_text()** 함수를 생성하겠습니다.
```py
from nltk import word_tokenize, sent_tokenize

#여러개의 문장으로 된 입력 데이터를 문장별로 단어 토큰화 만드는 함수 생성
def tokenize_text(text):
    
    # 문장별로 분리 토큰
    sentences = sent_tokenize(text)
    
    # 분리된 문장별 단어 토큰화
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens

#여러 문장들에 대해 문장별 단어 토큰화 수행. 
word_tokens = tokenize_text(text_sample)
print(type(word_tokens), len(word_tokens))
print(word_tokens)
```
```
[output]
<class 'list'> 3
[['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.'], ['You', 'can', 'see', 'it', 'out', 'your', 'window', 'or', 'on', 'your', 'television', '.'], ['You', 'feel', 'it', 'when', 'you', 'go', 'to', 'work', ',', 'or', 'go', 'to', 'church', 'or', 'pay', 'your', 'taxes', '.']]
```


<br>




## 3. 스톱 워드(Stop word) 제거

**<span style="color:red">스톱 워드(Stop word)</span>** 는 **<u>분석에 큰 의미가 없는 단어를 지칭</u>** 합니다. 가령 영어에서 is, the, a, will 등 문장을 구성하는 필수 문법 요소지만 문맥적으로 큰 의미가 없는 단어가 이에 해당하는데, 이 단어의 경우 문법적인 특성으로 인해 특히 빈번하게 텍스트에 나타나므로 이것들을 사전에 제거하지 않으면 그 빈번함으로 인해 오히려 중요한 단어로 인지될 수 있습니다. **<span style="color:red">따라서 이 의미 없는 단어를 제거하는것이 중요한 전처리 작업</span>** 입니다.



**NLTK** 의 경우 다양한 언어의 스톱 워드를 제공합니다. **NTLK** 의 스톱 워드에는 어떤 것이 있는지 확인해 보겠습니다. 이를 위해 먼저 **NLTK** 의 **stopwords** 목록을 내려받고, 다운로드가 완료되고 나면 **NTLK** 의 **English** 의 경우 몇 개의 **stopwords** 가 있는지 알아보고 그중 20개만 확인해 보겠습니다.
```py
import nltk
nltk.download('stopwords')

print('영어 stop words 갯수:', len(nltk.corpus.stopwords.words('english')))
print(nltk.corpus.stopwords.words('english')[:20])
```
```
[output]
영어 stop words 갯수: 179
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his']
```


<br>



영어의 경우 스톱 워드의 개수가 179개이며, 그중 20개만 살펴보면 위의 결과와 같습니다. 위 예제에서 3개의 문장별로 단어를 토큰화해 생성된 **word_tokens** 리스트에 대해서 **stopwords**를 필터링으로 제거해 분석을 위한 의미 있는 단어만 추출해 보겠습니다.
```py
import nltk

stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []

# 위 예제의 3개의 문장별로 얻은 word_tokens list 에 대해 stop word 제거 Loop
for sentence in word_tokens:
    filtered_words=[]
    
    # 개별 문장별로 tokenize된 sentence list에 대해 stop word 제거 Loop
    for word in sentence:
        #소문자로 모두 변환합니다. 
        word = word.lower()
        
        # tokenize 된 개별 word가 stop words 들의 단어에 포함되지 않으면 word_tokens에 추가
        if word not in stopwords:
            filtered_words.append(word)

    all_tokens.append(filtered_words)
    
print(all_tokens)
```
```
[output]
[['matrix', 'everywhere', 'around', 'us', ',', 'even', 'room', '.'], ['see', 'window', 'television', '.'], ['feel', 'go', 'work', ',', 'go', 'church', 'pay', 'taxes', '.']]
```


<br>



**is**, **this** 와 같은 스톱 워드가 필터링을 통해 제거됐음을 알 수 있습니다.

<br>



## 4. Stemming과 Lemmatization

많은 언어에서 문법적인 요소에 따라 단어가 다양하게 변합니다. 가령 **work** 는 동사 원형인 단어지만, 과거형은 **worked**, 3인칭 단수일 때 **works**, 진행형인 경우 **working** 등 다양하게 달라집니다. **<span style="color:red">Stemming</span>** 과 **<span style="color:red">Lemmatization</span>** 은 **<u>문법적 또는 의미적으로 변화하는 단어의 원형을 찾는 것</u>** 입니다.

- **Stemming**
    - 원형 단어로 변환 시 일반적인 방법을 적용하거나 더 단순화된 방법을 적용해 원래 단어에서 일부 철자가 훼손된 어근 단어를 추출하는 경향이 있습니다.
- **Lemmatization**
    - 품사와 같은 문법적인 요소와 더 의미적인 부분을 감안해 정확한 철자로 된 어근 단어를 찾아줍니다.
    - Stemming과 Lemmatization은 원형 단어를 찾는다는 목적은 유사하지만, Lemmatization이 Stemming보다 정교하며 의미론적인 기반에서 단어의 원형을 찾습니다.

<br>



먼저 **NLTK** 의 **LancasterStemmer** 를 이용해 **Stemmer** 부터 살펴보겠습니다. 진행형, 3인칭 단수, 과거형에 따른 동사, 그리고 비교, 최상에 따른 형용사의 변화에 따라 **<span style="color:red">Stemming</span>** 은 더 단순하게 원형 단어를 찾아줍니다. **NTLK** 에서는 **LancasterStemmer()** 와 같이 필요한 **Stemmer** 객체를 생성한 뒤 이 객체의 **stem('단어')** 메서드를 호출하면 **<u>원하는 '단어'의 Stemming이 가능</u>** 합니다.
```py
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('working'), stemmer.stem('works'), stemmer.stem('worked'))
print(stemmer.stem('amusing'), stemmer.stem('amuses'), stemmer.stem('amused'))
print(stemmer.stem('happier'), stemmer.stem('happiest'))
print(stemmer.stem('fancier'), stemmer.stem('fanciest'))
```
```
[output]
work work work
amus amus amus
happy happiest
fant fanciest
```


<br>




**work** 의 경우 진행형(working), 3인칭 단수(works), 과거형(worked) 모두 기본 단어인 **work** 에 **ing**, **s**, **ed** 가 붙는 단순한 변화이므로 원형 단어로 **work** 를 제대로 인식합니다. 하지만 **amuse** 의 경우, 각 변화가 **amuse** 가 아닌 **amus** 에 **ing**, **s**, **ed** 가 붙으므로 정확한 단어인 **amuse** 가 아닌 **amus** 를 원형 단어로 인식합니다. 형용사인 **happy**, **fancy** 의 경우도 비교형, 최상급형으로 변형된 단어의 정확한 원형을 찾지 못하고 원형 단어에서 철자가 다른 어근 단어로 인식하는 경우가 발생합니다.


이번에는 **WordNetLemmatizer** 를 이용해 **<span style="color:red">Lemmatization</span>** 을 수행해 보겠습니다. 일반적으로 **<u>Lemmatization 은 보다 정확한 원형 단어 추출을 위해 단어의 '품사'를 입력해줘야</u>** 합니다. 다음 예제에서 볼 수 있듯이 **lemmatize()** 의 파라미터로 동사의 경우 **'v'**, 형용사의 경우 **'a'** 를 입력합니다.
```py
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

lemma = WordNetLemmatizer()
print(lemma.lemmatize('amusing','v'), lemma.lemmatize('amuses','v'), lemma.lemmatize('amused','v'))
print(lemma.lemmatize('happier','a'), lemma.lemmatize('happiest','a'))
print(lemma.lemmatize('fancier','a'), lemma.lemmatize('fanciest','a'))
```
```
[output]
amuse amuse amuse
happy happy
fancy fancy
```


<br>


앞의 Stemmer 보다 정확하게 원형 단어를 추출해줌을 알 수 있습니다.



