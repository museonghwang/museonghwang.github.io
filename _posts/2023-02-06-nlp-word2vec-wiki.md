---
layout: post
title: 한국어 위키피디아로 Word2Vec 학습하기
category: NLP(Natural Language Processing)
tag: NLP(Natural Language Processing)
---





한국어 위키피디아 데이터로 **Word2Vec** 학습을 진행해보겠습니다.

<br>





# 1. 위키피디아로부터 데이터 다운로드 및 통합

위키피디아로부터 데이터를 파싱하기 위한 파이썬 패키지인 **wikiextractor** 를 설치하겠습니다.
```
pip install wikiextractor
```


<br>


위키피디아 덤프(위키피디아 데이터)를 다운로드 하겠습니다.
```
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
```


<br>


**wikiextractor** 를 사용하여 위키피디아 덤프를 파싱합니다.
```
python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2
```
```
[output]
INFO: Preprocessing 'kowiki-latest-pages-articles.xml.bz2' to collect template definitions: this may take some time.
INFO: Preprocessed 100000 pages
INFO: Preprocessed 200000 pages
INFO: Preprocessed 300000 pages
INFO: Preprocessed 400000 pages
INFO: Preprocessed 500000 pages
INFO: Preprocessed 600000 pages
INFO: Preprocessed 700000 pages
INFO: Preprocessed 800000 pages
INFO: Preprocessed 900000 pages
INFO: Preprocessed 1000000 pages
INFO: Preprocessed 1100000 pages
INFO: Preprocessed 1200000 pages
INFO: Preprocessed 1300000 pages
INFO: Preprocessed 1400000 pages
INFO: Preprocessed 1500000 pages
INFO: Preprocessed 1600000 pages
INFO: Preprocessed 1700000 pages
INFO: Preprocessed 1800000 pages
INFO: Loaded 63223 templates in 375.0s
INFO: Starting page extraction from kowiki-latest-pages-articles.xml.bz2.
INFO: Using 3 extract processes.
INFO: Extracted 100000 articles (1147.9 art/s)
INFO: Extracted 200000 articles (1685.0 art/s)
INFO: Extracted 300000 articles (1874.9 art/s)
INFO: Extracted 400000 articles (1980.8 art/s)
INFO: Extracted 500000 articles (2131.4 art/s)
INFO: Extracted 600000 articles (2112.3 art/s)
INFO: Extracted 700000 articles (2165.1 art/s)
INFO: Extracted 800000 articles (2406.7 art/s)
INFO: Extracted 900000 articles (5877.7 art/s)
INFO: Extracted 1000000 articles (3890.4 art/s)
INFO: Extracted 1100000 articles (2399.0 art/s)
INFO: Extracted 1200000 articles (2572.4 art/s)
INFO: Extracted 1300000 articles (2481.7 art/s)
INFO: Extracted 1400000 articles (2611.3 art/s)
INFO: Finished 3-process extraction of 1400056 articles in 634.2s (2207.6 art/s)
```


<br>



현재 경로에 있는 디렉토리와 파일들의 리스트를 받아오겠습니다.
```
%ls
```
```
kowiki-latest-pages-articles.xml.bz2  sample_data/  text/
```


<br>




**text** 라는 디렉토리 안에는 또 어떤 디렉토리들이 있는지 파이썬을 사용하여 확인해보겠습니다.
```py
import os
import re

os.listdir('text')
```
```
[output]
['AG', 'AA', 'AI', 'AH', 'AD', 'AB', 'AE', 'AJ', 'AF', 'AC']
```


<br>


**AA** 라는 디렉토리의 파일들을 확인해보겠습니다.
```
%ls text/AA
```
```
[output]
wiki_00  wiki_12  wiki_24  wiki_36  wiki_48  wiki_60  wiki_72  wiki_84  wiki_96
wiki_01  wiki_13  wiki_25  wiki_37  wiki_49  wiki_61  wiki_73  wiki_85  wiki_97
wiki_02  wiki_14  wiki_26  wiki_38  wiki_50  wiki_62  wiki_74  wiki_86  wiki_98
wiki_03  wiki_15  wiki_27  wiki_39  wiki_51  wiki_63  wiki_75  wiki_87  wiki_99
wiki_04  wiki_16  wiki_28  wiki_40  wiki_52  wiki_64  wiki_76  wiki_88
wiki_05  wiki_17  wiki_29  wiki_41  wiki_53  wiki_65  wiki_77  wiki_89
wiki_06  wiki_18  wiki_30  wiki_42  wiki_54  wiki_66  wiki_78  wiki_90
wiki_07  wiki_19  wiki_31  wiki_43  wiki_55  wiki_67  wiki_79  wiki_91
wiki_08  wiki_20  wiki_32  wiki_44  wiki_56  wiki_68  wiki_80  wiki_92
wiki_09  wiki_21  wiki_33  wiki_45  wiki_57  wiki_69  wiki_81  wiki_93
wiki_10  wiki_22  wiki_34  wiki_46  wiki_58  wiki_70  wiki_82  wiki_94
wiki_11  wiki_23  wiki_35  wiki_47  wiki_59  wiki_71  wiki_83  wiki_95
```


<br>




**텍스트 파일로 변환된 위키피디아 한국어 덤프는 총 10개의 디렉토리로 구성** 되어져 있습니다. **AA ~ AJ** 의 디렉토리로 각 디렉토리 내에는 **'wiki_00 ~ wiki_약 90내외의 숫자'** 의 파일들이 들어있습니다. 다시 말해 각 디렉토리에는 약 90여개의 파일들이 들어있습니다. 각 파일들을 열어보면 다음과 같은 구성이 반복되고 있습니다.

```
<doc id="문서 번호" url="실제 위키피디아 문서 주소" title="문서 제목">

내용

</doc>
```

<br>


예를 들어서 **AA** 디렉토리의 **wiki_00** 파일을 읽어보면, 지미 카터에 대한 내용이 나옵니다.
```
<doc id="5" url="https://ko.wikipedia.org/wiki?curid=5" title="지미 카터">
지미 카터
제임스 얼 "지미" 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39번째 대통령(1977년 ~ 1981년)이다.
지미 카터는 조지아 주 섬터 카운티 플레인스 마을에서 태어났다. 조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대
위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다.
... 이하 중략...
</doc>
```

<br>



이제 이 10개 **AA ~ AJ** 디렉토리 안의 wiki 숫자 형태의 수많은 파일들을 하나로 통합하는 과정을 진행해야 합니다. **AA ~ AJ** 디렉토리 안의 모든 파일들의 경로를 파이썬의 리스트 형태로 저장하겠습니다.
```py
def list_wiki(dirname):
    filepaths = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)

        if os.path.isdir(filepath):
            # 재귀 함수
            filepaths.extend(list_wiki(filepath))
        else:
            find = re.findall(r"wiki_[0-9][0-9]", filepath)
            if 0 < len(find):
                filepaths.append(filepath)
    return sorted(filepaths)
```

```py
filepaths = list_wiki('text')
```


<br>


총 파일의 개수를 확인하겠습니다.
```py
len(filepaths)
```
```
[output]
970
```


<br>


**총 파일의 개수는 970개** 입니다. 이제 **output_file.txt** 라는 파일에 970개의 파일을 전부 하나로 합치겠습니다.
```py
with open("output_file.txt", "w") as outfile:
    for filename in filepaths:
        with open(filename) as infile:
            contents = infile.read()
            outfile.write(contents)
```


<br>


파일을 읽고 10줄만 출력해보겠습니다.
```py
f = open('output_file.txt', encoding="utf8")

i = 0
while True:
    line = f.readline()
    if line != '\n':
        i = i + 1
        print("%d번째 줄 :"%i + line)
    if i==10:
        break 
f.close()
```
```
[output]
1번째 줄 :<doc id="5" url="https://ko.wikipedia.org/wiki?curid=5" title="지미 카터">

2번째 줄 :지미 카터

3번째 줄 :제임스 얼 카터 주니어(, 1924년 10월 1일~)는 민주당 출신 미국의 제39대 대통령(1977년~1981년)이다.

4번째 줄 :생애.

5번째 줄 :어린 시절.

6번째 줄 :지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.

7번째 줄 :조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다. 그의 별명이 "땅콩 농부" (Peanut Farmer)로 알려졌다.

8번째 줄 :정계 입문.

9번째 줄 :1962년 조지아주 상원 의원 선거에서 낙선하였으나, 그 선거가 부정선거 였음을 입증하게 되어 당선되고, 1966년 조지아 주지사 선거에 낙선하지만, 1970년 조지아 주지사 선거에서 당선됐다. 대통령이 되기 전 조지아주 상원의원을 두번 연임했으며, 1971년부터 1975년까지 조지아 지사로 근무했다. 조지아 주지사로 지내면서, 미국에 사는 흑인 등용법을 내세웠다.

10번째 줄 :대통령 재임.
```


<br>





# 2. 형태소 분석


형태소 분석기 **Mecab** 을 사용하여 토큰화를 진행해보겠습니다.
```py
from tqdm import tqdm
from konlpy.tag import Mecab 

mecab = Mecab()
```


<br>



우선 **output_file** 에는 총 몇 줄이 있는지 확인하겠습니다.
```py
f = open('output_file.txt', encoding="utf8")

lines = f.read().splitlines()
print(len(lines))
```
```
[output]
11003273
```


<br>






11,003,273개의 줄이 존재합니다. 상위 10개만 출력해보겠습니다.
```py
lines[:10]
```
```
[output]
['<doc id="5" url="https://ko.wikipedia.org/wiki?curid=5" title="지미 카터">',
 '지미 카터',
 '',
 '제임스 얼 카터 주니어(, 1924년 10월 1일~)는 민주당 출신 미국의 제39대 대통령(1977년~1981년)이다.',
 '생애.',
 '어린 시절.',
 '지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.',
 '조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다. 그의 별명이 "땅콩 농부" (Peanut Farmer)로 알려졌다.',
 '정계 입문.',
 '1962년 조지아주 상원 의원 선거에서 낙선하였으나, 그 선거가 부정선거 였음을 입증하게 되어 당선되고, 1966년 조지아 주지사 선거에 낙선하지만, 1970년 조지아 주지사 선거에서 당선됐다. 대통령이 되기 전 조지아주 상원의원을 두번 연임했으며, 1971년부터 1975년까지 조지아 지사로 근무했다. 조지아 주지사로 지내면서, 미국에 사는 흑인 등용법을 내세웠다.']
```


<br>




두번째 줄을 보면 아무런 단어도 들어있지 않은 **''** 와 같은 줄도 존재합니다. 해당 문자열은 형태소 분석에서 제외하도록 하고 형태소 분석을 수행하겠습니다.
```py
result = []

for line in tqdm(lines):
    # 빈 문자열이 아닌 경우에만 수행
    if line:
        result.append(mecab.morphs(line))
```
```
[output]
100%|██████████| 11003273/11003273 [14:52<00:00, 12328.56it/s]
```


<br>




빈 문자열은 제외하고 형태소 분석을 진행했습니다. 이제 몇 개의 줄. 즉, 몇 개의 문장이 존재하는지 확인해보겠습니다.
```py
len(result)
```
```
[output]
7434242
```


<br>





7,434,242개로 문장의 수가 줄었습니다.

<br>






# 3. Word2Vec 학습

형태소 분석을 통해서 토큰화가 진행된 상태이므로 **Word2Vec** 을 학습합니다.
```py
from gensim.models import Word2Vec

model = Word2Vec(
    result,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=0
)
```


<br>


```py
model_result1 = model.wv.most_similar("대한민국")
print(model_result1)
```
```
[output]
[('한국', 0.7304196357727051), ('미국', 0.6692186594009399), ('일본', 0.6360577344894409), ('부산', 0.5860438346862793), ('홍콩', 0.5758174061775208), ('태국', 0.5587795972824097), ('오스트레일리아', 0.5537664294242859), ('서울', 0.551203191280365), ('중화민국', 0.5402842164039612), ('대구', 0.5301536321640015)]
```


<br>

```py
model_result2 = model.wv.most_similar("어벤져스")
print(model_result2)
```
```
[output]
[('어벤저스', 0.7947301864624023), ('엑스맨', 0.7730240821838379), ('아이언맨', 0.7690291404724121), ('스파이더맨', 0.7656212449073792), ('테일즈', 0.7625358700752258), ('에일리언', 0.7526578903198242), ('트랜스포머', 0.7525742053985596), ('솔저', 0.7348356246948242), ('헐크', 0.7259199619293213), ('스타트렉', 0.7257904410362244)]
```


<br>


```py
model_result3 = model.wv.most_similar("반도체")
print(model_result3)
```
```
[output]
[('연료전지', 0.794660210609436), ('집적회로', 0.7878057956695557), ('전자', 0.7571590542793274), ('웨이퍼', 0.7522547841072083), ('실리콘', 0.7403302788734436), ('트랜지스터', 0.7359127402305603), ('그래핀', 0.7252676486968994), ('PCB', 0.703812837600708), ('가전제품', 0.6887932419776917), ('전기차', 0.6878519058227539)]
```


<br>


```py
model_result4 = model.wv.most_similar("자연어")
print(model_result4)
```
```
[output]
[('구문', 0.7062239646911621), ('메타데이터', 0.7035123109817505), ('시각화', 0.6815322637557983), ('설명서', 0.6801173686981201), ('텍스트', 0.6770368814468384), ('말뭉치', 0.6747072339057922), ('매크로', 0.6737524271011353), ('데이터베이스', 0.6698500514030457), ('스키마', 0.6683913469314575), ('XML', 0.6677115559577942)]

```





