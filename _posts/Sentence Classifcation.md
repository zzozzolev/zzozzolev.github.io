
YBIGTA 10기 노혜미

# Classification, flagged or not flagged1 <br>- term existence와 tf-idf 이용하기


```python
import pandas as pd
import numpy as np
```

## loading data


```python
data = pd.read_csv('Sheet_1.csv')
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }
    
    .dataframe thead th {
        text-align: left;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>response_id</th>
      <th>class</th>
      <th>response_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>response_1</td>
      <td>not_flagged</td>
      <td>I try and avoid this sort of conflict</td>
    </tr>
    <tr>
      <th>1</th>
      <td>response_2</td>
      <td>flagged</td>
      <td>Had a friend open up to me about his mental ad...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>response_3</td>
      <td>flagged</td>
      <td>I saved a girl from suicide once. She was goin...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>response_4</td>
      <td>not_flagged</td>
      <td>i cant think of one really...i think i may hav...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>response_5</td>
      <td>not_flagged</td>
      <td>Only really one friend who doesn't fit into th...</td>
    </tr>
  </tbody>
</table>
</div>



### data 설명

data는 캐글의 [여기](https://www.kaggle.com/samdeeplearning/deepnlp)에서 가져왔다. <br>
치료 챗봇과 사람이 대화를 하는데 사람이 어떤 반응을 했냐에 따라 flagged가 될 수도 있고 not flagged가 될 수도 있다.<br>
자세히는 모르겠지만, flagged가 되면 도움을 받으라고 챗봇이 메세지를 보낼 것이다.<br>
우리가 풀어야 할 것은 test response가 주어졌을 때, **flagged에 해당하는지 not flagged에 해당하는지** 분류하는 것이다. 

## deleting needless colunms


```python
data = data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
```


```python
x = data['response_text']
y = data['class']
```

# term-existence를 이용한 문서간 유사도 구하기.

## text class


```python
# series를 list로 만든다.

x_list = x.tolist()
```


```python
import string
```


```python
# 단어들을 담을 set을 만들고 text를 공백기준으로 분리한 뒤, 해당 단어들의 punctuation을 없애고 소문자로 만든 뒤 words_set에 더해준다.
# set을 이용하는 이유는 중복을 제거해주기 위해서이다.

words_set = set()
for text in x_list:
    words = text.split()
    for word in words:
        words_set.add(word.strip(string.punctuation).lower())
```


```python
# 단어를 인덱싱한다.

words_dic = {w: i for i, w in enumerate(words_set)}
```


```python
# ''는 아무의미가 없으므로 제거해준다.

del words_dic['']
```


```python
# sklearn의 DictVectorizer를 써서 one hot vector를 만들기 위해서는 딕셔너리의 리스트를 만들 필요가 있다. 

one_hot_dicts = []

for text in x_list:
    words = text.split()
    one_hot_dic = {}
    for word in words:
        word = word.strip(string.punctuation).lower()
        if word in words_dic.keys():
            one_hot_dic[word] = 1
    one_hot_dicts.append(one_hot_dic)
```


```python
x_list[0]
```




    'I try and avoid this sort of conflict'




```python
one_hot_dicts[0]
```




    {'and': 1,
     'avoid': 1,
     'conflict': 1,
     'i': 1,
     'of': 1,
     'sort': 1,
     'this': 1,
     'try': 1}




```python
from sklearn.feature_extraction import DictVectorizer
```


```python
# DictVectorizer를 써서 one hot vector를 만들어준다.

vec = DictVectorizer()
one_hot = vec.fit_transform(one_hot_dicts).toarray()
```


```python
print('words dictionary length:', len(words_dic))
print('one_hot length: ', len(one_hot[0]))
print(one_hot[0])
```

    words dictionary length: 675
    one_hot length:  675
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.]


## train test split

sklearn의 [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)을 더 권장한다...


```python
len(one_hot)
```




    80




```python
# 총 개수가 80개이므로 적당히 75개는 train으로 5개는 test로 한다.

x_train = one_hot[:75]
y_train = y[:75]
x_test = one_hot[75:]
y_test = y[75:]
```

밑의 과정은 x_train의 one_hot이 잘 됐는지 확인하는 것이다. (필요 없다고 생각하면 생략 가능)


```python
index = []

for i in range(len(x_train[0])):
    if x_train[0][i] == 1.:
        index.append(i)
```


```python
index
```




    [34, 55, 120, 290, 409, 529, 594, 619]




```python
for idx in index:
    print(vec.get_feature_names()[idx], end= ' ')
```

    and avoid conflict i of sort this try 


```python
x_list[0]
```




    'I try and avoid this sort of conflict'



잘 된 걸 확인할 수 있다!

### doc product?

- 두 행렬을 곱할 때, 같은 행과 같은 열끼리 곱해주는 연산이다.
- 예를 들면 행렬 A와 B가 있을 때, A의 1행1열과 B의 1행 1열을 곱해주는 식이다.


```python
# one_hot 시킨 test response와 train response를 dot product한다.
# 비슷한 단어가 많을 수록 dot product 값이 클 것이다. 
# 이전의 결과보다 크다면 현재 계산한 doc product의 결과를 추가해준다. 

sim_test = []

for i in range(len(x_test)):
    max_sum = 0.
    max_idx = 0
    for j in range(len(x_train)):
        result = np.dot(x_test[i], x_train[j])
        
        if result > max_sum:
            max_sum = result
            max_idx = j
            sim_test.append('test{} -> train{}'.format(i+len(x_train), max_idx))

```


```python
# 계속 비교해서 큰 걸 추가시켰으니, 맨 마지막에 있는 게 가장 큰 것이다. 
# 가장 큰 거만 출력하고 싶은데 못하겠다 ㅠㅠ...

sim_test
```




    ['test75 -> train0',
     'test75 -> train1',
     'test75 -> train4',
     'test75 -> train22',
     'test75 -> train48',
     'test76 -> train1',
     'test76 -> train2',
     'test76 -> train9',
     'test76 -> train33',
     'test76 -> train38',
     'test77 -> train0',
     'test77 -> train2',
     'test77 -> train22',
     'test77 -> train41',
     'test78 -> train0',
     'test78 -> train1',
     'test78 -> train2',
     'test78 -> train4',
     'test78 -> train36',
     'test78 -> train48',
     'test79 -> train0',
     'test79 -> train1',
     'test79 -> train2',
     'test79 -> train36',
     'test79 -> train48']




```python
# test와 가장 비슷하다고 나온 train의 결과값을 예측 list에 차례대로 추가해준다.

predicted_list = []
predicted_list.append(y_train[48])
predicted_list.append(y_train[38]) 
predicted_list.append(y_train[41]) 
predicted_list.append(y_train[48]) 
predicted_list.append(y_train[48]) 
```

과연...?


```python
correct = 0
for i in range(len(y_test)):
    if y_test[i+len(x_train)]==predicted_list[i]:
        correct += 1 
print(len(y_test), "개 중에", correct,"개 맞췄음.")
```

    5 개 중에 3 개 맞췄음.
​    

크~구린(?) 방법임에도 반 이상은 맞았다.<br>
데이터 자체가 워낙 작으니 이런 방법을 쓸 수 있던 것이지 데이터가 크다면 시도하면 안 된다...!<br>
모든 단어들의 크기가 one hot vector의 크기가 되기 때문이다.

# TF-IDF를 이용한 문서간 유사도 구하기

## TF-IDF ?

참고:
[위키백과](https://ko.wikipedia.org/wiki/TF-IDF)

- term frequency - inverse document frequency의 약자이다.
- 어떤 단어가 문서에서 **얼마나 중요한지**를 나타내는 통계적 수치.
- 문서의 핵심어, 문서 간의 유사도등을 구할 때 이용할 수 있다.
- 단어 빈도인 TF는 문서에서 단어가 얼마나 등장하는 지를 나타내는 값이다.
- 역문서 빈도인 IDF는 단어가 여러 문서에서 잘 나오지 않는 지를 나타내는 값이다. 만약 여러 문서에서 잘 나타나지 않는 단어라면 IDF 값은 높아질 것이다.
- 그리고 TF-IDF값은 이 두 값을 곱한 결과이다.
- 예를 들어 보자. A,B,C,D 문서가 있을 때 A 문서내에 있는 ybigta라는 단어의 TF-IDF 값이 크다면 ybigta라는 단어는 A문서 내에서 많이 등장하고 다른 문서에서는 잘 등장하지 않는 단어이다. 따라서 A문서의 고유한 특성이 될 수 있다...! 즉, A문서를 대표하는 값으로 쓸 수 있다는 말이다.

소스 코드 출처: <br>
https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity

## skleran 


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
x = vec.fit_transform(x_list)
x
```




    <80x660 sparse matrix of type '<class 'numpy.float64'>'
    	with 1863 stored elements in Compressed Sparse Row format>



### cosine similarity ?

- 간단하게 말하면 두 비교 대상이 얼마나 가까운지를 측정하는 방법이다.<br>
- 자세하게 알고 싶다면 [코사인 유사도](http://euriion.com/?p=548)를 참고하기 바란다.


```python
from sklearn.metrics.pairwise import linear_kernel

# linear_kernel은 dot product이다.
# doc product를 통해 cosine similarity를 구한다.

cos_sim_list = []
for i in range(75,80):
    cosine_sim = linear_kernel(x[i], x).flatten()
    cos_sim_list.append(cosine_sim)
```


```python
# test 문장과 가장 가까운 문장을 2개 뽑는다.
# argsort()는 가장 큰 값을 1으로 indexing하고 그 다음 큰 값을 2로 indexing하고 이를 
# 이를 반복하는 메서드이다.

rel_doc_list = []

for i in range(len(cos_sim_list)):
    related_doc_idx = cos_sim_list[i].argsort()[:-3:-1]
    rel_doc_list.append(related_doc_idx)
```


```python
# 2개를 뽑은 이유는 test역시 cosine similarity를 구할 때 고려됐으므로, 똑같은 문장 
# 다음에 큰 값이 필요하기 때문이다.

rel_doc_list
```




    [array([75, 48], dtype=int64),
     array([76, 38], dtype=int64),
     array([77, 32], dtype=int64),
     array([78, 57], dtype=int64),
     array([79, 63], dtype=int64)]




```python
# 가장 유사한 문장을 추가해준다.

most_similar_doc = []

most_similar_doc.append(y_train[48])
most_similar_doc.append(y_train[38])
most_similar_doc.append(y_train[32])
most_similar_doc.append(y_train[57])
most_similar_doc.append(y_train[63])
```


```python
correct = 0
for i in range(len(y_test)):
    if y_test[i+len(x_train)]==most_similar_doc[i]:
        correct += 1 
print(len(y_test), "개 중에", correct,"개 맞췄음.")
```

    5 개 중에 4 개 맞췄음.
​    

확실히 단순하게 one-hot vector를 이용하는 것보다 좋은 결과가 나왔다.<br>
그리고 문서내에서 키워드를 추출하거나 비교를 할 때 tf-idf를 많이 이용하는 것 같다!

## Conclusion

- term-existence보다는 tf-idf를 이용하자~

## Reference

- https://www.kaggle.com/samdeeplearning/deepnlp
- https://www.lucypark.kr/slides/2015-pyconkr/#13
- https://ko.wikipedia.org/wiki/TF-IDF
- https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity


