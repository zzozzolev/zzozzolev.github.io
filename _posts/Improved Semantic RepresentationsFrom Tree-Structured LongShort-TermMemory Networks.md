# Improved Semantic RepresentationsFrom Tree-Structured LongShort-TermMemory Networks

## 1. Long Short-Term Memory Networks

### 1.1 Overview

####1.1.1 LSTM

- LSTM은 $R^d$ vector들의 collection이다. 여기서 d는 LSTM의 memory dimension이다. 
- input gate, forget gate, output gate, memory cell, hidden state가 있다.

![Imgur](https://i.imgur.com/kfQHehd.png)

- input gate, forget gate, output gate는 0에서 1사이의 값을 갖는다. 

- forget gate는 이전 memory cell을 얼마나 잊을 지를 결정한다.

- input gate는 각각의 unit을 얼마나 update할 것인지 결정한다.

- output gate는 내부 memory state를 얼마만큼 이용할 건지를 결정한다.

  ####1.1.2 Dependency vs. constituency

  ![Imgur](https://i.imgur.com/ArbjJng.png)

  - dependency grammar는 단어를 중심으로 하고, constituency grammar는 문장 성분을 중심으로 한다.  즉, dependency grammar에서는 특정 단어가 어떤 단어에 의존하는지(수식하는지)에 관심이 있고, constituency grammar에서는 구나 단어가 문장 내에서 어떤 역할(명사구, 동사구 등등)을 하는지에 관심이 있다.

  ##2. Tree-Structured LSTMs

- basic LSTM 구조의 두 가지 응용인 the $Child-Sum Tree-LSTM$, the $N-ary Tree-LSTM$을 소개하겠다.

- 두 구조는 각각의 LSTM unit이 multiple child unit의 정보를 합칠 수 있게 해준다.

![Imgur](https://i.imgur.com/JlZjXmJ.png)

- 일반적인 LSTM unit과 Tree-LSTM이 다른 점이 있다면, gating vector들과 memory cell의 update가 많은 child unit들에 의존한다는 것이다. 그리고 forget gate는 각각의 child k만큼 있게 된다. 
- 위와 같은 구조로 Tree-LSTM은 문장에서 중심적인 의미를 가진 단어(semantic head)를 강조하는 것을 배우게 된다. 그리고 감성 분류에서 sentiment를 잘 담고 있는 children의 representation을 보존하는 것도 배울 수 있다.
- 논문에서 각각의 $x_j$는 문장 내에서 단어의 vector 표현이다.

### 2.1 Child-Sum Tree-LSTMs

![Imgur](https://i.imgur.com/wEykU51.jpg)

![Imgur](https://i.imgur.com/tecqQ7n.png)

*$C(j)$는 j번째 node의 children을 가리킨다.

- 각각의 parameter matrix(W)를 Tree-LSTM unit의 component vector들과 input, unit의 children의 hidden states 사이의 상관 관계를 encoding한 거로 이해하면 된다. 예를 들면 동사와 같이 의미적으로 중요한 word가 input으로 주어지면 $W^{(i)}$의 값이 1에 가깝게 된다.
- Child-Sum Tree-LSTM의 unit은 child hidden states의 합을 참고해 components의 값을 결정하기 때문에 각 node의 children의 수(branching factor)가 많거나 children에 문장의 순서가 반영되지 않았을 때 적절하다.

*head란 구의 문법적 범주를 결정하는 단어를 말한다. 예를 들어 big red dog라는 구가 있을 때 명사 구란 것을 결정짓는 것은 dog이기 때문에 dog가 head이고 나머지 big과 red는 dog를 수식하기 때문에 dog의 dependent이다.

- dependency tree에 적용된 Child-Sum Tree-LSTM를 *Dependency Tree-LSTM*라고 칭하겠다.

### 2.2 N-ary Tree-LSTMs

- branching factor가 최대 N개이고 children이 순서대로 정렬되어 있을 때 사용된다.

![Imgur](https://i.imgur.com/g2oGYtA.png)

![Imgur](https://i.imgur.com/GWZm0Kx.png)

- Child-Sum Tree-LSTM과 다른 점은 딱 형광펜으로 강조한 부분 밖에 없다. child의 순서를 고려하는게 유일한 차이인 것 같다.
- 각각의 child에 대해 별개의 parameter matrix를 설정해주는 것은 model이 unit의 children의 state에 대해 더 잘 값을 설정할 수 있게 해준다. 예를 들면 왼쪽 child가 명사구이고 오른쪽 child가 동사구라고 하면, 동사 구를 강조해 주는 게 더 좋다. 따라서 명사구에 해당하는 forget gate의 components를 0에 가깝게 설정되도록 하고 동사구에 해당하는 forget gate는 반대로 설정되도록 학습시켜주면 된다.
- child를 왼쪽과 오른쪽으로 구분할 수 있기 때문에, 일반적으로 *Binary Tree-LSTM* unit을 *binarized constituency trees*에 적용한다.
- *Binary Tree-LSTM*의 적용을 *Constituency Tree-LSTM*이라고 하겠다.
- 적용에서의 주요한 차이는 compositional parameter이다. Dependency Tree-LSTM에서는 **dependent와 head**, Constituency Tree-LSTM에서는 **left child와 right child**가 되겠다. 



## 3. Models

### 3.1 Tree-LSTM Classification

- 별개의 class인 y를 예측하는 작업을 한다.

- 각각의 node j에서  $\{x\}_j$에 해당하는 label $\hat y_j$를 맞추기 위해 softmax classifier를 사용할 것이다.

- classifier는 해당 node의 hidden state $h_j$를 input으로 받을 것이다.

   ![Imgur](https://i.imgur.com/tqCe5e1.png)

- cost function은 true class label $y^{(k)}$의 negative log-likelihood이다.

![Imgur](https://i.imgur.com/Wrg2XsS.png)

$m$: training set에서 labeling된 node의 개수

$k$: k번째 labeled node

$\lambda$ :  L2 regularization의 hyperparameter

- 해당 논문에서는 sentiment classification을 했다.

### 3.2  Semantic Relatedness of Sentence Pairs

- [1, K]의 범주를 가지는 sentence pair의 유사도를 구한다. K는 정수이다.

- 높은 점수는 높은 유사도를 가지는 걸 나타낸다. 

- 일단 Tree-LSTM model을 사용해서 $h_L, h_R$ sentence representation을 만든다.

-  $(h_L, h_R)$의 사이의 거리와 각을 이용해 유사도를 예측한다.

  ![Imgur](https://i.imgur.com/90GzEK4.png)

*$r^T$ = [1 2 ... K]

- $h_X$는 input representation의 elementwise comparison이다.

- 예측 분포인 $\hat p_\theta$와 $r^T$ 내적한 값인 $\hat y$가 y와 비슷하게 나오도록 해야했다.

- 그래서 $y = r^Tp$를 만족하는 sparse한 target distribution인 p를 다음과 같이 정의했다.

  ![Imgur](https://i.imgur.com/GFWTaTK.png)

- $r^T$가 점수 class이므로 p는 해당 점수가 나올 확률이 된다. 그러므로 $r^T$와 p를 곱했을 때, 해당 class의 값이 가장 크게 나와야 한다.

- 예를 들어 유사도가 10.7이라고 해보자. 그렇다면 $p_{11}$이 가장 큰 값을 가져야 한다.  이때 $p_{11}= 0.7,$ $p_{10}= 0.3$이 된다. 따라서 10.7과 가장 가까운 정수값인 11이 내적에 큰 영향을 주게되는 것이다.

- cost 함수는 $p$와 $\hat p_\theta$사이의 정규화된 KL-divergence이다.

  ![Imgur](https://i.imgur.com/vgshbkv.png)

  m: training pair의 개수

  k: k번째 sentence pair 



## 4. Results

### 4.1 Sentiment Classification

![Imgur](https://i.imgur.com/afY2DDe.png)

*Fine-grained: 5-class sentiment classification. 

*Binary: positive/negative sentiment classification.

- constituency Tree-LSTM이 가장 성적이 좋았다.
- training 동안 word representation을 update하는 것(tuned)은 fine-grained에서는 좋은 결과를 내는 데 많은 기여를 했지만, binary classification에서는 미미한 영향을 끼쳤다.

### 4.2 Semantic Relatedness

![Imgur](https://i.imgur.com/60YE27H.png)

*pearson's r, spearmam's p, mse는 evaluation matrics

*처음 두 metrics는 사람의 평가와 비교했을 때 상관 관계

- Tree-LSTM 모델이 어떤 feature engineering이 없이 다른 system보다 성능이 좋았다.
- 제일 좋은 결과는 dependency Tree-LSTM을 사용했을 때였다.
- sentiment classification task와 달리 true label값(supervision)을 tree의 root에서만 받았다.



## Reference

- Introduction to Tree-LSTMs (해당 논문에 대한 presentation)                                                  https://www.slideshare.net/tuvistavie/tree-lstm
- https://en.wikipedia.org/wiki/Head_(linguistics)
- http://taweb.aichi-u.ac.jp/tmgross/DG.html