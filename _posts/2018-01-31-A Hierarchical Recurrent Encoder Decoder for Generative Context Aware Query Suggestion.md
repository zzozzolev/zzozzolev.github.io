---
comments: true
title: "A Hierarchical Recurrent Encoder Decoder for Generative Context Aware Query Suggestion Review"
date: 2018-01-31 20:56
categories: 2015 RNN encoder-decoder generative-model 
---
# Abstract
- 임의의 길이를 갖는 이전 쿼리 sequence를 고려할 수 있는 확률적 suggestion 모델이다. 즉, 맥락을 알아내 generation하는 모델이다. 각각의 encoder와 decoder에 GRU를 이용했고 LTR 알고리즘을 사용했다. HRED를 사용하니 빈번하지 않은 context에 대해서도 잘 예측할 수 있었다.


# 1. Introduction
- generative한 통계 모델을 제시했다.
- 많이 발생한 쿼리뿐만 아니라 training data에서 보지 못한 쿼리 역시 synthetic하고 맥락을 아는 모델이다. 
- 주어진 순차적인 쿼리를 prefix로 가정하고, prefix 다음에 올 가장 확률이 높은 순차적인 단어를 예측한다.
- 쿼리 suggestions은 주어진 하나 이상의 쿼리에 대해 가능성 있는 연속적인 단어를 샘플링해서 얻어낼 수 있다.
- 예를 들어, 유저 쿼리 세션이 cleveland grallery -> lake erie art와 같이 두 가지 쿼리로 이루어져 있다면, 모델은 순서대로 cleveland, indian, art, $$\circ$$을 예측한다. ($$\circ$$은 end-of-query symbol)
- suggestion은 첫번째 쿼리로 봤을 때 cleveland의 concept이 적절하기 때문에 contextual하다. 그래서 모델은 단지 최근의 쿼리에만 의존하지 않는다. 그리고 만들어진 쿼리는 training set에 있을 필요가 없기 때문에 synthetic하다.
- count-based 모델과 달리 하나의 단어들, 쿼리, 순차적인 쿼리를 임베딩에 할당해 data sparsity를 피했다.
- 우리 모델은 메모리를 많이 차지하지 않고 쿼리 세션에서 end-to-end 학습될 수 있다.


# 2. Key Idea
- suggestion 모델은 쿼리 사이의 근본적인 유사도를 알아내야 한다.
- 해당 논문에서는 문법적인이고 의미적인 특성을 인코딩하는 임베딩을 이용해서 어떻게 쿼리의 유사도와 쿼리 term의 유사도를 알아내는지에 초점을 맞췄다.
- 모델로 학습시킨 임베딩의 벡터들은 유사한 주제일 수록 가까이에 있다.
- 일반적으로 구절에 대한 벡터 표현은 단어들의 벡터를 평균내서 얻지만, 쿼리에서 단어의 순서는 중요하기 때문에 RNN을 이용했다.
- RNN을 이용한 인코딩과 디코딩은 이전 쿼리에 대해 다음 쿼리를 예측하는데 쓰일 수 있다.

![Imgur](https://i.imgur.com/Qile1ni.png)

모델은 연속적인 쿼리인 cleveland gallery와 lake erie art 사이의 mapping을 학습한다.
- 테스트 시 유저의 쿼리는 인코딩 되고 가능성 있는 연속적인 단어로 디코딩 된다.
- 비록 강력하지만...그런 mapping은 pairwise해서 쿼리의 context를 잃게 된다. 
- 쿼리의 예측을 조건화하기 위해 session-level RNN을 query-level RNN encoder에 추가했다.

**The hierarchical recurrent encoder-decoder (HRED) for query suggestion**

![Imgur](https://i.imgur.com/VMSD3OD.png)

- query-level RNN은 쿼리를 인코딩하고, session-level RNN은 쿼리 인코딩을 인풋으로 받아 자신의 recurrent state를 업데이트한다.
- session-level recurrent state는 과거 쿼리를 학습한 요약이다.
- decoder RNN은 session-level recurrent state를 인풋으로 받아서 다음에 나올 쿼리를 contextual하게 만들어낸다.
- query-level encoder RNN은 비슷한 쿼리를 임베딩 공간에서 비슷한 벡터로 mapping한다.
- mapping은 쿼리에서 나오는 단어들이 model의 vocabulary에 나타나는 한, training data에서 보지 못했던 쿼리를 일반화한다. (training에 없던 것도 맞출 수 있게 해준다는 소리인 듯)
- 이러한 방법은 모델이 드문 쿼리에 대해 단순히 동시에 발생했던 쿼리를 예측했던 걸 넘어서 더 유용하고 일반적인 formulation에 mapping하게 만들어준다.


# 3. Mathematical Framework

## 3.2 Architecture
- hierarchical recurrent encoder-decoder는 위의 그림과 같다.
- forward에서 query-level encodings, session-level recurrent states, 각각의 쿼리에서 log-likelihood를 계산한다.
- backward에서는 gradients가 계산되고, parameter들이 업데이트된다.

### 3.2.1 Query-Level Encoding
- training session S에 있는 각각의 쿼리 $$Q_m = {w_{m,1},...,w_{m,N_m}}$$($$m$$은 쿼리의 순서, $$N_m은 쿼리의 길이$$)에 대해 query-level RNN은 쿼리의 단어를 순차적으로 읽고 hidden state를 GRU를 이용해 업데이트 한다.
- 요약하자면, query-level RNN encoder는 쿼리를 고정된 길이의 벡터로 mapping한다.
- parameter들은 공유되므로 recurrent state는 쿼리에 대해 general하고 contextual한 representation을 갖게 된다.

### 3.2.2 Session-Level Encoding
- session-level RNN은 쿼리 representation의 sequence(query-level RNN의 가장 마지막에 있는 recurrent state)를 인풋으로 받아서 session-level recurrent states를 계산한다.
- 역시 GRU를 사용한다.
- session-level recurrent state $$s_m$$는 m번째까지의 쿼리에 대해 요약한다.
- $$s_m$$은 이전 쿼리의 순서에 대해 민감하기 때문에 순서에 의존하는 reformulation을 인코딩한다.
- 게다가 query vector를 인풋으로 받기 때문에 쿼리 내에 있는 단어의 순서에도 민감하다.

### 3.2.3 Next-Query Decoding
- RNN decoder는 $$Q_{1:m-1}$$이 주어졌을 때 다음 쿼리$$Q_m$$을 예측한다.<p>
$$P(Q_m|Q_{1:m-1})=\prod^{N_m}_{n=1}P(w_n|w_{1:n-1},Q_{1:m-1})$$
- 이전 쿼리에 대한 조건화는 RNN decoder의 recurrence를, $$s_{m-1}$$의 non-linear transformation을 통해 초기화함으로써 이루어진다.<p>
$$d_{m,0}=tanh(D_{0S_{m-1}}+b_0)$$<p>
$$d_{m,0}$$은 decoder의 초기 recurrent state<p>
- recurrence는 다음과 같은 form을 갖는다.<p>
$$d_{m,n}=GRU_{dec}(d_{m,n-1},w_{m,n}), n=1,...,N_m$$<p>
$$GRU_{dec}$$는 decoder GRU, recurrent state $$d_{m,n-1}$$는 다음 단어 $$w_{m,n}$$의 확률을 계산하는데 사용된다.
- 다음 단어 $$w_{m,n}$$에 대한 확률은 다음의 식으로 계산된다.
 
![Imgur](https://i.imgur.com/MtCWJou.png)

$$o_v$$는 v의 embedding, $$\omega$$는 함수이다.
$$\omega(d_{m,n-1},w_{m,n-1})=H_o d_{m,n-1}+E_o w_{m,n-1}+b_0$$
$$H_o$$와 $$E_o$$은 parameter, 만약 $$o_v$$가 vector $$\omega(d_{m,n-1},w_{m,n-1})$$과 가까이 있다면 모델 내에서 $$w_{m,n}은 높은 확률을 갖는다.$$

## 3.3 Learning
- model의 parameter들은 encoder GRU, decoder GRU, session GRU 3개의 GRU의 parameter들로 구성되어있다.
- output parameter들은 $$H_o,E_o,b_o,o_i$$이다.
- 위의 parameter들은 session S의 log-likelihood를 최대화하도록 학습된다.

![Imgur](https://i.imgur.com/auwd8PB.png)

## 3.4 Generation and Rescoring
**Generation**
- 유저가 쿼리를 순서대로 입력하면, 쿼리 suggestion은 쿼리 $$Q*$$이다.
$$Q*=argmax_{Q\in \mathcal{Q}}P(Q|Q_{1:M}$$
$$\mathcal{Q}$$는 가능한 쿼리 공간
*Example*

![Imgur](https://i.imgur.com/VMSD3OD.png)

- 유저가 cleveland gallery -> lake erie artist라는 쿼리를 입력했다고 해보자.
- 쿼리 내에 있는 단어들은 GRU를 통해 query-level encoding이 된다. 따라서 쿼리 vector인 $$q_{cleveland gallery}$$와 $$q_{lake erie art}$$를 얻게 된다.
- 그 다음, 쿼리 벡터에 GRU를 이용해 session-level recurrent state를 계산한다. 이를 통해, 두 개의 session-level recurrent state인 $$s_{cleveland gallery}$$와 $$s_{lake erie art}$$를 얻는다.
- 마지막 session-level recurrent state를 초기 decoder input으로 넣는다.(tanh 이용)
- beam search 크기가 1이라고 가정하자.
- 첫번째 단어인 $$w_1$$이 나올 확률은 softmax를 이용해서 계산되는데, 이때 $$d_0$$와 $$w_0$$는 null vector이다. (논문에서 언급은 안했지만 b_0는 1로 초기화해서 $$O_v$$를 거의 그대로 유지하는 것 같다.) 
- 가장 높은 확률을 가진 단어인 cleveland는 beam에 추가된다.
- 그 다음 decoder recurrent state인 $$d_1$$은 $$d_0$$와 $$w_1=cleveland$$를 이용해 decoder GRU에 의해 계산된다.
- $$d_1$$을 이용해 가장 가능성 있는 두번째 단어로 $$w_2=indian$$를 고를 수 있다.
- 과정은 반복되고 모델은 art를 고르고 $$\circ$$를 고를 것이다.
- $$\circ$$ 즉, end-of-query symbol이 나오게 되면 user에게 *cleveland*을 유저에게 보여주게 된다.
 

# 4. Experiments

## 4.1 Dataset
- search log from AOL: 2006년 3월 1일 부터 5월 31일까지의 dataset. 657,426의 개별 유저들에게서 입력된 16,946,938 개의 쿼리들. 
- background data: 2006년 3월 1일부터 5월 1일 이전까지의 쿼리. 모델과 baseline들을 평가하기 위해 만들었다.
- training set: background data의 다음 두 주의 쿼리. ranking model을 튜닝하기 위해 만들었다.
- validation set, test set: 나머지 두 주의 쿼리. 

## 4.2 Model Training
- background data에서 가장 빈번한 9만 개의 단어를 vocabulary V로 뒀다.
- parameter 최적화는 mini-batch RMSPROP을 이용했다.
- gradient의 norm이 1을 넘어가지 않도록 normalizing했다.
- validation set의 likelihood가 5번이 지나도 오르지 않으면 학습을 그만시켰다.
- Theano 사용.(2015년이라서 그런 듯)
- query-level RNN dim: 1000, session-level RNN dim: 1500, output word embedding dim: 300

## 4.3 Learning to Rank
- context(anchor) $$Q_1,...,Q_{M-1}$$이 주어졌을 때 target 쿼리 $$Q_M$$을 예측하는 게 목표이다.
- 모델의 성능을 시험하기 위해 세 가지 다른 상황에서 다음 쿼리를 예측해봤다.
- 첫번째는 context 쿼리가 background data에 있을때, 두번째는 context 쿼리가 흔한 쿼리들과 매우 유사할 때, 세번째는 background data에 없을 때이다.
- 각각의 상황에서 20개의 후보를 골랐고, 후보한테는 true target labeling을 하고 나머지는 모두 관련없는 것으로 labeling을 했다.
- (2015년 당시)최신 랭킹 알고리즘인 LambdaMART를 supervised 랭커로 선택했다.

**Pairwise and Suggestion Features**
- 각각의 후보 suggestion에 대해, 얼마나 많이 context 쿼리 뒤에 왔나를 셌다. 그리고 이 카운트를 feature로 더했다.
- context 쿼리의 빈도도 사용했다.
- 또한 context와 suggestion 사이의 Levenshtein 거리도 feature에 더했다.
- suggestion feature는 suggestion의 길이와 background set에서의 빈도도 포함한다.
- LambdaMART는 Learning to Rank 알고리즘 중 하나이다. 아이템 리스트에서 랭킹 문제를 해결하는 알고리즘이다. 
- Levenshtein 거리는 두 문자열 간의 유사도의 측정 기준이 되는 거리이다. 원래 문자열 s와 타겟 문자열 t와의 거리는 s를 t로 바꾸는 데 필요한 삭제, 삽입, 대치의 횟수이다. ex) s="test", t="tent" => LD(s,t) = 1

**Contextual Features**
- suggestion과 context에서 가장 최근의 10개의 쿼리 사이의 character n-gram 유사도에 해당하는 10가지 feature를 더했다.(쿼리 하나당 1개씩 더한 듯)
- suggestion과 context내에서 각각의 쿼리 사이의 평균 Levenshtein 거리도 더했다.
- Query Variable Markov Model (QVMM)을 추가적인 feature로 사용해서 추정된 score를 사용했다.
- character n-gram similarity는 두 문자열 사이의 유사도를 n-gram을 이용해 알아내는 방법이다. 
- bi-gram일때, 
$$similarity(korea, korean) = \frac{2X|\{ko,or,re,ea\}|}{|\{ko,or,re,ea\}|+|\{ko,or,re,ea,an\}|}=\frac{2X4}{4+5}\approx 0.89$$
- markov model은 stochastic한 모델이다. 미래의 상태는 이전에 발생한 사건이 아니라 오직 현재의 상태에만 의존한다고 가정한다.

**HRED Score**
- 주어진 context에 대해서 suggestion의 log-likelihood에 해당하는 추가적인 feature를 추가하는데 사용됐다.


# Result
- HRED는 long-tail queries 즉, background set에서 본 적 없는 쿼리(빈도수가 많지 않은 쿼리)에 대해 많은 성능 향상을 보여줬다.


# Reference
- [Intuitive explanation of Learning to Rank (and RankNet, LambdaRank and LambdaMART)](https://medium.com/@nikhilbd/intuitive-explanation-of-learning-to-rank-and-ranknet-lambdarank-and-lambdamart-fe1e17fac418)
- [Levenshtein Distance, in Three Flavors](https://people.cs.pitt.edu/~kirk/cs1501/Pruhs/Spring2006/assignments/editdistance/Levenshtein%20Distance.htm)
- [Query Variable Markov Model (QVMM)](https://stackoverflow.com/questions/26037351/n-gram-character-based-similarity-measure)

<script id="dsq-count-scr" src="//nlp-with-koding.disqus.com/count.js" async></script>

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
