---
comments: true
title: "A Hierarchical Latent Variable Encoder Decoder Model for Generating Dialogues Review"
date: 2018-02-07 15:56
categories: 2016 RNN encoder-decoder generative-model 
---
# Abstract
- RNN의 hidden state는 해당 문장내에서의 token의 정보를 요약하는 데 집중하기 때문에, 오래 전의 문장들에 대한 정보는 잘 저장하지 못했다. 하지만 VHRED의 경우, 현재 time step에서의 context RNN의 output과 다음 sub-sequence에 대한 encoder RNN의 output을 input으로 받는 latent variable z를 이용했고, token에 대한 요약을 넘어 더 높은 수준의 정보를 저장할 수 있게 됐다. 


# 1. Introduction
- 근본적으로 RNN은 샘플링할 때, output을 다양하게 만들어내지 못했다. 즉, 대화 주제, 대화 목표, 화자의 발화 양식과 같은 특성을 반영하지 못했다.
- 다양한 레벨의 가변성을 가지는 generative한 과정을 만들어내는 architecture를 소개하겠다.


# 2. Technical Background

## 2.2 A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues
- hierarchical recurrent encoder-decoder model (HRED)는 RNN의 확장판이다.
- encoder-decoder 구조를 대화 생성까지 확장시킨 것이다.
- 해당 모델은 각각의 output sequence가 두 가지 level의 체계로 만들어질 수 있다고 가정한다. **즉, sequences of sub-sequences와 sub-sequences of tokens 두 가지 level로 나눠질 수 있다는 것이다.** 예를 들어, 대화는 발화들의 sequence(sub-sequence)로 만들어지고 각각의 발화는 단어들의 sequence로 만들어질 수 있다. 
- HRED는 세 개의 RNN 모듈 *encoder RNN, context RNN, decoder RNN*로 구성된다.
- 각각의 단어들의 sub-sequence는 encoder RNN에 의해 실수값을 가지는 vector로 인코딩된다. 문장이 RNN을 통해 인코딩된다는 의미이다.
- 이 인코딩된 값들은 context RNN에 input으로 주어진다. 여러 문장이 있을 때 context RNN은 인코딩된 여러 문장의 vector값을 input으로 받으며, 1~t번째까지의 문장의 모든 정보를 담기 위해 hidden state를 업데이트 하는 것이다. 
- context RNN은 실수 값을 가지는 vector를 output으로 출력한다.
- decoder RNN이 다음 문장(sub-sequence of tokens)을 만들어 낼 때, context RNN이 만들어낸 vector를 인풋으로 받는다.

## 2.3 A Deficient Generation Process
- RNN에 기반하는 구조는 의미있는 대화를 만들어내는 데 심각한 문제들을 가지고 있었다.
- 추측하건대 원인은 **output distribution(RNN의 결과)의 매개변수화**에 있다고 생각한다. output을 다시 input으로 사용하는 것이 문제이다. 가변성은 output distribution을 통해서만 만들어 지고 결국 생성 과정에 강한 제약을 가져오게 된다.
- 위와 같은 현상은 두 가지 관점에서 좋지 않다. 

*통계적 관점*
- low level에서만 넣게 되는 stochastic variation과 함께, 모델은 전체적인 구조를 보기 보다는 sequence내에서의 지엽적인 구조만을 보게 된다.
- 이렇게 되는 이유는 lower level에서 넣게되는 random variation이 바로 이전의 observation에 일치하도록 만들고, 더 오래 전의 observation이나 미래의 observation에는 영향을 덜 받게 된다.
- random variation은 종속적인 component에 더해지는 noise viable이라고 생각하면 되겠다.
- 만약 noise가 higher level of representation에 넣어진다면, 그 효과는 longer-term dependency에 상응한다. [long-term dependency](http://www.whydsp.org/280)는 output을 만들어 내는 데 중요한 정보가 현재의 time step에서 멀리 떨어져 있는 걸 말한다. 

*계산을 필요로 하는 학습 관점* 
- HRED의 decoder RNN의 hidden state $$h_m$$은 가장 그럴듯한 다음 token(단기 목표)을 만들어내기 위해 time step m까지의 모든 과거 정보를 요약해야한다.
- 또한 가장 그럴듯한 future token(장기 목표)을 만들어내기 위해 embedding 공간에 위치해야 한다.
- vanishing gradient effect때문에 더 단기적인 목표가 더 영향을 받는다.
- 이 장기와 단기라는 목표를 위한 타협점을 찾아내는 것은 결국 학습 과정에서 모델의 parameter들이 너무 지나치게 다음 token을 예측하는데만 집중하도록 할 것이다.
- 특히 복잡하고 긴 문장에서 모델은 단기 예측에 더 유리할 것이다. 왜냐면 $$h_m$$이 오직 다음 toke을 예측하도록 학습시키는 게 더 쉽기 때문이다. 긴 문장의 경우, hidden state가 담고 있어야 할 정보가 더 많아지게 되고, 결국 매 time step마다 매우 중구난방인 source에 방해받게 된다. (과유불급잼...)


# 3. Latent Variable Hierarchical Recurrent Encoder-Decoder (VHRED)
- VHRED는 decoder에서 latent variable을 이용해 HRED를 강화한 것이다.
- log-likelihood에서 [variational lower-bound](http://www.umiacs.umd.edu/~xyang35/files/understanding-variational-lower.pdf)를 최대화하는 것을 목표로 한다.
- 위와 같이 한다면, 두 가지 생성 과정(hidden state 샘플링, output sequence 생성)에서 계층적으로 구조화된 sequence를 만들 수 있게 된다.
- $$w_1,...,w_N$$을 N개의 sub-sequence들로 구성된 sequence라고 해보자. 여기서 $$w_n=(w_{n,1},...,w_{n,M_n}$$ 이고, n번째 sub-sequence이다. 그리고 $$w_{n,m} \in V$$이고 해당 sequence에서 m번째 token이다.
- VHRED 모델은 각각의 sub-sequence에 대한 stochastic latent variable $$z_n$$을 사용한다. 이때 sub-sequence들은 모든 이전의 token들에 영향을 받는다.
- $$z_n$$이 주어졌을 때, 모델은 n번째 sub-sequence tokens인 $$w_n=(w_{n,1},...,w_{n,M_n}$$을 만들어낸다. 밑은 이에 대한 식이다.

![Imgur](https://i.imgur.com/rESzH4J.png)

- $$\mathcal{N}(\mu,\sum)$$은 정규 분표를 다차원 공간에 대해 확장한 [다변수 정규 분포](https://ko.wikipedia.org/wiki/%EB%8B%A4%EB%B3%80%EC%88%98_%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC)이고 평균이 $$\mu$$이고 covariance matrix $$\sum$$는 diagonal matrix로 제약된다.

![Imgur](https://i.imgur.com/nutsbnM.png)

- VHRED 모델은 HRED 모델과 같은 세 개의 component를 포함한다. 
- encoder RNN은 하나의sub-sequence를 고정된 크기의 실수 벡터로 인코딩한다.
- context RNN은 encoder RNN의 output을 input으로 받고 모든 이전의sub-sequence들을 고정된 크기의 실수 벡터로 인코딩한다.
- 이 벡터는 tanh를 gating 함수로 가지고 두 개의 층으로된 feed-forward neural network에 feed된다.
- 행렬곱은 feed-forward network의 output에 적용된다. 이 때 output은 multivariate normal mean $$\mu_{prior}$$을 결정한다. 비슷하게 diagonal covariance matrix $$\sum_{prior}$$에 대해서는 output에 다른 행렬곱이 적용되고 양수를 보장하기 위해 (softplus)[https://sefiks.com/2017/08/11/softplus-as-a-neural-networks-activation-function/] 함수를 거치게 된다.
- 모델의 latent variable들은 variational lower-bound를 최대화하는 것을 통해 도출된다. 그리고 이때 각각의 sub-sequence에 대해 독립적인 term으로 인수분해한다.

![Imgur](https://i.imgur.com/rzhGYts.png)

- $$Q_\psi(z|w_1,...,w_M)$$은 사후 추정 분포(approximate posterior distribution)이다. 거의 변동인 없는 진짜 사후 분포를 추정하는 것을 목표로 한다.

![Imgur](https://i.imgur.com/DvCEXeA.png)

- $$\psi_{posterior}$$는 추정 사후 평균이고 $$\sum_{posterior}$$는 추정 사후 covariance matrix이다. 그리고 이전의sub-sequence와 현재 sub-sequence의 함수이다.
- $$\psi_{posterior}$$과 $$\sum_{posterior}$$는 prior에 의해 결정된다. feed-forward network의 output과 행렬곱을 하고 covariance에 softplus 함수가 적용된다.
- test때는 sample $$z_n$$은 이전의 sub-subsequences $$(w_1,...,w_{n-1})$$에 영향을 받게 되고 각각의 sub-sequence에 대한 prior $$\mathcal{N}(\mu_{prior}(w_1,...,w_{n-1}),\sum_{prior}(w_1,...,w_{n-1}))$$에서 얻어진다.
- 이 샘플은 context RNN의 output과 concatenate되고 decoder RNN에 input으로 주어지게 된다. 그리고 token 단위로 sub-sequence를 만들어낸다.
- training때는 sample$$z_n$$은 추정 사후 $$\mathcal{N}(\mu_{posterial}(w_1,...,w_{n-1}),\sum_{posterial}(w_1,...,w_{n-1}))$$에서 얻어진다. 그리고 (4)공식에 의해 주어지는 variational lower-bound의 gradient를 추정하는데 사용된다.
- 추정 사후는 자신의 one-layer feed-forward neural network에 의해 매개변수화된다. 이때 network는 현재 time step에서 context RNN의 output을 input으로 받고 다음 sub-sequence에 대한 encoder RNN의 output을 input으로 받는다. 
- output sequence의 가변성은 두 가지 방식으로 만들어진다.
> **sequence 수준**에서 z에 대한 conditional prior distribution
 
> **token 수준**에서 token들에 대한 conditional distribution  
- variable z는 sequence에 대한 high-level 정보를 표현해 model이 오랜 시간동안 이전까지의 output을 잘 기억할 수 있도록 해준다.
- hidden state $$h_m$$은 주로 M번째까지의 token들의 정보를 요약하는데 초점을 맞춘다.
- 직관적으로 variable z로 randomness를 집어 넣는 것은 문장의 topic이나 sentiment같은 더 높은 수준 판단에 해당한다.


# 4. Experimental Evaluation
*Twitter dialogue*
- model이 한개 혹은 더 많은 발화로 이루어진 대화를 받았을 때, 주어진 맥락에 적절한 다음 반응을 생성하는 게 목표이다.
- '->' token은 발화자의 변화를 나타낸다.

![Imgur](https://i.imgur.com/GVKxO9n.png)

- 위에서 보다싶이 VHRED가 가장 적절한 반응을 만들어내는 걸 알 수 있다.

<script id="dsq-count-scr" src="//nlp-with-koding.disqus.com/count.js" async></script>
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

