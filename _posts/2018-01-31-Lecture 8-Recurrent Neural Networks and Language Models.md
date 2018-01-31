---
comments: true
title: "[Stanford NLP] Lecture 8-Recurrent Neural Networks and Language Models"
date: 2017-10-14 
categories: 2017 stanford-nlp RNN Language-model
---
강의 출처:

https://www.youtube.com/watch?v=Keqep_PKrY8&index=8&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6

![Imgur](https://i.imgur.com/uq7ETOp.png)

- 일반적으로, 단어의 전체 sequence의 확률을 계산하는 것을 목표로 한다.
- 왜냐하면 문장에서 단어의 순서를 고려해야 되기 때문이다. 즉, 남은 sequence중에 어떤 단어가 와야 가장 적절한지 알아내야 한다.

![Imgur](https://i.imgur.com/aaa86cd.png)

- 하지만 다음 단어를 예측할 때, 모든 sequence를 고려하기 힘들기 때문에 이전 n개의 단어로만 제한시킨다.
- 잘못된 가정이지만 Markov chain을 사용하기 위해서는 어쩔 수 없다.
- $w1$부터 $w_m$까지의 sequence가 있을 확률은$\prod_{i=1}^mP(w_i|w_1, ..., w_{i-1})$이다. 이는 처음부터 바로 이전 단어까지를 모두 고려한 product이다. 그렇지만 machine learning에서는 계산을 하기가 힘들다. 그래서 n번째 전까지의 각각의 단어들을 통해 추정한다. 
- 이런 방법은 단지 단어들의 빈도수에만 기반을 둔 거다. 그래서 확률을 예측하기 위해서,  unigrams와 bigrams을 이용한다. Unigrams는 1개의 단어를 고려한 것이고 bigrams는 2개의단어를 고려한 것이다. 예를 들면, 'Dog is cute'라는 문장이 있을 때 unigrams는 'dog', 'is', 'cute'로 쪼개는 거고 bigram은 'dog is', 'is cute'와 같이 쪼개는 거다. 3개의 단어를 고려하면 trigrams가 된다.
- 첫 번째 단어가 주어졌을 때, 두 번째 단어가 나타날 확률을 예측하기 위해 두 단어가 이러한 순서로 얼마나 나타났는지 빈도수를 센뒤, 이를 전체 코퍼스에서 첫 번째 단어가 나타난 빈도수로 나눠준다. 만약 한 개의 단어만이 아니라, 이전 두 개의 단어를 고려하고 싶다면 빈도의 대상만 바꿔서 하면 된다.
- 생각해보면 위와 같은 방식처럼 하기 위해서는 엄청나게 많은 숫자를 저장하고 역시 무진장 많은 확률을 계산해야 한다.

![Imgur](https://i.imgur.com/fhs5CPT.png)

- performance가 counts가 늘어날 때마다 좋아지는 데, 그래서 좋은 성능을 내기 위해서는 겁나 많은 RAM을 필요로 한다. 비효율적으로 RAM을 쓰게 된다.

![Imgur](https://i.imgur.com/X8gKyMP.png)

- 기존의 neural network와 비슷하지만, 다른 점은 다른 time step에 있는 weight를 묶는 것이다. 즉, 같은 선형 layer와 비선형 layer를 다시 사용한다.
- 여기서 RAM은 단어의 수를 측정하는데만 쓰인다.
- $h_t$는 $h_{t-1}$의 영향을 받고, 그 다음에 $h_{t+1}$을 계산한다.

![Imgur](https://i.imgur.com/WrbOr3G.png)

($W^{(hh)}$인 이유는 $h_{t-1}$을 input으로 받아서 h를 계산하기 때문이고, $W^{(hx)}$인 이유는 x를 같은 vector space에 mapping 시키는 것을 표현하기 때문이다. S는 softmax 이다. )

- word vector들이 변하지 않는다고 가정하자.

- time step의 $h_t$에서 반드시 두 개의 matrix를 갖는데, 그 두 개를 더한다. 이는 $h_{t-1}$과 t에서의 word vector를 concatenate하는 것과 같다.* 그리고 그 다음에 sigmoid같은 함수에 input으로 넣어준다.

  *벡터를 더하지 않고 concatenate하는 이유는 matrix 곱을 할 때 차원을 맞춰주기 위해서이다.

- 모든 단어들에 대한 확률을 구하기 위해 $h_t$를 feature vector로 사용해, 즉 softmax classification layer의 input으로 사용한다. 

- $\hat{P}(x_{t+1}=v_j|x_t, ...,x_1)$은 다음 단어에 대한 확률이다.  다음 단어인 j번째 단어의 vector는 모든 이전 단어로부터 영향을 받는다. $\hat{y}_{t,j}$는 $\hat{y}$의 j번째 element이다. 즉, j번째 column vector이다.

![Imgur](https://i.imgur.com/JTjgJMb.png)

(V=단어 크기, $D_h$=hidden state 크기)

- $h_0$는 막 시작했을때의 hidden state이다. 
- $x_[t]$는 embedding matrix의 t번째 column vector이다.
- $W^{(S)}$의 dimension이 단어수Xhidden state 크기이기 때문에 $\hat{y}_t$는 단어 개수 만큼의 길이를 갖는다.

![Imgur](https://i.imgur.com/WobMPjz.png)

- 단어에 대한 확률 분포를 가지고 전에 했던 것과 마찬가지로 cross entropy loss function을 사용할 것이다.

![Imgur](https://i.imgur.com/mxrBlzL.png)

- 얼마나 잘했는지에 대한 평가는 전체 데이터에 대한 평균 log probability의 음수로 한다.
- 위의 수식이 잘 이해가 안 되면 word embedding을 떠올려보면 이해가 될 것이다. 
- 하지만 좀 더 많이 쓰이는 건 perplexity라는 평가 방법이다. 그냥 위에서 계산한 값을 2의 지수 함수의 input으로 넣어주면 된다. 

![Imgur](https://i.imgur.com/0DXdtJw.png)

- 각각의 time step에서 같은 matrix를 곱한다. 특정한 패턴을 늘리는 거라고 생각하면 된다.

![Imgur](https://i.imgur.com/Ws2oG2e.png)

- foreprop에서 그랬던 것처럼, backprop에서도 비슷하게 한다. delta, error signal, gradient의 global element또한 갖고 있다. 이러한 것들은 각각의 time step에서 network를 타고 뒤쪽으로 흘러갈 것이다.
- 해당 output에 가까우면 가까울수록 parameter들은 update가 잘 된다. 하지만, 뒤로가면 갈수록 signal이 너무 약해지거나 강해진다.

![Imgur](https://i.imgur.com/t0yRscr.png)

(softmax가 없다고 가정한다. $y_t$는 softmax output이거나 정규화되지 않은 score이다.)

- 바로 그런 문제를 **vanishing gradient problem**이라고 한다. t번째 time step에서의 error signal을 매우 뒤쪽으려 보내려고 할 때, 발생한다.
- 전체 error는 각각의 time step에서의 error의 총합이다.
- time step t에서의 미분 값은 모든 이전 time step의 값으로부터 영향을 받는다.

![Imgur](https://i.imgur.com/dJ48Idm.png)

- 미분 값을 구하기 위해 chain rule을 적용하다 보면, 결국 각각의 미분값은 jacobian*이 된다. 즉, 모든 아웃풋에 대한 편미분 값을 element로 하는 jacobian matrix를 얻게 된다.

  *[jacobian matrix](https://ko.wikipedia.org/wiki/%EC%95%BC%EC%BD%94%EB%B9%84_%ED%96%89%EB%A0%AC)

![Imgur](https://i.imgur.com/JS2yDTA.png)

- j번째 hidden state를 이전 hidden state로 미분한 값의 norm은 w의 upper bound*와 h의 upper bound보다 작거나 크다.

  *upper bound는 찾고자 하는 값보다 큰 값이 처음으로 나타나는 위치이다.

- 정확히는 모르겠지만, 위의 식 때문에 sequence가 길어질수록 그래서 t가 커질수록 beta값에 점점 의존하게 된다. 그리고 $(\beta_W\beta_h)^{t-k}$는 수렴하거나 발산한다.

- 결국 t에서의 gradient값은 너무 작아지거나 커지게 된다.

![Imgur](https://i.imgur.com/idE46mB.png)

- vanishing gradient는 매우 이전에 나온 단어가 다음 단어를 예측하는데 영향을 주지 못하게 만든다.

![Imgur](https://i.imgur.com/2qk6EpX.png)

- sequence가 중요한 language modeling에서 이것 때문에 문제를 겪는다.
- 예시를 보면, 빈칸에 들어갈 말은 John이라는 것을 쉽게 알 수 있다. 그래서 모델은 John에 해당하는 word vector에 높은 확률을 줘야하지만 실제로 그렇게 하지 못한다.

![Imgur](https://i.imgur.com/4hM9P98.png)

- 간단하게 vanishing gradient problem을 두 개의 neural network를 통해 살펴보자.
- error가 network를 거칠수록 gradient의 norm은 점점 작아지게 된다.

(화면이 갑자기 누래진다.)

![Imgur](https://i.imgur.com/xKatjx2.png)

- y축은 gradient의 크기이고, 파란색 그래프는 첫번째 sigmoid layer이고 초록색 그래프는 두번째 sigmoid layer이다. 초록색은 softmax classifier에 가깝다.
- 두번째 layer에서의 magnitude가 첫번째 보다 크다. 이런걸 100번 했다고 생각해보면 1번째 layer의 magnitude는 거의 0인 된다고 유추해 볼 수 있다. 두 번만 해도 0의 절반에 가까우니 말이다.

![Imgur](https://i.imgur.com/FZ5iG1t.png)

- 비슷하게 ReLu를 이용하면 sigmoid보다는 괜찮지만, 역시  vanishing gradient문제를 겪는다.

![Imgur](https://i.imgur.com/eklfn65.png)

- exploding gradient에 대해서는 hack을 쓸 수 있다.
- 간단하게 말하자면 특정 값보다 커지기 전에 gradient값을 자르는 방법이다.

![Imgur](https://i.imgur.com/tjPyrJS.png)

- 하나의 hidden unit을 가지고 rnn을 했을 때, error space를 시각화 한 것이다.
- model이 해결해야 되는 문제는 대략 value를 기억하고 있다가 50 time step이 지났을 때 반환하는 것처럼 매우 간단한 문제이다.
- 만약 gradient를 고정된 크기로 만든다면, 계곡(?)같이 생긴 저 곳을 뛰어넘을 수 없다. error를 낮게 만드는 공간에만 있게 된다.

![Imgur](https://i.imgur.com/BDKzLvb.png)

- vanishing gradient에 대한 간단한 해결 방법은 W를 identity matrix로 초기화하고 activation function으로 ReLU를 사용하는 것이다.
- 일반적으로는 W를 초기화할 때 랜덤한 값을 이용하는데, 의미하는 바는 x가 뭐든지 간에 hidden state에 랜덤하게 projection을 하는 거다.
- 하지만 Identity matrix를 이용해 초기화를 하면,  hidden state와 word vector를 섞는 게 된다.
- 즉, 처음에 뭐가 뭔지 잘 모르니 random하게 projection을 하지 말고, hidden state가 단지 word vector의 평균을 거쳐가게 만든다.
- 그래프는 random 초기화를 하는 것보다 identity 초기화를 할 때, 정확도가 증가한다는 것을 보여준다.

![Imgur](https://i.imgur.com/pntha9P.png)

- vanishing gradient말고도 만나게 되는 다른 문제는 softmax가  매우 커진다는 점이다. word vector와 softmax는 별개의 parameter이다. 만약 1000개의 hidden state를 가지고 100000개의 각각 다른 단어들을 가진다고 할 때, 모든 time step에서 hidden state와 곱해야 할 100000*1000의 dimension matrix를 가지게 된다.
- class-based word prediction에서 이러한 문제점을 해결하기 위한 한 가지 방법은 빈도수로 단어를 sort하는 것이다. 즉, 빈도수가 많을 수록 이전 class로 취급한다. 가장 많이 나온 건 1번 class이고 그 다음으로 많이 나온건 2번 class인 식이다.
- class가 많을 수록 perplexity는 좋아지지만, 그 만큼 계산해야 될 양이 많아지니 속도가 느려지게 된다.

![Imgur](https://i.imgur.com/wpKNb0a.png)

- RNN을 다양하게 이용할 수 있다.

![Imgur](https://i.imgur.com/l5mxuCC.png)

- opinion mining은 각각의 단어를 DSEs와 ESEs로 구분하는 걸 목표로 한다. DSE는 opinion을 직접적으로 드러내는 것이고 ESE는 화자의 태도를 단어 선택을 통해 드러내는 것이다. 쉽게 말하자면 화자의 생각과 감정을 직접적으로 드러내는지 그렇지 않은 지를 판단하는 거라고 생각하면 된다. 
- 예를 들면 "This is viewed as the main impediment"라는 문장이 있을 때, viewed는 DSE이고 main impediement는 ESE이다. 

![Imgur](https://i.imgur.com/8fFWIF3.png)

- bidirectional rnn은 단지 왼쪽에서 오른쪽으로만 가는 게 아니라, 오른쪽에서 왼쪽으로 가기도 한다.
- 두 $y_t$를 얻기 위해 두 가지 방향에서의 hidden state를 concatenation한다.
- 양방향에서 오는, 즉 이전 단어와 이후 단어를 이용해 특정 time step에서의 단어를 예측하므로 더 정확하게 단어를 예측할 수 있다.

![Imgur](https://i.imgur.com/diPT3CX.png)

- bidirectional하게 할 뿐만 아니라, 거기다 deep을 더할 수도 있다.
- t번째 hidden state를 계산할 때 t-1번째의 hidden state를 고려할 뿐만 아니라 밑에 있는 hidden layer의 t번째 hidden state역시 고려한다. 

<script id="dsq-count-scr" src="//nlp-with-koding.disqus.com/count.js" async></script>
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
