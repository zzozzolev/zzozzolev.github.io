---
comments: true
title: "[Stanford NLP] Lecture 5-Backpropagation"
date: 2017-09-04 
categories: 2017 stanford-nlp backprop
---
강의 출처:

https://www.youtube.com/watch?v=isPiE-DBagM&t=2043s&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=5



### 들어가기 전에

- 이번 강의에서는 backprop에 대해 4가지 방식으로 설명을 해줄 것이다. 
- backprop을 통달시키려는 socher 교수님의 친절함을 알 수 있다.

------

## Explanation #1 for backprop

![Imgur](https://i.imgur.com/DHt5MCj.png)

- 전의 강의와  모두 똑같지만, hidden layer하나를 더할 것이다.
- $x$는 첫번째 activation이자 첫번째 hidden layer이다. 그리고 윈도우, 즉 단어를 concatenate한 것이다. 
- $a^{(2)}$에서 sigmoid 함수같은 element-wise nonlinearity를 적용한다. 
- $z^{(3)}$는 $z^{(2)}$와 같은 아이디어이지만,  $z^{(2)}$와는 다른 차원을 갖는다.
- $U$는 column vector이다.  모든 x또한 column vector이다.
- 마지막 layer에 꼭 linear layer를 선택해야되는 건 아니다. 원한다면 sigmoid함수를 써도 상관없다.

![Imgur](https://i.imgur.com/YHIuPou.png)

- 전의 강의에서 했던 것처럼 $w^{(2)}$에 대해 미분을 해준다.

![Imgur](https://i.imgur.com/mXiTFsJ.png)

- $\delta$는 두 vector $U_i$와 $f\prime(z_i^{3})$의 곱이다. 그리고 해당 layer에서 오는 error 값이다.

- $\delta^{(3)}a^{(2)T}$는 외적이다.  외적을 하는 이유는 모든 i에 대한 쌍(pair)에 대해서 cross product를 주기 때문이다.  즉, 외적을 통해 $w_{ij}$에 있는 모든 정보를 이용하는 것이다.*

  *잘모르겠으면 [해당 링크](https://github.com/YBIGTA/Deep_learning/blob/master/RNN/nlp/%EC%9D%B4%EB%A1%A0/Natural%20Language%20Processing%20with%20Deep%20Learning/Lecture%204%20Word%20Window%20Classification%20and%20Neural%20Networks.ipynb)의 Training with backpropagation의 5번째 슬라이드 참조.

![Imgur](https://i.imgur.com/SaY39Yj.png)

- 2번째 layer에서의 update는 $w^{(2)T}\delta^{(3)}$이다. 지난번에 이미 배웠던 거다. 하지만 지난번에는 layer가 1개었기 때문에, 여기서 끝이었지만 지금은 layer를 한번 더 쌓았기 때문에 chain rule을 다시 적용할 필요가 있다.

![Imgur](https://i.imgur.com/pVkYv7U.png)

- vector의 미분이라 다소 복잡할 수 있지만 스칼라(차원이 없고 크기만 가지는 것)라고 생각하면 괜찮다고 한다...
- $\delta^{(2)}$는 이전 layer에서 구한 미분 값에 현재 layer의 미분 값을 element-wise 곱을 해준다.

![Imgur](https://i.imgur.com/dLIES7r.png)

- 빨간 네모 안에 있는 두 방정식을 이해하면, 이제 모든 multilayer neural networks에 대한 update의 끝판왕을 안 것이다. update 마스터가 될 수 있다.
- 각각의 $W$에 대한 마지막 update는 항상 오른쪽 위에 나와있는 외적이 될 것이다. 전의 강의에서 나온 것과 살짝 다른 형태를 띄고 있는데, 단지 regularization을 더해줬다. 해당 layer의 activation에 $\delta$배를 해주고 정규화를 해주는 것이다.

## Explanation #2 for backprop: Circuits

- neural network에서 나오는 많은 matrix때문에 골머리가 날 뻔했다. 대신 이번 설명에서는 그냥 간단한 함수로 생각해 볼 것이다. :smiley:

![Imgur](https://i.imgur.com/2YK9sJl.png)

- 즉, network를 이동하면서 반복적으로 error signal혹은 local gradient들을 적용, 계산할 것이다.
- f는 lost function이고 x,y,x는 parameter들이다.
- 예를 들어, x=-2, y=5, z=-4라고 해보자. q와 f에 대해 편미분을 해보면 왼쪽과 같은 결과를 얻을 수 있다.
- 우리는 이 변수들을 update하기 위해서 해당 변수들의 미분 값을 알아야 한다.
- 오른쪽 그림을 봐보자. 빨간색으로 쓰인 숫자가 미분 값이다. 변수들의 미분 값을 알기 위해서는 오른쪽에서부터 시작해야한다. f-12를 f로 미분하면 1이 나온다.
- z벡터를 update하기 위한 미분값은 $\partial f\over\partial z$이다.  이 값은 손쉽게 구할 수 있다. 파란 박스에서 정의했듯이 이 값은 $q$이고, $q$는 $x+y$의 값인 3이다.
- $\partial f\over\partial q$는 위와 비슷하 방식으로 파란 박스에서 나와있듯이 z이다. 하지만 주의할 것은 chain rule을 적용했기 때문에, higher node에서 온 미분 값을 곱해줘야 한다. 첫번째 설명에서 언급했던 것처럼 위에서 온 error signal을 전해받는 거라고 생각하면 된다. 그렇지만 여기서는 이전 미분값이 1이기 때문에 그냥 -4라고해도 상관없다.
- $\partial f\over\partial y$는 chain rule을 적용해, 이전 node의 미분값인 -4와 local gradient, 해당 node에서의 미분값인 $\partial q\over\partial y$, 1을 곱해주면 된다. 나머지 미분값인 $\partial f\over\partial x$도 이와 마찬가지로 구해주면 된다.

![Imgur](https://i.imgur.com/BKxi5Tn.png)

- 위의 그림에서 보듯이 각 node에 chain rule을 적용해, 우리가 최종적으로 구하고자 하는 변수의 미분 값, 즉 update값을 알아낼 수 있다. 
- 여기서 중요한 것은 local gradient는 forward propagation때 구할 수 있다는 사실이다. 이 때 구한 값을 저장해뒀다가 back prop때 이용하면 된다.

![Imgur](https://i.imgur.com/GHzH3Ms.png)

![Imgur](https://i.imgur.com/7Goo5Bv.png)

(초록색 숫자는 forward prop이고, 빨간색 숫자는 backprop이다. 앞에 미분과정이 더 나와있는 슬라이드도 있지만 생략하겠다. 위의 예시하나로 어떻게 하는지 감이 올것이다.)

- $f(w,x)$는 sigmoid 함수이다. 여기서 $x$는 input이고 $w$는 weights이다. 이때 목표는 모든 elements, w와 x에 대한 편미분을 계산하는 것이다.
- $x$를 2-dimension으로 $w$를 3-dimension으로 가정하자. $w_2$는 bias term이다.
- $f(w,x)$의 값은 parameter가 뭐든지 간에 분자를 1로 취할 것이다.
- 그래프의 밑에 나와있는 식을 이용하면 , 위의 슬라이드에서처럼 미분 값을 구할 수 있다. 제일 첫번째 미분 값이 1인 이유는 $\partial f\over\partial f$를 하기 때문이다.
- higher layers에서 온 error signal을 계속해서 곱해나감으로써 미분 값을 다시 쓴다. 나중에는 변수의 미분 값인 local gradient와 위에서 부터 내려온 error signal을 곱함으로써 변수를 update할 수 있게 된다.

![Imgur](https://i.imgur.com/RWsZtOK.png)

- 위의 슬라이드처럼 함수의 부분을 일일이 쓰는 건 매우 번거롭다. 그래서 위처럼 나타낸다. $\sigma(x)$로 sigmoid 함수를 나타낸다.
- 위 슬라이처럼 일일이 계산하지 않아도, sigmoid 함수를 $x$에 대해 미분하면 $(1-\sigma(x))\sigma(x)$ 가 나온다.
- 스탠포드 학생이 여기서 질문을 하는데, 그럼 forward prop은 뭐냐고 한다. forward prop은 그냥 전반적인 함수 값을 계산하는 것이다. 그리고 test time에 이뤄지는 과정이다.

## Explanation #3 for backprop: The high-level flow graph

![Imgur](https://i.imgur.com/vz7YhFE.png)

- $x$에서 시작해 어떤 값을 계산하기 위해, 중간 변수 y를 거쳐 forward prop을 할 것이다. 그리고 backprop에서는 forward prop과는 반대 방향으로 gradient를 계산할 것이다.

![Imgur](https://i.imgur.com/jzdZbf9.png)

- 위의 슬라이스에서와 달리, $y1$과 $y2$에서 온 error signal들을 더해야한다.

![Imgur](https://i.imgur.com/lTctzuG.png)

- 일반적으로 $x$가 flow graph에서 multiple한 element들을 거친다면, 위같이 sigma를 써서 편미분 값을 더해주면 된다.

![Imgur](https://i.imgur.com/4teFjwM.png)

- 각각의 node는 계산 결과이고, 각각의 화살표는 계산이 어떻게 이루어지는지 보여준다. 화살표로 이어진 노드는 미분 값을 계산하기 위해 서로의 미분 값을 필요로 한다.
- 좀 더 복잡한 것도 정의할 수 있는데, 그림의 왼쪽을 보면 한 layer를 뛰어넘는 화살표가 보일 것이다. 이런 방식을 short circuit connections라고 부른다.

![Imgur](https://i.imgur.com/ss3wZ78.png)

(검은 화살표가 forward prop, 분홍 화살표가 back prop)

- $x$는 input이고 $y$는 class이다. 

- forward prop에서 sigmoid neural layer를 거칠 것이다. $h$는 $Vx$의 sigma이다.

- 다음 layer로 이동할 것이고, 마지막쯤에 softmax layer를 만날 것이다.

- 그 다음에 negative log likelihood를 거쳐 x와 y의 pair에 대한 cost function을 계산할 것이다.

- 그리고 forward를 했으니, parameter들을 update하기위해 back prop을 한다.

  > 하지만 굿뉴스는 일일이 계산할 필요가 없다 ㅎㅎㅎ 이미 좋은 패키지들이 나와서 알아서 다해주기 때문이다. socher 교수님이 박사 시작하실때만 해도 이런게 없었다고 한다...

![Imgur](https://i.imgur.com/KBvBCVi.png)

- gradient의 계산은 알아서 도출된다.

## Explanation #4 for backprop: The delta error signals in real neural nets

![Imgur](https://i.imgur.com/OTTY4dh.png)

- 첫번째 설명에서 나오던 복잡한 함수 구성을 좀 더 간편하게 표현했다.
- $\delta(3)$는 score에서 오는 error signal이다.
- 이때, $W(2), W(1)$모두 update하고 싶다.
- linear score을 지나갈 때 delta는 변하지 않는다.
- $W(2)$에 대한 update는 그냥 $\delta^{(3)}a^{(2)T}$의 외적값이다.

![Imgur](https://i.imgur.com/7LFL4Xi.png)

- matrix vector product 대한 간단한 affine 변환*을 거칠때,  그냥 forward prop matrix를 transpose한 것만 있으면 된다.  

  *[affine transformation 설명](http://darkpgmr.tistory.com/79)

- output에서의 dimension은 n행 1열이다. 이 vector에 $\delta$를 곱한다. 그러면 output의 dimension과 같아진다.

- $W^{(2)T}\delta^{(3)}$는 $W$와 같은 dimension을 갖는다.

![Imgur](https://i.imgur.com/YY7lpyt.png)

- $\sigma$는 element wise nonlinearity를 특성으로 갖는다. (여기서는 sigmoid)

- 그래서 다음 $\delta$를 update할 때, 즉, error vector(error signal)을 point-wise* nonlinearity을 통과할 때, non-linearity의 local gradient와 point-wise 곱셈을 적용해야 한다.

  *element wise

- 이 과정을 거쳐 $W(1)$에 도달하는 $\delta(2)$를 얻었다.

![Imgur](https://i.imgur.com/ex8yHNS.png)

- 이제 $W(1)$에 대한 마지막 gradient를 계산할 수 있게 되었다.

<script id="dsq-count-scr" src="//nlp-with-koding.disqus.com/count.js" async></script>
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
