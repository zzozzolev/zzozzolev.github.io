---
comments: true
title: "[Stanford NLP] Lecture 6-Dependency Parsing"
date: 2017-09-23 
categories: 2017 stanford-nlp GloVe word-representation 
---
강의 출처: https://www.youtube.com/watch?v=PVShkZgXznc&index=6&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6

![Imgur](https://i.imgur.com/F00xB8K.png)

- 위의 구조는 문장이 어떻게 만들어지는지를 보여준다. 아이디어는 문장이든 문장의 부분이든 사람들이 문장 구조를 만드는 방식에는 regular way가 있다는 것이다.
- 위의 명사구의 관사와 명사 사이에 형용사를 넣어서 "large dog" 등등 구를 더 크게 만들 수도 있고, 명사 뒤에 전치사구를 넣어서도 역시 가능하다.
- 언어학자들과 NLP를 연구하는 사람들은 인간 언어의 구조를 묘사하길 원했다. 그리고 이러한 일을 해낼 수 있는 두 가지 핵심적인 도구가 있다.
- 한 가지 방법은 **context free grammars**이다. 이건 컴퓨터 과학쪽 영어이고 언어학자들은 **phrase structure grammars**라고 부른다.
- free라는 말에서 느낌이 오겠지만 문장 구조는 언어마다 조금씩 다르나 고정되어 있지 않다. 위처럼 같은 명사구라고 해도 determiner(a, the, some...)다음에 명사가 올 수도 있고, 'determiner 명사 전치사구' 이런식으로 올 수도 있다. 그리고 여러개의 형용사와 전치사구를 넣어서 더 복잡하게 만들 수 있다.

![Imgur](https://i.imgur.com/1z00OGh.png)

- 위의 방식처럼 형용사, 명사, 전치사구가 언제 어떻게 온다라는 식으로  문장 구조의 계층을 분석하는 게 아니라 어떤 단어가 어떤 단어에 의존하는지를 보는 걸 **dependency structure**라고 한다.
- 어떤 단어가 다른 단어를 수식하거나 또 다른 단어의 일부분인  것을 해당 단어에 **의존**한다고 한다.
- 예를 들어 barking dog라는 명사구가 있을 때 barking은 dog를 꾸며주기 때문에 dog에 의존한다. 
- 의존 관계를 화살표로 표시한다. 화살표의 시작점에 있는 게 의존 받는 단어이고 화살표의 끝점에 있는게 의존 하는 단어이다. 좀 이해하기 쉽게 비유하자면, '너는 나한테 의존해'를 화살표로 표시한 거라고 생각하면 된다.

![Imgur](https://i.imgur.com/azTr0tm.png)

- 하지만 이때 ambiguity(모호함)이 발생한다. 하나의 문장에서 여러 의존 관계를 생각할 수 있다. 
- 연구하는 과학자이기 때문에, 과학자는 주어이자 study의 argument이다. whales는 연구되기 때문에 역시 study의 목적어이자 argument이다. 여기까지는 두 가지 가능성에서 큰 차이가 없다.
- 그러나 전치사 뒤에 오는 명사인 space때문에 문제가 발생한다. space에서 연구를 하는 걸까 아니면 whales가 space에서 온 것일까? 즉, space는 study를 수식하는 걸까 아니면 whales를 수식하는 걸까?
- 결국 의존의 모호함은 단어 간의 수식관계에서 발생한다.

![Imgur](https://i.imgur.com/hOCX3xa.png)

- 문장의 구조를 보면 주어 동사 목적어 순으로 되어있다. 목적어 뒤에는 모두 전치사구이다.
- 그렇다면 각각의 전치사구가 수식하고 있는 건 뭘까? 'by Royal Trustco'는 바로 앞에 있는 'its acquisition'을 수식하고 'of Toronto'역시 바로 앞에 있는 걸 수식한다. 그래서 단순히 전치사구는 바로 앞에 있는 걸 수식한다고 생각할 수 있다. 하지만...'for $27 a share'는 뭐를 수식할까? 바로 앞에 있는 'of  Toronto'가 아니라 'acquisition'을 수식한다. 그리고 'at its monthly meeting'는 'approval'이 일어난 때이므로 'approved'를 수식한다.
- 즉 전치사구는 명사구를 수식할 수도 동사를 수식할 수도 있다.
- 어떤 단어가 어떤 단어를 수식하는지를 정할 때 완전히 자유롭게 선택하는 건 아니다. nesting(얽혀있는) 제약이 있기 때문이다.  'for $27 a share'가 acquisition을 수식하면, 일반적으로 다음 전치사구인 'at its monthly meeting'는 acquisition을 수식하든 그 앞에 있는 'approved'를 수식한다. 물론 예외도 있기는 있다.
- nesting relationship을 따라가다 보면,  문장에서 전치사구의 개수를 기준으로 exponential한 개수의 모호함을 가질 것이다. 쉽게 말하면, 전치사 구가 많으면 많을 수록 수식 관계에서의 모호함이 커진다는 거다. 그래서 exponential series인 Catalan numbers만큼 복잡해진다.

![Imgur](https://i.imgur.com/E16E7Jl.png)

- treebanks에서는  더 이상 이전처럼 grammar rule을 쓰지 않아도 된다. 그냥 '문장 내놔. 다이어그램 쓱싹. 구조 보여줌!'이 되기 때문이다.

![Imgur](https://i.imgur.com/x72fane.png)

- 솔직히 grammar를 이용하는 것보다 느리고 비효율적인 것처럼 보일 수있다. 하지만 machine learning 관점에서 보면 make sense하다. 알고보니 treebank는 슈퍼 짱짱이었다.
- 좋은 점은 위에 쓰인대로 이다. (읽어보면 알 거 같아서 넘어가겠다:happy:)

![Imgur](https://i.imgur.com/yW3MDPe.png)

- dependency syntax의 아이디어는 lexical(사전적인) items 사이에서 관계를 찾아내는 것이다.

- 그 관계는 binary하고 asymmetric(불균형적인) 관계이다.

- 관계를 나타내기 위한 화살표를 그리는데, 화살표를 dependency라고 한다. 화살표는 subject, prepositional object apposition등 문법적 관계로 분류된다.

  *dependency grammar가 궁금한 사람은 [이 논문](http://ufal.mff.cuni.cz/~hladka/2015/docs/Stanford-dependencies-manual.pdf)을 참고하기 바란다.

- 화살표의 시작에 있는 단어를 head라고 하고 끝에 있는 단어를 dependent라고 한다. 위의 사진에서 보면 'Bills'는 dependent이고 'submitted' head이다. 그래서 'Bills'는 'submitted'의 argument이다. 

- 만약 'by Senator Brownback, Republican of Texas'와 같은 구가 있을 때, 하나의 head 'Brownback'을 가지고 밑에 그 외의 단어를 가진다.

- dependency grammar의 주요한 부분은 어떤 단어가 head이고 어떤 단어들이 dependents인지를 정하는 것이다.

- 해당 다이어그램과 이후에 나오는 것들은 **universal dependencies**에 의한 분석이다. universal dependencies는 새로운 tree banking이다. 무수히 많은 언어들에 잘 들어맞는 공통적인 dependency representation을 찾기 위해 노력한 결과물이라고 한다.

- universal dependencies에서는 전치사를 다른 grammar와 다르게 다룬다. 일반적으로 전치사는 목적어를 갖는다. 의존 관계로 따지자면 전치사는 head이고 목적어는 dependent가 된다. 하지만 universal dependencies에서는 전치사가 어떤 dependent도 갖지 않는다. 전치사는 case marker의 역할만 할 뿐이다. case marker는 문장에서 명사의 무슨 역할을 하고 있는지 보여준다.

![Imgur](https://i.imgur.com/YO5b4b3.png)

- 사람들이 화살표를 그리는 방식은 고정적이지 않다. 위의 사진에서는 head->dependent였지만 dependent->head일수도 있다. 
- 보통 fake인 ROOT를 추가해 모든 단어가 다른 단어에 의존하게 만든다. 이렇게 하면 수식에서 더 쉬워진다고 한다.
- 모든 문장이 ROOT로 시작하고 어떤 단어는 ROOT의 dependent이다. 이때, 어떤 단어는 실제 문장내의 어떤 단어에도 의존하지 않는 전체 문장의 head이다.

![Imgur](https://i.imgur.com/Pxs23LN.png)

- 여기서 생각해봐야 될 것은 어떤 단어가 어떤 단어에 의존하는지를 어떻게 정하냐는 거다.
- 이전에 했던 distributed word representation을 통해 단어간의 관계를 보고 비슷한 단어끼리 dependency를  정할 수도 있을 것이다.  discussion과 issue는 비슷한 단어이다.
- dependency distance를 이용할 수도 있는데, 보통 의존 관계에 있는 단어들은 그렇게 멀리 떨어져 있지 않기 때문이다. 옹기종기 모여있다. 
- Intervening material은 의존 관계의 개입하지 않는 성질을 이용한 것이다. 만약 명사구 내에서 '형용사 명사' 처럼 의존 관계가 있다고 해보자. 그렇다면 형용사는 동사를 지나서 명사를 수식하지는 않는다. 대다수의 경우 구두점을 넘어서 의존 관계에 있지는 않는다.
- 마지막은 head의 필수 문장 성분을 고려하는 것이다.  예를 들면 give는 주어, 직접 목적어, 간접 목적을 필요로 하니 최소 3개의 dependent를 가진다. 이렇게 의존 관계의 수와 규모를 생각할 수 있게 된다.
- 결국 단어마다, 품사마다 각각 다른 패턴의 dependence를 갖는다.

![Imgur](https://i.imgur.com/Wnpxqs5.png)

- 문장은 어떤 단어가 어떤 단어(ROOT 포함)에 의존하냐를 결정함에 따라 parsing된다. 

- 의존 관계를 결정하는데는 몇 가지 제약 조건이 있다. 오직 하나의 단어만 ROOT의 dependent가 되는 것이다. 그리고 A가 B에 의존하고, B가 A에 의존하는 등 순환적인 관계는 허용하지 않는다.

- 위의 문장은 좋은 nesting structure를 갖는다. 하지만, 빨간 화살표를 보면 어쩔 수 없이 서로를 가로지르게 된다. 

- 대부분의 dependency 관계에서 상대적으로 linear order를 갖는다.* 그리고 만약 dependency tree가 fully nesting하다면, **projective dependency tree**라고 부른다. 평면에 표현할 수 있고 nesting relationship을 갖는다. 하지만, nesting하지 않고 crossing한 구조를 가질 수도 있다.

  *dependency를 그려보면 문장 순서가 어느 정도 투사된다고 생각하면 될 것 같다.

- 위의 문장을 보면, 긴 명사 수식어를 문장의 끝으로 옮길 수 있다. 문장에서의 위치가 바뀌더라도 'bootstrapping'은 여전히 talk를 수식한다. 결국 위치를 바꿈으로써 crossing line이 발생하게 된다. 이런 것을 **non-projective dependency tree**라고 한다. 즉, head에서 dependent로 화살표를 연결 할 때 다른 head와 dependent를 연결하는 화살표를 가로지를 수 밖에 없을 때이다.

  *잘 이해가 안 될 수 있는데 [이 사이트](https://en.wikipedia.org/wiki/Dependency_grammar#Linear_order_and_discontinuities)의 그림을 보면 좀 이해가 될 것이다.

  ![Imgur](https://i.imgur.com/cezd26W.png)

  - 우리가 parsing할 문장이 "I ate fish"라고 해보자.
  - 카키색으로 표현 된stack이 있다. stack에 문장을 넣어서 parsing을 시작한다. 오른쪽이 stack의 top이다.
  - 노란색으로 표현 된 건 buffer이다. buffer는 다뤄야할 sentence이다. buffer의 top은 왼쪽으로 여긴다. 
  -  *Shift, Left-Arc, Right-Arc* 세 가지 operation이 있다.
  - shift는 쉽다. 그냥 buffer의 top에 있는 단어를 stack의 top에 넘기는 operation이다.

  ![ Imgur](https://i.imgur.com/snZr42p.png)

  - left-arc는 stack의 top에서 두 번째, 즉 오른쪽기준으로 2번째에 있는 게 stack의 top의 dependent라는 걸 표시한다. <- 이렇게 하는 거라고 보면 되겠다. 그래서 위의 예시에서 I는 ate의 dependent이다. 그 후에 top에서 두 번째에 있는 걸 stack에서 없앤다. 이러한 과정을 해주는 게 left-arc이다.
  - buffer에 아직 단어가 있으니 shift를 해서 stack의 top으로 옮겨준다. 이때 눈여겨 볼 점은 모든 단어를 stack으로 옮겨서 buffer가 비었다는 것이다. right-arc는 그냥 left-arc의 반대이다. ->이렇게 하는 건다. 위의 예시에서 fish는 ate의 dependent이므로 fish를 stack에서 제거한다.
  - 마지막으로 right-arc를 한번 더 한다. 그래서 ate는 root의 dependent라는 걸 알아냈다. ate를 stack에서 없애면 root 밖에 남지 않고 buffer에는 아무것도 남지 않으므로 dependency parsing은 끝이 난다.

  ![Imgur](https://i.imgur.com/624EfoE.png)

  - start condition은 ROOT가 stack에 있고, sentence에 있는 모든 단어는 buffer에 있으면 어떤 arc도 없다. 
  - ' | '는 list에 element를 더하는 operation이다. 즉, 단어를 stack에 더하는 것이다. 단어 왼쪽에 vertical bar가 있으면 그 단어는 stack에 있는 단어이다.
  - left-arc를 하면 stack에서 $w_i$가 사라지고 $w_j$만 남게 된다. 그리고 arc에는 $w_j$가 head이고 $w_i$가 dependent가 된다. right-arc는 반대로 하면 된다.
  - ending condition은 stack에 root만 남고 buffer는 비는 것이다.
  - 그런데 언제 어떤 operation을 쓰는지는 어떻게 정할까? 그게 마지막으로 할 일이다...!

  ![Imgur](https://i.imgur.com/dpd1V7Z.png)

  - 다음 action을 결정하는 데, machine learning classifier가 쓰인다. 왜냐하면 sentence를 parse한 tree bank를 가지고 있어서 sentence를 parse한 것들을 라벨로 이용해 어떤 순서로 operation을 썼을 때 정확한 sentence의 parse가 나오는지 학습시킬 수 있다. 결국 지도 학습 문제가 된다.
  - 만약 dependency labels를 포함시키고 싶다면, 일반적인 방법은 left arc와 right arc에 sub type을 하는 것이다. 예를 들면, left arc as an object 이런 식이다.
  - 간단한 form에서는 beam search로 다른 경우의 수를 살펴보지 않고, greedy하게 바로 앞의 단어만 고려한다. 만약 beam search로 할 경우 매우느려지겠지만, 조금 좋아진다고 한다.(하지말라는 뜻 같다.)
  - 왔다갔다 하는 게 아니라 처음부터 끝까지 쭉 parsing하는데 정말 빠르다고 한다.

  ![Imgur](https://i.imgur.com/S9TwdGY.png)

  ![Imgur](https://i.imgur.com/hWyHZFZ.png)

  - 해당 단어를 인코딩하는 방법은 일단, 원핫벡터로 단어를 인코딩하되, stack과 buffer에서의 위치, 단어 자체(good이냐 bad이냐 등등), 품사등 1~3개의 요소를 조합해서 인코딩을 하는 것이다.
  - 하지만 이와 같이 한다면 차원이 매우 커져서 매우 sparse한 feature가 만들어 질 것이다. 다른 방법이 필요하다.
  - 다른 피쳐를 지시하는 피쳐를 이용하는 게 한 가지 방법이다. 위의 예시를 보면 stack의 두 번째 단어인 'has'의 feature를 가지고 있다. 'has'의 tag는 현재 시제이다.  stack의 top word는 'good'이다. 그런 것들이 하나의 feature가 될 수 있다.

  ![Imgur](https://i.imgur.com/0q7RO9H.png)

  - evaluation하는 건 어렵지 않다. treebank에서 없어낸 값과 parsing된 값을 비교해 얼마나 맞았는지 개수를 세면 된다.  이렇게 비교하는 방법에는 두 가지 가 있다.
  - 하나는 화살표만 보고 tag는 무시하는 거다. 즉, 의존 관계만 본다는 것이다. 이런 방법은 **UAS** measure라고 한다. label(tag)도 고려하는 것은 **LAS**라고 한다.

  ![Imgur](https://i.imgur.com/oD4DDXD.png)

  - neural dependency parser를 하는 이유는 위와 같다. machine learning dependency parser는 문제점이 많기 때문이다.

  ![Imgur](https://i.imgur.com/hIKq9Px.png)

  - dense하고 비교적 작은 차원의 feature representation을 얻어낼 수 있다.

![Imgur](https://i.imgur.com/2IJqVKb.png)

- word embedding을 이용해 word를 dense vector로 만들거다. 특이한 것은 POS와 dependency label역시 d-dim vector로 표현하는 것이다. 
- 이렇게 한다면 복수 명사와 단수 명사는 엄연히 다른 것이지만, 명사라는 공통점 때문에 가까이 모일 수 있을 것이다.

![Imgur](https://i.imgur.com/6hxxQvj.png)

(s: stack, b: buffer, lc: left-arc, rc:right-arc)

- 위에서 언급됐던 것과 거의 같은 transition을 할 것이다.
- starting point를 뽑아내는 건 똑같지만, stack의 top, stack의 두번째 top등 각각의 포지션들을 embedding matrix에서 찾아내 dense representation으로 표현할 것이다.
- 그 다음에 하는 건 각각의 것들을 concatenate해서 더 긴 vector로 만드는 것이다.
- dep에 input으로 들어오는 건 이전에 이뤄졌던 의존관계이다. 이전의 값을 토대로 다음 값을 판단한다.

![Imgur](https://i.imgur.com/tofP7Yr.png)

- input layer에 embedding representation과 vector들을 concat한 것을 input으로 준다.
- 그 다음 activation function이  ReLU인 hidden layer를 거치게 한다.
- 마지막으로 output layer에서 softmax를 거쳐 action에 대한 확률 값을 반환하게 한다.
- loss function으로는 cross-entropy error를 이용해 back-propagating한다.
- 잘 학습되면 좋은 dependency parser를 만들 수 있다고 한다.
- 구조가 엄청 복잡하지 않아서 만드는 것도 시도해볼만 할 것 같다...!

<script id="dsq-count-scr" src="//nlp-with-koding.disqus.com/count.js" async></script>
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
