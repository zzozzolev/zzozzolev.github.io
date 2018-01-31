# Lecture 3 | GloVe: Global Vectors for Word Representation   

강의 출처:

https://www.youtube.com/watch?v=ASn7ExxLZws&t=752s&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=3



#### 들어가기 전에...

- word vector의 초기화는 특정 값 사이에 있는 랜덤한 값으로 한다. 보통 유니폼 분포를 사용한다. 


------


![1](http://i.imgur.com/sSVkgFj.png)

- SGD는 window를 이동시키면서 업그레이드를 하는 방식이다. 하지만 이런 방식으로 나온 결과가 sparse하다. 즉, 중간중간 값이 없는 곳이 많다는 것이다. 예를 들면 위 그림에서 보듯이, "I like deep learning."이라는 문장이 있을 때 window내에서의 update는 해당 단어들만이 될 것이다. 가령 "dog"에 해당하는 부분은 0이 될 것이다.

![2](http://i.imgur.com/O99mJqs.png)

![3](http://i.imgur.com/zLCXnX6.png)

- 이처럼 sparse한 word vector들을 효율적으로 훈련시키는 방법이 나오는데, 바로 *negative sampling*이다.

- 주요한 아이디어는 center word와 context word를 묶은 *true pair*와 center word와 context word는 아니지만 corpus에 있는 random한 word를 묶은 *noise pairs*를 만드는 것이다.

- 그래서 center word 다음에 어떤 context word가 얼마나 나올지 확률을 맞추는 것이 아니라, 해당 pair가 *옳은지 틀린지를*  맞추는 binary logistic regression 문제가 되는 것이다.

  ![4](http://i.imgur.com/0bMtd5C.png) 

- objective function에서 T는 window와 time step을 지칭한다. $\theta$는 모델에서 parameter로 쓰이는 모든 변수이다. skip-gram 모델에서는 centor word vector와 context word vector이다.

- $J_t(\theta)$에서 왼쪽 부분은 center word와 context word가 동시에 일어날 로그 가능성이다. center word vector와 context word vector의 곱을 확률로 만들어 주기 위해 아래 보이는 sigmoid 함수에 넣어준다. 우리의 목표는 이 로그 가능성을 높이는 것이다.

![5](http://i.imgur.com/vZyuMJs.png)

- 식의 오른쪽 부분이 코퍼스에서 랜덤하게 몇몇 단어 샘플을 뽑아주는 것이다. 그리고 이 가능성을 최소화 시켜야 한다. 

- 랜덤하게 뽑는 방법은 간단한 유니폼 분포나 유니그램 분포*에서 뽑아내는 것이다.

  *모든 단어의 활용이 서로 독립이라고 가정하고 단어 열의 확률을 각 단어의 확률 곱으로 나타낸 모형. 

  ([여기](https://datascienceschool.net/view-notebook/a0c848e1e2d343d685e6077c35c4203b/) 참고)

- 유니그램 분포에서 $3\over4$을 지수로 취하는 이유는, 매우 자주 등장하지만 쓸모 없는 단어들 예를 들면 the, a 이런 것들을 뽑지 않기 위해서이다.

- 랜덤하게 뽑은 샘플 단어가 context 단어와 같지 않아야 하지만, 매우 큰 코퍼스에서 확률은 매우 작아서 엄청 드물게 발생한다고 한다.

  ![6](http://i.imgur.com/ijbeCJK.png)

- CBOW는 skip-gram model과 달리 context words로 center word를 예측하는 것이다.

![7](http://i.imgur.com/906DRk4.png)

- 비슷한 의미를 지닌 단어끼리 클러스터를 형성하게 된다. 위의 그림에서 숫자를 나타내는 단어끼리, 요일을 나타내는 단어끼리 묶인 것을 알 수 있다.

![8](http://i.imgur.com/R5I8nep.png)

- 여기서 필수적으로 해야할 것은 단어들의 coocurrence를 알아내는 것이다. 예를 들면 'deep'과 'learning'이 함께 쓰이는 것을 보고 두 vector를 update한 뒤, 다시 코퍼스를 살펴보니 두 단어가 또 동시에 쓰였다. 그렇다면 다시 separate update를 한다.
- 이때 의문이 들 수도 있다...! '왜 전체 코퍼스를 살펴보고 한 번 update를 하지 않고, sample마다 할까?'라고...답은 밑의 이미지에서 socher 교수님이 줄 것이다.

![9](http://i.imgur.com/wjsNXkE.png)

- word2vec이 나오기 이전 다르게 하는 두 가지 방법이 있었다.
- 하나는 window를 쓴다는 점이 word2vec과 비슷하지만, window마다 update를 하지 않는다.  먼저 counts를 하고 matrix에 변화를 주는 것이다. 이때 해당 단어의 문법적인 정보(POS)와 의미적인 정보를 동시에 알아낸다. 예를 들면 해당 단어가 명사인지 동사인지 이런 정보도 알 수 있다.
- 두번째 방법은 전체 문서에서의 coocuurence를 보는 방법이다. 해당 단어가 전체 문서에서 어떤 단어와 자주 등장해는지를 파악하는 것이다. 이때 알아낼 수 있는 것은 topic일 것이다. 이러한 방법을 *Latent Semantic Analysis*라고부른다. 단어의 syntactic information은 무시한다. 예를 들면 수영, 배, 물, 날씨가 이 topic에서 document에서 같이 나타났다는 것을 알 수 있다.

![10](http://i.imgur.com/LmVNEL8.png)

![11](http://i.imgur.com/GdPUXh0.png)

![12](http://i.imgur.com/aT8OGj5.png)

- 만약 새로운 단어가 추가된다면 word vector가 바뀔 것이다. 그리고 downstream machine learning model이 있다면, 항상 vector를 바꿔줘야 할 것이고 약간씩 parameter 손실이 있을 것이다.
- 위의 예시는 매~우 작은 코퍼스이니 상관이 없었지만, 보통은 수만 개의 단어를 가진다. 그래서 매우 큰 차원의 vector를 가진다.
- word vector의 특성상 sparse한 데이터가 만들어 질 수 밖에 없다.
- 따라서 모델은 robust하지 않게 된다.

![13](http://i.imgur.com/uRi8nRq.png)

- counts matrix의 모든 정보를 저장하지 않고 중요한 정보만 저장하는 것이 하나의 해결책이다. 이렇게 할 경우, vector의 dimension이 줄어들기 때문이다.

![14](http://i.imgur.com/2Qwx7zT.png)

*[SVD 설명](http://rfriend.tistory.com/185)

- U는 left-singular vectors이고 S는 가장 큰 것부터 가장 작은 singular value를 가지고 있는 diagonal matrix이다. V는 right-singular vectors이다. 

![15](http://i.imgur.com/qE0oC3E.png)

- 이를 코드로 구현하면 간단하다. (SVD 설명은 간단하지 않지만...)
- X는 위에서 봤던 counts matrix이다.

![16](http://i.imgur.com/AendKb4.png)

- 'I'와 'like'는 코퍼스에서 자주 등장하기 때문에 왼쪽에 위치해있다.
- 매우 간단한 method로 word vectors가 어떻게 표현될 수 있는지를 알아낼 수 있다.

![17](http://i.imgur.com/MvyxEbc.png)

- 하지만 coouccurence matrix에도 문제는 있다.
- 일단, 문장내에서 다른 단어와 문법적 혹은 구조적 관계를 보여주는 'he', 'has', 'the'같은 단어가 너무 빈번하게 등장한다. 그래서 최대 등장할 수 있는 빈도 수를 100같은 특정 값으로 고정하는 방법도 있고, 그러한 단어들을 무시하는 방법도 있다. 이렇게 하는 이유는 등장 빈도가 매우 적은 단어이지만 중요한 의미를 내포하고 있을 수도 있기 때문이다.
- 단어들을 counting할 때, 동일하게 세지 않는 방법도 있다. 가까운 단어일수록 더 counts를 크게 세는 것이다. 예를 들면  바로 옆에 있는 단어는 1로, 5단어 이전에 나타나는 단어는 0.5 이런 식으로 세는 것이다.
- counts 대신에 pearson 상관계수를 쓰는 방법도 있다. [여기](http://blog.daum.net/123kp456/11416177)를 참고했다. 계수가 양수 일때는 한 변수 값이 크거나 작아질수록 다른 변수 역시도 크거나 작아지는 것이다. 음수 일때는 한 변수 값이 커질때 작아지고, 작아질때 커지는 것이다. hack에서 음의 값은 0으로 처리한다. 

![18](http://i.imgur.com/dXbAQ9f.png)

- 2005년에 SVD를 이용해서 위와 같은 결과를 얻어냈다. 
- high dimension vector이기 때문에 시각화하기 위해, 몇 개의 단어를 골라 서로 가장 가까운 단어들을 살펴봤다.
- 비슷한 단어끼리 클러스터링 되는 결과를 확인할 수 있었다.

![19](http://i.imgur.com/MQMwqRh.png)

- 비슷한 문법적 패턴을 보이는 단어끼리 묶어졌다.

![20](http://i.imgur.com/ANiLeMC.png)

- 비슷한 의미 패턴을 보이는 단어끼리 묶어졌다.

![Imgur](http://i.imgur.com/HOxKScL.png)

- 하지만 간단해 보이는 SVD도 문제가 있다.
- 계산 비용 문제가 있고, 새로운 단어나 문서와 통합시키기가 힘들다. 그리고 다른 딥러닝 모델들과 optimization이 매우 다르다. 

![Imgur](http://i.imgur.com/ersowZw.png)

- 기본적으로 counts based methods는 SVD를 기반으로, direct prediction은 window를 기반으로 한다.
- direct prediction의 경우, entity recognition(개체 인식)이나 speech tagging과 같은 문제에 적용시킬 수 있다.

![Imgur](http://i.imgur.com/683blyF.png)

- u와 v vector가 parameter이다. 하지만 더 symmetric하다.
- 동시에 나타나는 모든 단어들의 pairs를 살펴본다. 
- skip-gram 모델이 하나의 윈도우에서 한번 씩 co-occurrence를 봤다면, Glove model은 전반적인 counts를 살펴본다. 즉, 해당 단어들이 함께 얼마나 나타났는지를 파악하는 것이다.
- 기본적으로 inner product와 두 단어의 log count를 최소하시키는 것이 목표이다. 
- f는 그래프를 통해 알 수 있듯이 최대 빈도를 제한시켜주는 함수이다. 

![Imgur](http://i.imgur.com/BPXwp70.png)

- 왜 굳이 U와 V를 나누는지 궁금할 수 있다. 그 이유는 optimization의 관점에서 생각해봤을 때, optimization을 하는 동안 분리된 vector를 가지고 있다가 맨 마지막에 합치는 것이 더 안정적이라는 결과가 나왔기 때문이다.

![Imgur](http://i.imgur.com/t0AsjRC.png)

- 매우 드문 단어들일지라도, Glove가 근접한 단어들을 적절하게 주는 것을 알 수 있다.

![Imgur](http://i.imgur.com/qdLVut6.png)

- intrinsic evaluations는 몇몇의 specific하거나 intermediate한 subtask이다. 예를 들면 vector들의 차이, vector들의 유사성, 내적이 인간의 판단과 얼마나 유사하냐를 판단하는 것이다.
- 일단 계산하기가 빠르다는 장점이 있고, 때때로 system이 어떻게 작동하는지 이해하는데 도움을 준다. 어떤 종류의 hyperparameters가 실제로 유사도 matrix에 영향을 끼치는지와 같은 것들을 알려주는 것이다.
- 하지만, 기계 번역이라든지 실제 task에 적용하지 않는 이상 정말 도움이 되는지 명확하지 않다. 한마디로 잘못하면 삽질할수도 있다는 거다.
- extrinsic evaluations는 실제 task에서 시험해보는 것이다. 물론 비교적 정확할 수는 있으나, 정확도를 측정하는데 오랜 시간이 걸린다.
- 실제 task에서의 system이 잘못된 것인지, 아니면 word vectors가 잘못된 것인지 명확하지 않을 수 있다. 그래서 모델을 향상시킬 때, 한번에 둘 다 바꾸지 말고 한가지만을 바꿔야 한다.

![Imgur](http://i.imgur.com/sD5kHjE.png)

- vector간의 cosine distance를 통해 semantic analogies와 syntactic analogies를 잘 할 수 있었다.
- 위의 그림에서 나오듯이 :man:::woman:의 대응과 비슷한 대응은 :king:::princess:이다. woman vector에서 man vector를 빼고 king vector를 더한뒤, 가장 큰 cosine similarity를 가지는 vector는 queen이었다.

![Imgur](http://i.imgur.com/MK3QFlA.png)

- 다른 단어들 사이에서도 이와 같은 결과를 발견할 수 있었다.

![Imgur](http://i.imgur.com/5DHeUUW.png)

- 회사이름에서 CEO를 빼고 다른 회사이름을 더하면 다른 회사의 CEO가 나왔다.

![Imgur](http://i.imgur.com/ewby6Aq.png)

- semantic 정보만 알 수 있었던게 아니라 syntactic 정보 또한 알 수 있었다.

![Imgur](http://i.imgur.com/PJ8U5px.png)

![Imgur](http://i.imgur.com/btKXPRQ.png)

![Imgur](http://i.imgur.com/XWgWoKK.png)

![Imgur](http://i.imgur.com/zX02aLg.png)

- word vector를 계산하고 knob을 튜닝하고, 25 dimensions 대신에 50 dimensions를 하는 등 hyperparameter를 변화시켜 어떤 것이 해당 analogies에 적절한지 판단한다.

![Imgur](http://i.imgur.com/VarAAZ9.png)

- 다양한 모델과 GloVe를 비교해봤다. 그 결과...! GloVe가 위와 같은 단어 들의 관계를 가장 잘 파악하는 것을 알 수 있었다.
- 그리고 dimension이 많다고 해서 이러한 관계들을 더 잘 파악하는데 도움을 주지는 못했다.
- 대부분의 딥러닝 모델들이 data가 많을 수록 더 잘 작동하는 것도 알 수 있었다. 하지만 양만 크다고 모두 결과가 좋은 것은 아니다. 해당 data가 어떤 것이냐에 따라서 달라질 수도 있다. 예를 들면 wikipedia를 data로 이용하면 아프리카 수도라든지 여러나라의 수도의 semantic을 잘 파악할 수 있겠지만, 미국 뉴스를 data로 이용한다면 그렇지 않을 것이다. 즉, data의 양뿐만 아니라 질 또한 고려해줄 부분이다.

![Imgur](http://i.imgur.com/IVef4lD.png)

- 위같은 plot을 그리는 건 좋은 일이지만(socher 교수님이 plot 그리면 점수를 더 준다고 한다. :haha:) 향상된 결과만을 보여주면 안된다. (a)처럼 이후에는 상승하지 않고 감소할수도 거의 변화가 없을 수도 있기 때문이다. 즉, 해당 parameter에 대한 optimum value가 뭔지 파악해야 한다.
- window size의 경우 8에서 가장 좋은 결가가 나왔지만, training 시간이 증가한다.

![Imgur](http://i.imgur.com/fmk8My9.png)

![Imgur](http://i.imgur.com/BXwOI1S.png)

![Imgur](http://i.imgur.com/sxSjyHB.png)

- 사람에게 직접 물어봐서 판단을 하는 방법도 있다.
- 다수가 대학원 학생들로 구성된 집단에게 두 단어가 얼마나 유사한지 물어보는 방법이다.
- 해당 dataset을 이용해서 자신의 모델이 얼마나 잘 train됐는지 판단할 수 있다.

![Imgur](http://i.imgur.com/IDJkKMq.png)

- 비교적 간단하면서 좋은 extrinsic evaluation은 named entity recognition이다.

![Imgur](http://i.imgur.com/awVZBE9.png)

- word2vec과 같이 deep learning된 word vector의 장점은 비슷한 단어들끼리 뭉쳐지기 때문에, training data set에서 보지못했던 다른 단어들도 더 견고하게 분류할 수 있다. 예를 들면, 목표가 location 단어들을 분류하는 것이라면 모든 country 단어들을 비슷한 vector 공간에 위치시키는 게 좋을 것이다. 
- 하지만 모든 task에서 무조건 성능이 좋지는 않았다. 예를 들어, 감정 분석의 경우 좋은 결과가 나오지 않았는데, 왜냐하면 'bad'와 'good'이 비슷한 context에서 쓰이는 등 문제가 있었기 때문이다. 그래서 감정 분석과 같은 task를 할 때는 word vectors를 random하게 초기화하는 것이 더 좋다.
- 그래도 대부분의 경우 word2vec이나 glove model과 같이 deep learning된 word vector가 도움이 된다고 한다. 

![Imgur](http://i.imgur.com/fmxhxUZ.png)

- 전의 강의에서 softmax 함수를 본 적이 있을 것이다. 하지만 notation을 약간 바꾼다고 한다.
- logistic regression은 softmax classification의 다른 용어이다.
- matrix W는 class이다. 예를 들어, [0.3 , 0.2]와 같은 vector의 class는 dog라고 할 수 있다. 각각의 class는 하나의 row vector를 가진다. 그래서 y는 row의 개수이다. x는 word vector이다. $W_y \cdot x$ 는 row vector와 column vector의 내적이다.
- d는 input 개수이고 C는 class 개수이다.
- context없이 다양한 word vectors를 분류할 것이다.
- 예를 들면, x가 location인지 아닌지를 알고 싶다. 단순히 yes인지 no인지를 판단하면 된다. 

------

## Reference

- https://www.youtube.com/watch?v=ASn7ExxLZws&t=752s&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=3
- https://datascienceschool.net/view-notebook/a0c848e1e2d343d685e6077c35c4203b/
- http://rfriend.tistory.com/185