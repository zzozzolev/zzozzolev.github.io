---
comments: true
title: "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank Review"
date: 2017-07-18 
categories: 2013 RNTN sentiment-analysis 
---
##  Recursive Neural Models

![rnn 접근](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\rnn 접근.JPG)

- n-gram이 compositional model에 주어졌을 때, vector로 표현되는 단어들이 두 갈래로 쪼개지는 binary tree와 각각의 leaf node로 parsing 된다.
- Recursive neural models은 다른 input(자식 vector)을 가지는 함수 g를 이용해 parent vector를 계산한다. parent vector는 또 다른 함수의 input, 즉 feature로 제공된다.
- 이처럼 자식 vector가 합쳐져 부모 vector가 되고, 다시 자식 vector가 되어 또 다른 자식 vector와 합쳐지기 때문에 **Recursive Neural Models**라고 하는 것이다.



## 이전 모델들



## Recursive Neural Network

- 부모 vector는 자식 vector를 가지고 있다. 즉, 부모 단어는 자식 단어들의 특성을 띄고 있는 것이다.
- RNN의 기본적인 모델이라고 생각하면 될 것 같다.



## MV-RNN: Matrix-Vector RNN 

![mv-rnn](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\mv-rnn.JPG)

- 모든 단어와 긴 구문을 vector와 matrix로 된 parse tree로 나타내는 것이 주된 아이디어이다.
- 각각의 n-gram은 (vector, matrix) pair인 리스트로 나타나진다. 

![mv-rnn 계산](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\mv-rnn 계산.JPG)

(소문자로 된 것은 vector, 대문자로 된 것은 matrix)

- 단어 혹은 구문인 두 개의 구성요소가 결합될 때, 구성요소를 나타내는 matrix는 다른 구성요소의 vector와 곱해진다. 
- compositional 함수는 결합되는 단어들을 변수로 사용한다. (Hence,thecompositional function is parameterized by thewords that participate in it.) 



## RNTN: Recursive Neural Tensor Network

- 아이디어는 모든 노드에 tensor 기반의 composition function을 쓰는 것이다.

![rntn ntl](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\rntn ntl.JPG)

(위 그림과 아래 그림에 나오는 a, b, c는 Recursive Neural Models에 있는 것을 가리킨다. )

- V는 2dx2d를 d개 가지는 즉, [2dx2dxd]인 tensor이다. V는 composition의 특정 타입을 담고있다. (동사구냐 명사구냐 뭐 이런건가요ㅠㅠ?)

  (1차원 = vector, 2차원 = matrix, 3차원 = tensor)


![p1](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\p1.JPG)

![p2](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\p2.JPG)

- p1과 p2의 W는 같다. p1은 또 다른 인풋으로 들어가게 된다.
- V가 0으로 설정됐을 때, V는 직접적으로 input vector와 관련을 맺는다. (standard layer 값만 남게 되므로?)



### Tensor Backprop through Structure

- target vector와 predicted vector는 one-hot encoding으로 돼있다. 
- 각각의 노드는 softmax classifier를 갖는다.
- target vector와 predicted vector의 차이를 줄여야 한다. 즉, KL-divergence(Kullback Leibler divergence)를 최소화시켜야 한다. 
- 각각의 노드는 weights V,W을 반복적으로 사용해 backprop한다.

![에러](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\에러.JPG)

- 노드 i에서의 softmax error vector는 위와 같이 구해진다. 

1. 노드 i의 predicted vector와 target vector를 빼준 결과를 softmax하고 transpose한 W와 곱해준다.

2. 노드 i의 vector를 tanh인 f에 파라미터로 주고 미분한다.

3. 1의 결과와 2의 결과를 Hadamard product해준다. 

   *Hadamard product: 두 행렬의 element중 행과 열이 모두 같은 것 끼리 더해주는 것. 예를 들면 A행렬의 1행 1렬은 B행렬의 1행 1렬과 곱해주는 것이다.

- 나머지 미분은 top-down 방식으로만 계산될 수 있다. 
- V와 W에 대한 미분은 각각의 노드의 미분의 합이다.
- top 노드(부모 노드)는 top 노드의 softmax로부터만 에러를 받는다. 즉, top 노드의   에러는 top노드의 softmax error인 것이다.
- leaf 노드(자식 노드)는 error를 계산할 때, 자신의 softmax error와 부모 노드의 error 절반을 더한다. 만약 부모 노드의 왼쪽에 위치해 있다면 절반의 첫번째를 더하고, 오른쪽에 위치해 있다면 절반의 두번째를 더한다. (아마 slice의 윗부분과 아랫부분 아닐까 생각함...)



### Model Analysis: High Level Negation

#### Set 1: Negating Positive Sentences.

![긍부정](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\긍부정.JPG)

- "It's not good."과 같이 긍정적인 문장들과 그 문장들의 부정이 포함되어 있었다. 부정이 전반적인 sentiment를 긍정에서 부정으로 바꿔 놓았다.
- RNTN은 'least'와 같은 덜 명확한 표현이 있을 때도 분류할 수 있었다.
- 즉, 하나의 단어를 제외하고 모든 단어가 긍정을 나타날 때도, 그 하나의단어가 부정적인 단어라면 문장의 sentiment가 부정으로 바뀌었다.

![긍부정 비교](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\긍부정 비교.JPG)

- 긍정적인 문장이 부정적인 단어로 인해 부정적인 문장으로 변할 때, RNTN이 좀 더 positive sentiment를 감소시켰다.

#### Set 2: Negating Negative Sentences.

![부긍정](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\부긍정.JPG)

- "It's not bad."와 같이 부정적인 문장과 그 문장들의 부정이 포함되어 있었다.

- RNTN을 이용한 sentiment treebank에서는 이러한 문장들을 덜 부정적인 문장으로 처리했다. 부정적이지 않다고 해서 긍정적인 문장은 아니기 때문이다.

  ![부긍정 비교](C:\Users\노혜미\ybigta\17-2 사이언스\RNN\부긍정 비교.JPG)

- RNTN만 정확하게 부정적인 문장이 부정 표현으로 인해 덜 부정적인 문장으로 바뀌는 것을 감지해낼 수 있었다. neutral activation과 positive activation을 증가시키는 것이다.



## Conclusion

- sentimental semantic을 분석하는 데는 RNTN이 현재 짱짱이다.
- 긍정과 부정표현을 구분해낼 뿐만 아니라, 긍정 표현이 부정 표현이 되는 것과 부정 표현이 중립 표현이 되는 것 또한 정확히 포착해낼 수 있다.

## Reference
-[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
