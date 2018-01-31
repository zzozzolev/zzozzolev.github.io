# LSTM Neural Network for Language Modeling

## 1. Introduction

- 일반 rnn의 경우 vanishing gradient 문제가 있었음.
- 논문의 경우 영어랑 프랑스 코퍼스에 효율성  둠.

## 2. LSTM neural networks

- recurrent neural network에서 gradient가 시간*이 지날수록 발산하거나 수렴함. 

  *랭귀지 모델링에서 time step은 문장에서 단어의 위치와 대응.

- unit은 몇몇 gating unit에 의해 향상된다.

- 기존 LSTM unit에서 두 가지 수정을 함.

![lstm](C:\Users\nhm\ybigta\NLP\캡처.PNG)

- 기존의 neural network와 달리 lstm은 중간 step이라는 게 있다.

- $$a_i$$에  activation 함수를 적용한 후, 그 결과를 factor $$b_l$$과 곱한다.

- $$b_\phi$$과 곱해진 이전 time step의 inner activation value가 더해진다?

- 그 결과는 $$b_w$$에 의해 평가? 올라가고?(scale) $$b_i$$값을 뽑아내면서 또 다른 activation 함수로 보내진다.

- factor인 $$b_l, b_\phi, b_w$$는 (0, 1)에 속하고 input, output, forget gate에 의해 각각 통제된다.*

  *위의 그림에서 하얀 동그라미가 $$b_l, b_\phi, b_w$$이고 파란 동그라미가 gate이다.

- gating unit은 LSTM unit의 내부 activation 뿐만 아니라, 이전 hidden layer의 activation과 이전 time step에서 온 현재 layer의 activation도 더한다.

- 결과 값은 sigmoid 함수에 의해 0~1사이의 값이 된다.(set to factor)

  *본 논문에서는 LSTM의 긴 수식 생략

  ## 3. Neural network language models

- input 단어들은 1-of-K* coding으로 인코딩 됨.

  *K: vocabulary에서 단어 갯수.

- output layer에서 softmax 함수가 사용됨.

- cross entropy error가 사용됨.

- data는 쉽게 정규화될 수 있는데, 오직 training data에서 관측되는 단어들의 unigram counts에 달려있다.

  ![캡처2](C:\Users\nhm\ybigta\NLP\캡처2.PNG)

- 첫번째 layer는 continuous space에 input 단어를 projecting하는 것에 대한 interpretation을 갖는다.

- LSTM unit을 두번째 recurrent layer에 plug한다.

- 이 유닛을 standard neural network인 다른 projection layer과 combining 한다.
  $$
  a_i = \sum^J_{j=1}w_{ij}b_j
  $$

- 큰 단어 수를 가진 언어 모델링에서, softmax output layer의 input activation인 $$a_i$$의 계산이 traing에 많은 영향을 끼친다. 

- J는 last hidden layer에서 노드의 갯수이다.

- $$w_{ij}$$는 last hidden layer와 output layer 사이의 weight이다.

- $$i=1,...,V$$이고 V는 단어 사이즈이다.

- computational effort를 줄이기 위해,  단어들을 disjoint word classes로 나누는 것이 제안됐다. 

- define a reasonable set of classes is described in Mikolov, T., Kombrink, S., Burget, L., Cˇ ernocky´, J., Khudanpur,
  S., “Extensions of Recurrent Neural Network Language Model”,

## 4. Conclusion

- 단어 sequence에 대한 확률의 정확한 모델링을 할 수 있는 언어 모델에 적합하다. 
- standard neural net의 training 문제는 겪지 않았다. 
- 추가적인 hidden projection layer의 wide-spread한 사용이 중요했다.
- 적은 loss를 가지면서 traing 시간과 testing 시간을 단축시키기 위해, LSTM network가 clustering 기법과 혼합될 수 있었다.

![캡처3](C:\Users\nhm\ybigta\NLP\캡처3.PNG)





