---
comments: true
title: "Attend to You: Personalized Image Captioning with Context Sequence Memory Networks Review"
date: 2018-03-06 03:31
categories: 2017 CNN memory network generative-model 
---
# Abstract
- 다양한 타입의 context 정보에 대한 저장소로 memory를 이용, 이전에 생성된 단어들을 memory에 추가, CNN을 이용해는 memory 정보를 추출하는 과정을 통해 개인화된 captioning을 할 수 있었다.

# 1. Introduction
- 이미지에 대한 알고리즘 뿐만 아니라 자연스러운 문장을 만들기 위해 이미지에 대한 해석과 언어 모델을 연결시켜야 됐다.
- 해당 논문은 image captioning에서 개인화 문제를 다룬다.
- 논문에서는 **유저가 사용하는 단어와 문체**를 고려해 이미지에 대한 묘사를 만들어낸다. 
- 초점을 맞춘 task는 **해쉬 태그 예측**과 **포스트 생성**이 되겠다. (이모티콘도 포함)
- *context sequence memory network* 즉 *CSMN*을 제시하겠다.
- 두 task 모두 word sequence 예측으로 볼 수 있기 때문에 dict만 바꾸면서 같은 model을 사용했다.
- 다만, 해쉬 태그를 순서가 없다고 간주하지 않고 게시글과 같이 순서가 있다고 전제했다.

# 4. The Context Sequence Memory Network
![Imgur](https://i.imgur.com/NdPFKhP.png)
- (a): image description과 유저의 이전 게시글을 통해 context memory를 설정한다.
- (b): memory state를 기반은 매 time step t마다 단어를 예측한다.
- (c): 새로운 아웃풋 단어가 만들어 진다면 word output을 업데이트한다.
- 인풋은 특정 유저의 query image $$I_q$$이고 아웃풋은 문장이다. $${y_t}=y_1,...,y_T$$
- 즉, $${y_t}$$는 해쉬 태그와 게시글에 대응한다.
- 선택적인 인풋은 메모리에 더해지는 해당 유저의 활성 단어들과 같은 context 정보이다.
## 4.1. Construction of Context Memory
- 세 가지 종류의 context 정보를 저장하기 위해 메모리를 만들었다. 
 1. 쿼리 이미지에 대한 표현을 위한 *image memory*
 2. 유저의 이전 게시글을 기반으로 하고 TF-IDF로 가중치를 둔 D개의 빈번한 단어들에 대한 *user context memory*
 3. 이전에 생성된 단어들에 대한 *word output memory*
- 메모리에 대한 각각의 인풋은 인풋과 아웃풋 memory representation으로 임베딩된다. 이에 대해 위첨자 a와 c를 각각 쓰겠다.

**Image Memory**
- ImageNet 2012 dataset에 대해 pretrained된 ResNet101을 이용해 이미지를 표현했다.
- 두 개의 다른 description을 사용했는데 하나는 res5c layer의 feature maps와 pool5 feature vectors이다.
- res5c의 feature map $$I^{r5c}$$은 모델이 spatial attention을 사용할 때 유용했고 pool5의 feature $$I^{p5}$$는 이미지의 feature vector로 사용됐다.
- 그래서 pool5는 single memory cell에 삽입되었지만 rec5c feature map은 49개의 cell들을 차지한다.
- 이미지 memory vector $$m_{im}$$은 다음과 같다.

$$m_^{a/c}_{im,j}=ReLU(W^{a/c}_{im}I^{r5c/p5}_j+b^{a/c}_{im})$$

- input memory와 output memory 모두 같고 r5c와 p5 모두 같아서 편의상 위와 같이 한번에 썼다. (논문에서도 약간 이런 식으로 함...)
- 아래 나오는 식들은 res5c feature를 사용한다는 걸 가정한다.

**User Context Memory**
- $${u_i}^D_{i=1}$$을 유저의 이전 게시글에서 가장 빈번한 D개의 단어들이라고 정의한다.
- $${u_i}^D_{i=1}$$을 점수를 감소시키면서 user context memory에 인풋으로 넣는다.(CNN을 나중에 더 효과적으로 사용하기 위해서라고 한다.)
- context memory는 유저의 활성 단어들 혹은 해쉬 태그를 이용한 문체에 좀 더 집중해서 모델의 성능을 향상시킨다.
- $${u_i}^D_{i=1}$$을 만들 때는 단순히 가장 빈번한 단어만 고려하지 않았다. TF-IDF 점수를 이용했다. 즉, 많은 유저들이 흔하게 사용하는 단어는 사용하지 않았다는 뜻이다. 이렇게 한 이유는 그런 단어들이 개인화에 도움이 되지 않기 때문이다.
- user context memory vector $$m^{a/c}_us$$는 다음과 같다.

$$u^a_j=W^a_e u_j, u^c_j = W^c_e;y_j; j \in 1,...,D$$
$$m_^{a/c}_{us,j}=ReLU(W_h[u^{a/c}_j]+b_h)$$

- $$u_j$$는 j번째 활성 단어에 대한 one-hot vector이다.
- input과 output memory에 같은 $$W_h$$를 사용했지만, 별 개의 단어 임베딩 매트릭스를 사용했다.

**Word Output Memory**
- word output memory에 이전에 만들어진 일련의 단어들 $$y_1,...,y_{t-1}$$을 삽입했다. 다음과 같이 표현된다.

$$o^a_j=W^a_e y_j, o^c_j = W^c_e;y_j; j \in 1,...,t-1$$
$$m_^{a/c}_{ot,j}=ReLU(W_h[o^{a/c}_j]+b_h)$$

- $$y_j$$는 j번째 이전 단어에 대한 one-hot vector이다.
- 같은 word embeddings $$W^{a/c}_e$$를 사용했고 파라미터 $$W_h,b_h$$는 user context memory와 같다.
- 새로운 단어가 만들어질 때마다 모든 iteration에 대해 $$m^{a/c}_{ot,j}$$를 업데이트 했다.

- 최종적으로 인풋과 memory representation을 concatenate했다.

$$M^{a/c}_t = [m^{a/c}_{im,1} \bigplus \cdots \bigplus m^{a/c}_{im,49} \bigplus m^{a/c}_{us,1} \bigplus \cdots \bigplus m^{a/c}_{us,D} \bigplus m^{a/c}_{ot,1} \bigplus \cdots \bigplus m^{a/c}_{ot,t-1}$$

- m은 메모리의 사이즈를 나타내고 세 개의 메모리 타입의 크기의 합이다.

## 4.2. State-Based Sequence Generation
- RNN이 sequence 생성에 많이 쓰이지만 우리는 쓰지 않았다. 대신 순차적으로 memory에 이전까지 만들어진 모든 단어를 저장한다.
- 각각의 아웃풋 단어를 예측하는 건 모든 이전 단어들, 이미지 구역, user context의 조합에 선별적으로 attention을 줌으로써 이루어진다.
- memory state를 기반으로 time step t에서 단어 $$y_t$$에 대한 예측이 어떻게 이루어지는지 살펴보자.
- $$y_{t-1}$$을 이전 단어에 대한 one hot vector라고 해보자.
- time t일 때, memory network에 다음과 같은 input vector $$q_t$$를 만들어 낸다.

$$q_t = ReLU(W_qx_t+b_q)$$, where $$x_t = W^b_ey_{t-1}$$

- 다음으로 $$q_t$$는 context memory의 attention model에 feed된다.

$$p_t = sotfmax(M^a_tq_t), M_{Ot}(*,i)=p_t \circ M^c_t(*,i)$$

- 행렬곱은 한 뒤 softmax를 취해서 input vector $$q_t$$가 각각의 memory $$M^a_t$$와 잘 맞는지 확인했다.
- $$p_t$$는 현재 time step에서 input memory의 어떤 부분이 input $$q_t$$에 중요한지 나타낸단.
- output memory representation $$M^c_t$$의 각각의 컬럼을 $$p_t$$와 element-wise 곱을 통해 rescale했다.
- 결과적으로, attention이 적용된 output memory representation $$M_{O_t}$$를 얻어낼 수 있었다.
- $$M_{O_t}$$는 세 가지 memory type들로 분해될 수 있다. [m^{o}_{im,1:49} \bigplus m^{a/c}_{us,1:D} \bigplus m^{a/c}_{ot,1:t-1}]

**Memory CNNs**
- 그 다음으로 $$M_{O_t}$$에 CNN을 적용했다. 확실히 CNN을 사용하면 captioning 성능을 확연히 올렸었다.
- depth가 300인 세 개의 filter를 정의했다. window sizes h는 [3,4,5]를 이용했다.
- 각각의 memory 타입에 별개로 하나의 convolutional layer와 max-pooling layer를 사용했다.

$$c^h_{im, t}=maxpool(ReLU(w^h_{im}*m^o_{im,1:49}+b^h_{im}))$$

- *는 convolutional operation을 나타낸다.
- 최종적으로 $$c^h_{im, t}$$을 h= 3~5까지 concatenate해서 $$c_{im, t}$$를 얻는다.
- 다른 memory type에도 이와 같은 연산을 적용해 $$c_t = [c_{im,t} \bigplus c_{us,t} \bigplus c_{ot,t}]$$

- 다음으로 output 단어 확률 $$s_t$$를 계산한다.

$$h_t=ReLU(W_o c_t+b_o), s_t = softmax(W_f h_t)$$

- 마지막으로 가장 높은 확률을 가지는 $$y_t$$를 $$argmax_{s \in \mathcal{V}}(s_t)$$
- 만약 output word $$y_t$$가 EOS 토큰이 아니라면 $$y_t$$를 word output memory에 feed하고 time step t+1의 input으로 feed한다.
- 요약하자면 논문에서 사용한 inference는 모델이 각각의 time step에서 최적의 단어를 순차적으로 찾는 과정을 통해 최적의 sequence를
 만들어내기 때문에 **greedy**하다. 

# Result
**post generation**
![Imgur](https://i.imgur.com/B3wEIkU.png)

**hashtag prediction**
![Imgur](https://i.imgur.com/Xe9EzI7.png)

<script id="dsq-count-scr" src="//nlp-with-koding.disqus.com/count.js" async></script>

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-113467528-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-113467528-1');
</script>
