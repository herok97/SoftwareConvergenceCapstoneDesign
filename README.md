# Unsupervised Video Summarization with Adversarial LSTM Networks 모델 구현 및 성능 향상에 대한 연구

**2021-1 소프트웨어융합캡스톤디자인** 

**[김영웅 (응용수학과)](https://khero97.tistory.com/)**

## 개요

### **Video Summarization**

- **필요성**

  효율적인 비디오 분석과 자료 조사를 위해 비디오 요약을 자동화하는 기술은 많은 응용분야에서 필요하다. 응용하는 분야가 어떤 것인가에 따라 비디오 요약의 정의는 다양하다. 비디오 내의 중요한 정지 영상을 여러 개 추출하거나(Key frame selection), 비디오 내의 비슷한 정지 영상들을 제거하여 영상을 거의 똑같이 표현함과 동시에 비디오 길이를 줄이는 방법 등이 있다. 또한 비디오가 나타내는 문맥을 이해하여 텍스트로써 표현하는 것도 있다.
  여기서 말하고자하는 비디오 요약이란 key frame selection을 의미하며, 전체 비디오 frame들 중 중요하다고 생각되는 프레임들의 sparse subset을 선택하는 것이다.

- **비지도 학습을 통한 비디오 요약**

  자동화된 Key frame 선택 모델을 만드는 것은 주로 지도 학습(Supervied Learning)을 통하여 이루어져왔다. 하지만 이는 사람이 직접 비디오 프레임에 어노테이션(참값)을 추가하여 만들어야 한다는 단점과, 이 과정에서 비디오 요약이 비디오의 분야(domain)에 크게 종속된다는 단점이 있다.(ex 군사 영상, 홈케어 영상) 이를 극복하기 위한 하나의 방법으로 비지도 학습을 통한 비디오 요약 모델이 제시되었다. 

- **LSTM(Long Short-Term Memory)**

  자동화된 비디오 요약 모델을 학습시키기 위한 가장 기본적인 네트워크 구조는 LSTM이다. LSTM은 일련의 비디오 영상 시퀀스를 입력으로 받아서, 각각의 프레임에 대한 importance score를 출력하는 목적으로 주로 사용된다.
  LSTM은 many to one, one to many, many to many 등의 방식으로 응용될 수 있으며, 위와 같은 상황에서는 many to many 방식이 유용하다.
  아래에서는 여러 개의 LSTM이 각 모델에서 어떻게 사용되는지 자세히 설명한다.


## Base Model: SUM-GAN

### ([Unsupervised Video Summarization with Adversarial LSTM Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.pdf/) )

- **SUM-GAN 모델 개요**

  위의 논문에서는 비지도 학습을 통한 비디오 요약을 위해 SUM-GAN이라는 모델을 제시한다. 본 모델은 비디오 시퀀스를 요약하는(각 프레임 별 가중치를 곱하여) 기능을 하게 설계되었으며, 요약된 비디오와 원본 비디오 간의 차이를 최소화하는 것을 목표로 학습된다.
  학습 과정에서 사용되는 주요 구성요소로는 요약기(Summarizer)와 그 내부의 VAE(Variational Auto-Encoder), 그리고 판별기(Discriminator)가 있다.
  아래는 전체적인 모델의 모습이다.
  ![](https://user-images.githubusercontent.com/62598121/120631026-5e410e80-c4a2-11eb-9242-bf11b7b9d779.PNG =300x300)

- **Main Components**

  - **Forward**
    - 입력 비디오에 대한 프레임별 Deep feature vectors (**x**) 추출
    - selector-LSTM(sLSTM)을 통해 프레임별 importance scores (**s**) 계산
    - original features (**x**)와 **x**에 가중치 **s**를 곱한 summary (**x'**)가 각각 encoder-LSTM(eLSTM)의 입력으로 주어짐
    - eLSTM를 통과한 **x**, **x'** 는 latent vector인 **e**, **e'** 로 압축됨
    - **e**, **e'**는 다시decoder-LSTM(dLSTM)의 입력으로 주어지고,  dLSTM은 비디오 시퀀스를 재구성한다.
    - 마지막으로 재구성된 비디오 시퀀스와 원본 비디오 시퀀스를 구별해내기 위한 classifier-LSTM(cLSTM)를 통해 구별

  

- **Loss function** 
  Loss funcfion 다음과 같이 정의된다.

  - Loss of Gan: 구별자가 재구성된 원본 비디오와 요약된 비디오를 제대로 구별하지 못하는 정도
  - Reconstruction loss: 재구성된 원본 비디오와 요약된 비디오 간의 차이
  - Prior loss: VAE의 분포와 주어진 사전 확률 분포의 차이

  Loss for regulization

  - Sparsity loss: sLSTM으로부터 계산된 importance score의 평균과 주어진 요약 비율(0.3)의 차이

  





## SUM-GAN-sl

- SUM-GAN 의 불안정적인 학습 과정 개선, 평가 방식 개선, 데이터 sampling
- 전체적인 모델 개요
- 주요 포인트 - incremental trainning, sampling
- 결과
  - 평가 방식
  - 벤치마킹
  - 데이터셋 소개
  - 학습 과정 / 학습 결과



## SUM-GAN-AAE

- SUM-GAN-sl 보다 더 안정적인 학습 과정
- 전체적인 모델 개요
- 주요 포인트 - Attention 도입
- 결과
  - 벤치마킹
  - 학습 과정 / 학습 결과 (SUM-GAN-sl 과 비교)



## 개선점

