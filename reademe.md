**SoftwareConvergenceCapstoneDesign2021**

## Unsupervised Video Summarization with Adversarial LSTM Networks 모델 구현 및 성능 향상에 대한 연구

* **2021-1 소프트웨어융합캡스톤디자인** 
* **[김영웅 (응용수학과)](https://khero97.tistory.com/)**



### 목차

##### 1. 개요

##### 2. SUM-GAN

##### 3. SUM-GAN-sl

##### 4. SUM-GAN-AAE







## 1. 개요

#### **Video Summarization**

- **필요성**
  효율적인 비디오 분석과 자료 조사를 위해 비디오 요약을 자동화하는 기술은 많은 응용분야에서 필요하다. 응용하는 분야가 어떤 것인가에 따라 비디오 요약의 정의는 다양하다. 비디오 내의 중요한 정지 영상을 여러 개 추출하거나(Key frame selection), 비디오 내의 비슷한 정지 영상들을 제거하여 영상을 거의 똑같이 표현함과 동시에 비디오 길이를 줄이는 방법 등이 있다. 여기서 말하고자하는 비디오 요약이란 key frame selection을 의미하며, 전체 비디오 frame들 중 중요하다고 생각되는 프레임들의 sparse subset을 선택하는 것이다.



- **비지도 학습을 통한 비디오 요약**
  자동화된 Key frame 선택 모델을 만드는 것은 주로 지도 학습(Supervied Learning)을 통하여 이루어져왔다. 하지만 이는 사람이 직접 비디오 프레임에 어노테이션(참값)을 데이터셋을 구축해야 한다는 단점과, 이 과정에서 비디오 요약이 비디오의 분야(domain)에 크게 종속된다는 단점이 있다.(ex 군사 영상, 홈케어 영상) 이를 극복하기 위해 비지도 학습을 통한 비디오 요약 모델이 제시되었다. 



- **LSTM(Long Short-Term Memory)**
  자동화된 비디오 요약 모델을 학습시키기 위한 가장 기본적인 네트워크 구조는 LSTM이다. LSTM은 일련의 비디오 영상 시퀀스를 입력으로 받아서, 각각의 프레임에 대한 importance score를 출력하는 목적으로 주로 사용된다. LSTM은 many to one, one to many, many to many 등의 여러 방식으로 사용될 수 있기 때문에, 비디오 요약 모델에서 여러 번 사용되기도 한다.







## 2. Base Model: SUM-GAN

#### Unsupervised Video Summarization with Adversarial LSTM Networks [(2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.pdf/) 

- **SUM-GAN 모델 개요**

  위의 논문에서는 비지도 학습을 통한 비디오 요약을 위해 SUM-GAN이라는 모델을 제시한다. 본 모델은 비디오 시퀀스를 요약하는(각 프레임 별 가중치를 곱하여) 기능을 하게 설계되었으며, 요약된 비디오와 원본 비디오 간의 차이를 최소화하는 것을 목표로 학습된다.

  학습 과정에서 사용되는 주요 구성요소로는 요약기(Summarizer)와 판별기(Discriminator)가 있다. 요약기는 selector LSTM, encoder LSTM, decoder LSTM를 포함하고 있그며, 판별기는 clssifier LSTM을 포함하고 있다.

  특히, 요약기의 decoder LSTM과 판별기의 classifier LSTM은 서로 적대적인 관계로 GAN(Generative-Adversarial Network)를 구성하고 있으며 각각 Generator와 Discriminator의 역할을 한다.

  아래는 전체적인 모델의 모습이다.

  <img src="C:\Users\duddn\Desktop\모델상세.PNG" alt="모델상세" style="zoom:50%;" />

  

- **Main Components**

  - **Forward**
    - 입력 비디오에 대한 프레임별 Deep feature vectors (**x**) 추출
    - selector-LSTM(sLSTM)을 통해 프레임별 importance scores (**s**) 계산
    - original features (**x**)에 가중치 **s**를 곱한 summary가 encoder-LSTM(eLSTM)의 입력으로 주어짐
    - eLSTM를 통과한 입력은 latent vector인 **e**로 압축
    -  **e**는 다시decoder-LSTM(dLSTM)의 입력으로 주어지고,  dLSTM은 비디오 시퀀스를 재구성
    - 마지막으로 재구성된 비디오 시퀀스와 원본 비디오 시퀀스를 구별해내기 위한 classifier-LSTM(cLSTM)를 통해 구별

    

  - **Loss function** 

    - **Loss of Gan**: 구별자가 재구성된 원본 비디오와 요약된 비디오를 제대로 구별하지 못하는 정도
    - **Prior loss**: VAE가 재구성하는 비디오 시퀀스의 분포와 주어진 사전 확률 분포의 차이
    - **Reconstruction loss**: 재구성된 원본 비디오와 요약된 비디오 간의 차이

    - **Sparsity loss**: sLSTM으로부터 계산된 importance score의 평균과 주어진 요약 비율(0.3)의 차이

    

  - **Additional regulization term**

    - 더 의미있는 요약을 위해 무작위로 요약하는 경우를 고려하여 패널티를 줌 (GAN loss에서 사용)

      <img src="C:\Users\duddn\Desktop\forward.PNG" alt="모델상세" style="zoom:75%;" />



- **Training Algorithm**

  ![image-20210603195458220](C:\Users\duddn\AppData\Roaming\Typora\typora-user-images\image-20210603195458220.png)

  

- **Datasets**
  - SumMe 
    - 25개의 다양한 주제의 비디오
    - 카메라 시점 1인칭 또는 3인칭 
    - 비디오 길이 약 1.5분 ~ 6.5분
    - 프레임별 importance scores 제공
  - TvSum
    - 10개의 주제, 각각 5개씩 총 50개 비디오 
    - 카메라 시점 1인칭 또는 3인칭 
    - 비디오 길이 1분 ~ 5분
    - 프레임별 importance scores 제공
  - OVP(Youtube)
    - 논문 [Vsumm: A mechanism designed to produce ´ static video summaries and a novel evaluation method] 에서 사용된 OVP 비디오 50개와 YouTube에서 수집한 다양 한 주제의 50개의 비디오
    - 비디오 길이 1분 ~ 10분



- **Evaluation Setup**
  - 평가는 keyshot-based metric을 사용하였다. keyshot-based metric을 간단히 설명하면 다음과 같다.
    - KTS(Kernel based Temporal segmentation)를 사용하여 keyshot이라는 구간으로 이루어진 영상의 파티션을 구함
    - keyshot 중 높은 importance score를 가진 keyshot을 knapsack 알고리즘을 통해 선택 (총 영상 길이의 15% 이하)
    - user-annotated keyshots 와 model-selected keyshots 간 intersection, union의 비율로 precision과 recall을 구해 F-1 score를 계산



- **Results**

  - SUM-GAN 모델과 여러 가지 Variants model의 성능

    ![selfbench](C:\Users\duddn\Desktop\selfbench.PNG)







## 3. SUM-GAN-sl

- SUM-GAN 의 불안정적인 학습 과정 개선, 평가 방식 개선, 데이터 sampling
- 전체적인 모델 개요
- 주요 포인트 - incremental trainning, sampling
- 결과
  - 평가 방식
  - 벤치마킹
  - 데이터셋 소개
  - 학습 과정 / 학습 결과





## 4. SUM-GAN-AAE

- SUM-GAN-sl 보다 더 안정적인 학습 과정
- 전체적인 모델 개요
- 주요 포인트 - Attention 도입
- 결과
  - 벤치마킹
  - 학습 과정 / 학습 결과 (SUM-GAN-sl 과 비교)



### 개선점

