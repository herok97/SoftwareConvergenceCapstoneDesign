**SoftwareConvergenceCapstoneDesign2021**

# Unsupervised Video Summarization with Adversarial LSTM Networks 모델 구현 및 성능 향상에 대한 연구

* **2021-1 소프트웨어융합캡스톤디자인** 
* **[김영웅 (응용수학과)](https://khero97.tistory.com/)**

<br>

## 목차

#### 1. 개요

#### 2. SUM-GAN

#### 3. SUM-GAN-sl

#### 4. SUM-GAN-AAE

#### 5. 모델 학습 재현 및 개선사항

<br>



## 1. 개요

- ### 비디오 요약(Video Summarization)의 필요성
  

여러 응용분야에서, 효율적인 비디오 분석과 자료 조사를 위한 자동화된 비디오 요약 기술이 요구되고 있다 . 응용하는 분야가 어떤 것인가에 따라 비디오 요약의 정의는 다양한데, 비디오에서 중요한 프레임들을 선택하거나(Key frame selection), 비디오 내의 유사한 프레임들을 제거하여 원본영상과 유사하게 표현함과 동시에 비디오 길이를 줄이는 방법 등이 있다. 여기서 말하고자하는 비디오 요약이란 key frame selection을 의미하며, 전체 비디오 frame들 중 중요하다고 생각되는 프레임들의 sparse subset을 선택하는 것이다.

<br>

- ### 비지도 학습을 통한 비디오 요약
  

자동화된 Key frame 선택 모델을 만드는 것은 주로 지도 학습(Supervied Learning)을 통하여 이루어져왔다. 하지만 이는 사람이 직접 비디오 프레임에 어노테이션(참값)을 데이터셋을 구축해야한다. 또한, 비디오 요약이 비디오의 분야(domain)에 크게 종속된다는 단점이 있다.(ex 군사 영상, 홈케어 영상) 이를 극복하기 위해 비지도 학습을 통한 비디오 요약 모델이 제시되었다. 

<br>

- ### 주제 선택과 이전 연구

선택한 주제는 Unsupervised Video Summarization with Adversarial LSTM Networks [(2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.pdf/)에서 제시된 모델의 성능을 향상시키는 것이다. 이를 위해 Base Model인 SUM-GAN 모델과, 이후 발전된 형태의 SUM-GAN-sl, SUM-GAN-AAE에 대한 스터디를 진행하였다.

<br>

<br>

## 2. Base Model: SUM-GAN

### Unsupervised Video Summarization with Adversarial LSTM Networks [(2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.pdf/) 

- ### SUM-GAN 모델 개요

  위의 논문에서는 비지도 학습을 통한 비디오 요약을 위해 SUM-GAN이라는 모델을 제시한다. 본 모델은 비디오의 각 프레임별로 Importance score를 부여함으로써 비디오를 요약한다. 모델은 크게 요약기(Summarizer)와 판별기(Discriminator)로 이루어진다. 요약기는 주어진 원본 비디오를 요약하는 역할을 수행하고, 판별기는 요약된 비디오와 원본 비디오를 구별해내는 역할을 수행한다.  요약된 비디오는 VAE(Variational Auto-encoder)를 거쳐 원본 비디오와의 차이가 작아지도록 재구성(reconstruction)되며, 이후 판별기를 속이는 것을 목표로 학습된다. 이 과정에서 VAE의 Decoder와 판별기는 GAN(Generative Adversarial Network) 구조를 형성한다.

  아래는 전체적인 모델의 모습이다.

  <img src="https://user-images.githubusercontent.com/62598121/121049394-9de55e80-c7f2-11eb-9c54-c83a8ac4386d.PNG" alt="모델상세" style="zoom:50%;" />

  <br>

- ### Main Components

  - **Forward**
    - 입력 비디오에 대한 프레임별 Deep feature vectors (**x**) 추출
    - selector-LSTM(sLSTM)을 통해 프레임별 importance scores (**s**) 계산
    - original features (**x**)에 가중치 **s**를 곱한 summary가 encoder-LSTM(eLSTM)의 입력으로 주어짐
    - eLSTM를 통과한 입력은 latent vector인 **e**로 압축
    -  **e**는 다시decoder-LSTM(dLSTM)의 입력으로 주어지고,  dLSTM은 비디오 시퀀스를 재구성
    - 마지막으로 재구성된 비디오 시퀀스와 원본 비디오 시퀀스를 구별해내기 위한 classifier-LSTM(cLSTM) 통과

<br>
    
- **Loss function** 
  
    - **Loss of Gan**: 구별자가 재구성된 원본 비디오와 요약된 비디오를 제대로 구별하지 못하는 정도
    - **Prior loss**: VAE가 재구성하는 비디오 시퀀스의 분포와 주어진 사전 확률 분포의 차이
  - **Reconstruction loss**: 재구성된 원본 비디오와 요약된 비디오 간의 차이
  
  - **Sparsity loss**: sLSTM으로부터 계산된 importance score의 평균과 주어진 요약 비율(0.3)의 차이
  
  <br>
  
- **Additional regulization term**
  
  - 더 의미있는 요약을 위해 무작위로 요약하는 경우를 고려하여 패널티를 줌 (GAN loss에서 사용)
  
      <img src="https://user-images.githubusercontent.com/62598121/121049679-e4d35400-c7f2-11eb-86f3-3bb489cfafff.PNG" alt="모델상세" style="zoom:75%;" />

<br>

- #### **Training Algorithm**

  ![image-20210603195458220](https://user-images.githubusercontent.com/62598121/121049754-f4eb3380-c7f2-11eb-9565-4d4be9abf493.PNG)

  <br>

- ### Datasets

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

<br>

- ### Evaluation Setup(with single gtscore)

  - 평가는 keyshot-based metric을 사용하였다. keyshot-based metric을 간단히 설명하면 다음과 같다.
    - KTS(Kernel based Temporal segmentation)를 사용하여 keyshot이라는 구간으로 이루어진 영상의 파티션을 구함
    - keyshot 중 높은 importance score를 가진 keyshot을 knapsack 알고리즘을 통해 선택 (총 영상 길이의 15% 이하)
    - user-annotated keyshots 와 model-selected keyshots 간 intersection, union의 비율로 precision과 recall을 구해 F-1 score를 계산

<br>

- ### Results

  - SUM-GAN 모델과 여러 가지 Variants model의 성능 (single gtscore)

    ![selfbench](https://user-images.githubusercontent.com/62598121/121049840-0a605d80-c7f3-11eb-85d1-68e5e6272840.PNG)

<br>

<br>

## 3. SUM-GAN-sl

### A Stepwise, Label-based Approach for Improving the Adversarial Training in Unsupervised Video Summarization [(2019)](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/62042/Apostolidis%20A%20Stepwise,%20Label%202019%20Accepted.pdf?sequence=15/) 

- ### 개요

  본 논문에서는 적대적 방법의 비지도학습을 통한 비디오 요약의 효율성과 성능을 향상시키기 위한 방법을 제시하였다. 그 시작점은 위에서 설명한 SUM-GAN 모델이다. 제안된 모델에서는 기존 모델의 앞단에 linear compression 층을 추가하고 뒤에서 사용되는 LSTM들의 hidden size를 조절하여 학습 파라미터를 크게 줄였으며, 학습과정에서 stepwise, label-based learning process를 사용하여 Generator와 Discriminator의 적대적 관계 망을 효율적으로 학습할 수 있도록 했다. 

  추가적으로, 새로 개선된 모델을 제안함과 동시에 비디오 요약에서 자주 사용되는 데이터셋인 SumMe, TvSum에 대한 몇 가지 실험 결과와 개선된 모델 성능 평가 방법을 제시한다.

  <br>

 

- ### Preliminary Study on Datasets

  - **Annotation information**
    - <u>SumMe</u>
      - 15-18명의 사람에 의해 user-annotated importance score이 <u>fragment</u> 단위로 부여되어 있음
      - 추가로 각 frame 별로 single importance score 제공 (각 frame이 속한 fragment의 점수 평균 계산)
    - <u>TvSum</u>
      - 20명의 사람에 의해  user-annotated importance score이 <u>frame</u> 단위로 부여되어 있음
      - 추가로 각 frame 별로 single importance score 제공 (각 frame별로 20명이 부여한 점수의 평균 계산)

  <br>

  - **General Performance**

    <img src="https://user-images.githubusercontent.com/62598121/121049930-219f4b00-c7f3-11eb-91dd-076a645061a2.PNG" alt="score2" style="zoom:50%;" />
    위의 표를 통해 기존의 비디오 요약 모델들을 살펴보면 일반적으로 SumMe 데이터에 대한 성능보다 TvSum에 대한 성능이 항상 더 좋음

  <br>

  - **Randomly generated summary 성능 평가**
    - 균등 분포에서 추출한 importance score로 비디오 요약 후 , 100회 반복

      <br>

  - **Human performance 평가**

    - 어노테이션 작업자 중 특정 작업자를 선택하여, 다른 작업자들의 어노테이션을 기준으로 성능평가했을 때의 성능 평가

    - 비디오 요약은 어노테이션 작업자들의 주관에 상당히 종속적이라는 사실 확인

      <br>

  - **Best human-generated summary 평가**

    - 데이터셋 내의 각 비디오에 대해 가장 높은 성능을 낸 human-generated summary 성능 평가

  <img src="https://user-images.githubusercontent.com/62598121/121050079-3e3b8300-c7f3-11eb-933d-3bb0afed885d.PNG" alt="score2" style="zoom:50%;" />

  - **The result of insights for dataset**

    논문에서는 SumMe, TvSum에 대해 두 가지 평가 방법인 Average(모든 비디오에 대한 평균), Max(가장 높은 점수를 받은 비디오)에 따른 F-score를 제시한다. 
    Best Possible은 "각 비디오별로, 가장 높은 F-score를 기록한 작업자들만을 골라 평가했을 때의 F-score"로 비디오 요약 성능의 상한선으로 해석할 수 있다.
    Average 평가 방법의 경우에는 SumMe: 44.7 , TvSum: 64.7 로 상한선이 정해지기 때문에, 모든 사람들의 선호를 만족시킬 수 있는 이상적인 비디오 요약은 존재하지 않는다는 것을 알 수 있다.
    하지만 Max 평가 방법의 경우를 생각하면, 즉, 각 비디오에 대해 작업자 중 어느 한 사람의 선호를 흉내낸다면 최대 100의 F-score를 받을 수 있다.
    따라서 논문에서는 Max 평가 방법을 사용하는 것이, 비디오 요약을 평가하는 척도로 더 적합하다고 결론지었다.

    하지만 더 정확한 비교를 위해 기존의 방법과 제시한 방법 모두 테스트하여 결과를 제시했다.

    

<br>

- ### Proposed Approach

  - **Linear compression layer 추가**
    모델의 구조에서 Selector LSTM(sLSTM)의 앞에 Linear compression layer를 추가하여 입력 시퀀스의 차원을 줄였다. 학습 속도 면에서 상당한 효율이 증가했다.

    <br>

  - **Training imcrementally**
    3-step의 점진적 학습 방법을 사용했다.

    - First forward pass
      전체 모델을 진행하며 reconstruction loss, prior loss, sparsity loss를 계산한 뒤 sLSTM과 eLSTM, Linear Compression layer 만을 업데이트 한다.

    - Second forward pass
      전체 모델을 진행하며 reconstruction loss, GAN loss를 계산한 뒤 dLSTM, Linear Compression layer 만을 업데이트 한다.

    - Third foward pass
      전체 모델을 진행하며 GAN loss를 계산한 뒤 cLSTM, Linear Compression layer 만을 업데이트 한다.

      ![3step](https://user-images.githubusercontent.com/62598121/121050168-50b5bc80-c7f3-11eb-9f12-ee2c2c8ee7ec.PNG)

    <br>

  - **Training GAN with stepwise & label-based manner**
    GAN loss를 제거하고 Discriminator loss를 아래와 같이 두 개로 나누어 도입했다. 

    ![new_loss](https://user-images.githubusercontent.com/62598121/121050219-5b705180-c7f3-11eb-9c1d-cd7a6572bb73.PNG)

    또한, Generator를 위한 loss를 별도로 아래와 같이 도입했다.![genloss](https://user-images.githubusercontent.com/62598121/121050273-662ae680-c7f3-11eb-90d1-5d4aac5120a3.PNG)

  - vanilla GAN을 학습시킬 때 도입된 Binary Cross Entropy(BCE)는 더 다양한 representation을 만들기 위해 사용된 것인데, 본 모델에서는 요약된 비디오의 다양성이 비교적 중요하지 않기 때문에 이를 MSE loss로 변경하였다고 설명한다.  

  - 또한 기존 SUM-GAN 모델에서 regulization을 위해 제안된 randomly generated summary는 제거되었다. 이는 BCE를 사용하지 않고 MSE를 사용하기 때문에 굳이 이용할 필요가 없다고 논문은 설명한다.

    최종적으로 제안된 학습 과정은 아래와 같다.

    ![forward2](https://user-images.githubusercontent.com/62598121/121050319-6f1bb800-c7f3-11eb-93c0-0313bb3d316d.PNG)

<br>

- #### **Implementation Details**

  - Dataset: SumMe, TvSum
  - Feature vector: output of pool5 layer of GoogleNEt
  - Output dimenssion of Linear compression layer & hidden unit of each LSTM: 500
  -  Optimizer: Adam (default)
  - Learning rate: 0.00001 for discriminator, 0.0001 for else.
  - Validation set: standard 5-fold cross validation approach
  - Hyperparameter for Sparsity: 0.5 (best)

<br>

- ### **Results**

![slresult](https://user-images.githubusercontent.com/62598121/121050361-78a52000-c7f3-11eb-8351-a504af302bb2.JPG)

<br>

<br>

## 4. SUM-GAN-AAE

### A Stepwise, Label-based Approach for Improving the Adversarial Training in Unsupervised Video Summarization [(2019)](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/62307/Apostolidis%20Unsupervised%20Video%20Summarization%202019%20Accepted.pdf?sequence=10) 

- ### 개요

  본 논문에서는 비지도학습을 통한 비디오 요약 모델 학습을 위해 Attention mechanism을 앞선 SUM-sl 모델에 추가하여 성능을 개선시키는 두 가지 방법을 제시한다. 하나는 기존 모델의 VAE (Variational auto-encoder)구조 안에 attention layer 추가하는 것(SUM-GAN-VAAE)이고, 다른 하나는 기존 모델의 VAE 구조를 deterministic attention auto-encoder로 대체하는 것(SUM-GAN-AAE)이다.  이를 통해, 기존 모델의 학습 속도와 안정성을 향상시켰으며, Base model인 SUM-GAN에 비해 상당한 성능 향상을 보였다. 결과적으로는 SUM-GAN-AAE가 더 좋은 성능을 보여주었으므로 SUM-GAN-VAAE에 대한 자세한 설명은 생략한다.


<br>

- ### Introducing an attention auto-encoder(AAE)

  ![AAE](https://user-images.githubusercontent.com/62598121/121050428-85297880-c7f3-11eb-9a74-91fef8fb8739.PNG)
  
  Attention mechanism을 이용하기 위한 두 번째 방식은 Attention auto-encoder(AAE)이다. 아래 그림에서 추가된 Attention LSTM은 현 시점(t)의 Encoder의 출력과 이전 시점(t-1)의 decoder의 hidden state를 입력으로 받아서 Attention energe vector를 출력한다. 이 출력은 score function과 softmax layer를 거쳐 현 시점의 frame과 전체 비디오 간의 상관 관계(correlation)를 수치화한다. 이 출력을 attention weight vector라고 하며, encoder의 출력과 곱해져(MM) decoder의 다음 입력으로 주어진다.
  
  ![(AAE2)](https://user-images.githubusercontent.com/62598121/121050462-8c508680-c7f3-11eb-8c28-a323e88a257d.PNG)
  
  <br>
  
- ### Implementation details

  - Dataset: SumMe, TvSum
  - Sampling: 2fps
  - Feature vector: output of pool5 layer of GoogleNet
  - Output dimenssion of Linear compression layer & hidden unit of each LSTM: 500
  - Optimizer: Adam (default)
  - Learning rate: 0.00001 for discriminator, 0.0001 for else.
  - Validation set: standard 5-fold cross validation approach
  - Hyperparameter for Sparsity: 0.5 (best)
  
  <br>
  
- ### Results

  이전 연구들과 공정한 비교를 위해 single ground-truth summary에 대한 F-Score를 수행하였으며, 결과는  다음과 같다.

  ![result_aae](https://user-images.githubusercontent.com/62598121/121050498-95d9ee80-c7f3-11eb-9881-09fe7fe1a818.JPG)

<br>

<br>

## 5. 모델 학습 재현 및 개선사항

- ### 모델 학습 재현

  이후 모델에 개선사항을 제시하기 위해서, SUM-GAN-AAE 모델 학습을 똑같이 재현해보고 결과를 살펴보았다. 논문에서의 학습과정과 유일한 차이점은 학습 데이터와 테스트 데이터를 4:1로 split 할 때 발생한 무작위성이다. 논문에서 제시한 것과 똑같이 총 5번의 random split을 진행하여 평균 F-score를 구하였다. 결과는 아래와 같다.

  **SumMe**: **53.55%** 
  **TvSum**: **62.39%** 

  ![fsc](https://user-images.githubusercontent.com/62598121/121050545-9ffbed00-c7f3-11eb-8792-99a131bba756.JPG)

  논문의 결과와의 약간의 차이가 발생했지만 이러한 차이는 Data split 과정에서 발생한 것이라고 생각된다. 
  또한, 논문에서 설명한 것과 동일하게 GAN 구조 학습시 loss function이 Stable하게 수렴하는 것을 확인할 수 있었다.

  ![tgen](https://user-images.githubusercontent.com/62598121/121050591-a8ecbe80-c7f3-11eb-8466-bf33eaf3acb5.JPG)



- ### 개선사항

  아래는 SUM-GAN-AAE 모델을 TvSum 데이터로 학습한 후 한 비디오에 대해 테스트한 결과이다. 모델은 비디오를 요약하기 위해 각 프레임별로 importance score를 부여하는데, 비디오의 처음과 끝에 아주 높은 점수가 부여되는 현상이 있었다. 물론 처음과 마지막이 영상에서 중요한 프레임이기는 하지만, 너무 편차가 큰 점수로 인해 중간에 있는 importance score가 무의미해질 수 있다고 판단했다.

  ![ip](https://user-images.githubusercontent.com/62598121/121050626-b1dd9000-c7f3-11eb-853b-326174f094a6.JPG)
  
  따라서 기존의 Sparsity loss를 수정하고 새로운 std(standard deviation) loss를 추가했다.

  먼저 비디오를 구성하고 있는 프레임을 세 구간으로 균등하게 나누었다. 이후 Sparsity loss를 통해 각 구간의 importance scores의 평균이 동일하도록 하였으며, std loss를 통해 각 구간의 표준편차 또한 동일해지도록 설정했다.

  

  ![spa](https://user-images.githubusercontent.com/62598121/121050670-bbff8e80-c7f3-11eb-996a-df7fbf21828b.JPG)

  ![std](https://user-images.githubusercontent.com/62598121/121050697-c3269c80-c7f3-11eb-877c-f54f79edf541.JPG)

  

  아래는 m=0.5, sigma=0.01로 설정하여 학습시켰을 때 동일한 비디오에 대한 테스트 결과이다. 의도했던 대로 importance score가 비디오의 처음과 끝에만 몰려있지 않게 되었다.

  

  ![ip2](https://user-images.githubusercontent.com/62598121/121050733-cae64100-c7f3-11eb-9b03-bc376934f4ee.JPG)

  전체 비디오에 대한 F-score또한 아래와 같은 결과를 보였다. SumMe 데이터의 경우 소폭 상승하였고 TvSum 데이터의 경우 소폭 하락하였다. 즉, 비디오 요약이 처음과 끝에 의존하지 않음과 동시에 F-score 면에서 큰 성능하락을 보이지 않았다고 판단할 수 있었다.

  **SumMe**: **54.07%** 
  **TvSum**: **61.88%** 

  이후 sigma 값을 조절해가며 실험을 반복했고 다음과 같은 결과를 얻을 수 있었다.

  
