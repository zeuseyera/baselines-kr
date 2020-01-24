
# ACKTR

> 크로네커-지수 신뢰영역을 사용하여 소행평가  
> Actor Critic using Kronecker-factored Trust Region  

- 논문 원본: https://arxiv.org/abs/1708.05144  
- 기준선 블로그 글: https://blog.openai.com/baselines-acktr-a2c/  
- `python -m baselines.run --alg=acktr --env=PongNoFrameskip-v4` 아타리 퐁(Pong)으로 1천만 보 = 4천만 장면에 대해 알고리즘 실행. 더많은 선택을 위해 도움말(`-h`)을 보라.  
- 또한 줄기저장소의 [README.md](../../README.md#training-models)를 주목하라.  

## 연속소행 행위 ACKTR

ACKTR 코드는 단일(discrete)소행과 연속소행 행위를 균일하게 처리하기 위해 재구성 했다. 원본 판에서는, 단일(discrete)소행과 연속소행 행위는 거의 겹치지 않는 다른 코드(`actkr_disc.py` 와 `acktr_cont.py`)로 처리되었다.  
만약 연속소행 행위를 위한 `acktr` 원본판에 관심이 있다면, `old_acktr_cont` 가지(branch)를 사용하라.  
원본코드는 재구성 판보다 다관절접합(mujoco) 작업에서 더 잘 수행된다; 우리는 여전히 그 이유를 조사하고 있다.  


---

# ACKTR 기준선 블로그

출처:
 - https://openai.com/blog/baselines-acktr-a2c/

> ACKTR는 연속적인 제어작업을 배울수 있다, 로봇팔을 목표위치로 옮기는 것과 같은, 아주 낮은 해상도 화소를 입력으로(왼쪽)  

| [![reacher](https://img.youtube.com/vi/7acartjPpZ0/0.jpg)](https://www.youtube.com/watch?v=7acartjPpZ0) | [![raw_2](https://img.youtube.com/vi/A-F5nZ1aIfs/0.jpg)](https://www.youtube.com/watch?v=A-F5nZ1aIfs) |  
| --- | --- |  

ACKTR("actor"로 발음됨, 크로네커-지수 신뢰영역 사용 소행평가)는 토론토대학교와 뉴욕대학교의 연구원들에 의해 개발되었다, 그리고 OpenAI의 우리는 기준선 구현을 발표하기 위해 그들과 협업했다. 저자는 모사 로봇(입력으로 화소를, 그리고 연속소행 행위)과 아타리 똘마니(입력으로 화소를, 그리고 단일(discrete)소행 행위)를 위한 제어정책을 벼림하기 위해 ACKTR를 사용한다.  

ACKTR는 세가지 고유기법을 결합했다: 소행-평가 기법, 더 일관된 개선을 위한 신뢰영역 최적화, 그리고 표집효율과 확장성을 개선하기 위한 분산 크로네커 지수화.  

## 표집과 계산 효율

머신러닝 알고리즘의 경우, 두개의 비용을 고려하는것이 중요하다: 표집 복잡성과 계산 복잡성. 표집 복잡성은 똘마니와 환경 사이의 상호작용 보(timestep) 수를 가리킨다, 그리고 계산 복잡성은 반드시 수행해야 하는 수치 연산의 양을 가리킨다.  

ACKTR는 A2C 같은 1차 방법보다 표집 복잡성이 더 우수하다, 때문에 이것은 기울기 방향(또는 ADAM 처럼 재조정된 판)이 아닌, 자연스러운 기울기 방향으로 한 단계를 취한다. 자연적인 기울기는 KL-발산을 사용하여 측정된 것으로, 망의 출력분포에서 변경 단위별 목표에서 가장 큰 개선을 (즉시)달성하는 참여공간의 방향을 제공한다. KL-발산을 제한함으로써, 우리는 새로운 정책이 이전의 정책과 근본적으로 다르게 행동하지 않는것을 보장한다, 이것은 성능이 붕괴되는 원인이 될수있다.  

계산 복잡성에 관해서는, ACKTR에 의해 사용되는 KFAC 갱신은 표준 기울기 갱신보다 갱신 단계당 단지 10~25% 더 비싸다. 이것은 TRPO 같은 기법과 대조된다(즉, 헤시안없이 최적화), 이것은 더 비싼 결합(conjugate)-기울기 계산이 요구된다.  

다음 동영상에서는 Q-Bert 놀이를 해결하기 위해 ACKTR로 벼림된 똘마니와 A2C로 벼림된 똘마니 간의 다른 시간간격을 배교해 볼수 있다. ACKTR 똘마니가 A2C 로 벼림된 것보다 높은 점수를 얻는다.

> ACKTR(우측) 로 벼림된 똘마니가 다른 알고리즘으로 벼림된 똘마니보다 짧은 시간에 더 높은 점수를 얻는다, A2C(좌측) 같은.  

| [![A2C](https://img.youtube.com/vi/D6OMH8DLcVY/0.jpg)](https://www.youtube.com/watch?v=D6OMH8DLcVY) | [![ACKTR](https://img.youtube.com/vi/LDw39LQqtwk/0.jpg)](https://www.youtube.com/watch?v=LDw39LQqtwk) |  
| --- | --- |  
| A2C | ACKTR |  

## 기준선과 비교막대

이 배포는 ACKTR의 공개인공지능 기준선 배포를 포함한다, 뿐만 아니라 A2C 배포 포함.

우리는 또한 다양 작업의 ACKTR 대비 A2C, PPO 와 ACER 를 평가한 비교막대를 게시한다. 다음은 다른 알고리즘과 비교한 49개의 아타리 놀이에서 ACKTR의 성능을 보여주는 그림이다: A2C, PPO, ACER. ACKTR의 잠정참여(hyperparameter)는 벽돌깨기(Breakout), 한 놀이만 ACKTR 저자에 의해 조정되었다.  

![비교표](https://openai.com/content/images/2017/08/Pasted-image-at-2017_08_18-09_01-AM.png)

ACKTR 의 성능은 또한 덩어리(batch) 크기에도 잘 조정된다, 왜냐하면 각 덩어리 정보로 부터 추정기울기만 도출하지 않을, 뿐만 아니라 참여(parameter)공간에서 지역 곡률을 근사하기위한 정보를 사용하기 때문이다. 이 특징은 큰 덩어리(batch) 크기가 사용되는, 대규모 분산 벼림에 대해 특히 유리하다.

![벽돌깨기](https://openai.com/content/images/2017/08/WX20170817-220206@2x-3.png)

## A2C 와 A3C

A3C(비동기 소행 강점 평가: Asynchronous Advantage Actor Critic) 기법은 논문이 발표된 이후 매우 영향력이 있다. 이 알고리즘은 몇가지 핵심 구상을 결합한다:  

- 지정한 길이(가령, 20 보) 경험조각으로 동작하고, 이 조각을 사용하여 수익(return)과 강점(advantage) 함수의 추정값을 계산하여, 갱신하는 계획.  

- 정책과 가치 함수 서로간에 단을 공유하는 구조.  

- 비동기 갱신.  

이 논문을 읽고, 인공지능 연구원들은 어느것인지 궁금해 했다, 비동기가 성능개선을 이끄는 것인지(예들들어, "아마도 정규화(고루기) 또는 탐사가 제공하는 일부가 더해진 잡음인가?"), 아니면 CPU-기반 구현이 빠른 벼림을 가능하게 하는 구현 세부사항인지.  

비동기 구현의 대안으로, 연구원들은 동기하여 기록할수 있다는 것을 찾아냈다, 결정적인(deterministic) 구현은, 각 소행기(actor)에 대해 경험조각이 끝날때까지 기다린다, 소행기(actor) 모두의 평균을, 갱신을 수행하기 전에. 이 기법의 한가지 강점은 GPU를 효과적으로 사용할수 있다는 것이다, 이것은 큰 덩어리크기에서 잘 수행된다. 이 알고리즘은 자연스럽게 A2C라고 불린다, 소행강점평가(Advantage Actor Critic)의 줄임말.(이 용어는 여러 논문에서 사용되었다.)

우리의 비동기 구현보다 우리의 동기 A2C 구현이 보다 더 성능이 뛰어나다 - 우리는 비동기에 의해 발생하는 잡음이 성능상 이점을 제공한다는 어떤 증거도 보지 못했다. 이 A2C 구현은 하나의 GPU를 사용하는 기계일때 A3C 보다 더 비용이 효율적이다, 그리고 큰 정책을 사용하는 경우 CPU만 사용하는 A3C 구현보다 빠르다.  

우리는 A2C 사용 아타리 비교막대에 순전파 나선망과 LSTM 벼림을 위해 기준선에 코드를 포함시켰다.  

