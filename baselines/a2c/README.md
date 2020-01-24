
# A2C

> 소행 강점 평가(Advantage Actor Critic)

- 논문 원본: https://arxiv.org/abs/1602.01783  
- 기준선 블로그 글: https://blog.openai.com/baselines-acktr-a2c/  
- `python -m baselines.run --alg=a2c --env=PongNoFrameskip-v4` 아타리 퐁(Pong)으로 1천만 보 = 4천만 장면에 대해 알고리즘 실행. 더많은 선택을 위해 도움말(`-h`)을 보라.  
- 또한 줄기저장소의 [README.md](../../README.md#training-models)를 주목하라.  

## 파일들

- `run_atari`: 알고리즘 실행에 사용되는 파일.  
- `policies.py`: A2C 구조의 다른 판을 담고있다(MlpPolicy, CNNPolicy, LstmPolicy...).  
- `a2c.py`:  
  - Model : step_model(표집모형) 과 train_model(벼림모형)을 초기화하는데 사용하는 클래스.  
  - learn : A2C 알고리즘에 대한 주 진입지점. `a2c` 알고리즘을 사용하여 주어진 환경에서 주어진 망으로 정책을 벼림한다.  
- `runner.py`: 경험 덩어리 생성에 사용하는 클래스.  
