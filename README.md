:kr: 강화학습 기준선(baseline)

- 이 저장소의 내용은 원본과 다름...  
- win10 에서 비스코드로 디버깅환경 추가( 아나콘다3 에 파이썬환경설정 필요 )  

출처:
 - https://github.com/openai/baselines

**상태:** 유지관리 (버그 수정 및 자잘한 갱신 예상됨)

<img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/openai/baselines.svg?branch=master)](https://travis-ci.org/openai/baselines)

# 기준선(Baselines)

공개인공지능 기준선은 고품질로 구현된 강화학습 알고리즘 모듬이다.

이러한 알고리즘의 복제, 개선은 이 연구소통(Research Community)을 통해 쉽게 할수있다, 그리고 새로운 아이디어를 식별할수 있고, 그리고 연구를위한 좋은 기준선을 만들 것이다. 우리가 구현하고 변형한 DQN은 발표된 논문의 점수와 거의 비슷하다. 우리는 이것이 새로운 구상(아이디어)을 추가할 수 있는 기준으로 사용될 것을 기대한다, 그리고 이것을 기반으로 새로운 기법을 비교하는 도구로 사용된다.

## 전제조건(Prerequisites)  

기준선은 개발 머리글로 파이썬3(>=3.5)이 필요하다. 또한 CMake, OpenMPI 와 zlib 씨스템패키지기 필요하다. 이것들은 다음처럼 설치할수 있다.

### 윈도우 10

- 아나콘다3를 설치하고 파이썬환경을 설정한 후 아래에서 필요한 패키지를 추가로 설치한다

### 우분투(Ubuntu )
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
    
### 맥 오에스 엑스(Mac OS X)

맥에 씨스템패키지를 설치하려면 [`Homebrew`](https://brew.sh)가 필요하다. `Homebrew`가 설치되었으면, 다음을 실행한다:

```bash
brew install cmake openmpi
```
    
## 가상환경(Virtual environment)

온전한 상태의 일반적인 파이썬 패키지에서, 다른 프로젝트의 패키지와 서로 간섭하지 않도록 가상환경(`virtualenvs`)을 사용하는 것이 좋다. pip 패키지(내장된)를 통해 가상환경(`virtualenvs`)을 설치할수 있다.

```bash
pip install virtualenv
```

가상환경(`virtualenvs`)은 실행할수 있는 파이썬 사본과 모든 파이썬 패키지를 가진 필수적인 폴더다. `venv`라는 파이썬3 가상환경(`virtualenvs`)을 생성하기 위해, 한번 실행한다.

```bash
virtualenv /path/to/venv --python=python3
```

가상환경(`virtualenvs`)을 구동(activate)하기 위해:

```
. /path/to/venv/bin/activate
```

더많은 가상환경(`virtualenvs`)의 가르침과 선택사항은 [여기](https://virtualenv.pypa.io/en/stable/)에서 찾을수 있다.

## 텐서플로우 판(Tensorflow versions)

지원하는 텐서플로우 줄기는 1.4에서 1.14 까지다. 텐서플로우 2.0 지원을 위해서는, `tf2` 가지를 사용하라.

## 설치(Installation)

- 저장소를 복제한다 그리고 현재위치를 `cd baselines` 으로 변경한다(cd: Change Directory):

    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```

- 만약 아직 설치된 텐서플로우가 없다면, 좋아하는 풍의 텐서플로우를 설치한다. 대부분의 경우, 이것을 사용할 것이다  

    ```bash 
    pip install tensorflow-gpu==1.14 # 만약 CUDA 호환 gpu 와 적절한 드라이버를 가지고 있다면
    ```

    또는  

    ```bash
    pip install tensorflow==1.14
    ```

    텐서플로우 1.14 를 설치하기 위해, 줄기에서 지원하는 텐서플로우 최신판이다. 더 자세한 [텐서플로우 설치 지침](https://www.tensorflow.org/install/)을 참조하라.

- 기준선(baselines) 패키지 설치

    ```bash
    pip install -e .
    ```

    윈도우10 에서 `pip install -e .`실행결과  

    ```shell
    Obtaining file:///G:/Work_GaeBal/baselines
    Requirement already satisfied: gym<0.16.0,>=0.15.4 in c:/anaconda3/envs/py35tf11/lib/site-packages (from baselines==0.1.6) (0.15.4)
    Requirement already satisfied: scipy in c:/anaconda3/envs/py35tf11/lib/site-packages (from baselines==0.1.6) (1.4.1)
    Collecting tqdm
      Downloading https://files.pythonhosted.org/packages/72/c9/7fc20feac72e79032a7c8138fd0d395dc6d8812b5b9edf53c3afd0b31017/tqdm-4.41.1-py2.py3-none-any.whl (56kB)
         |################################| 61kB 206kB/s
    Collecting joblib
      Downloading https://files.pythonhosted.org/packages/28/5c/cf6a2b65a321c4a209efcdf64c2689efae2cb62661f8f6f4bb28547cf1bf/joblib-0.14.1-py2.py3-none-any.whl (294kB)
         |################################| 296kB 731kB/s
    Requirement already satisfied: cloudpickle in c:/anaconda3/envs/py35tf11/lib/site-packages (from baselines==0.1.6) (1.2.2)
    Collecting click
      Downloading https://files.pythonhosted.org/packages/fa/37/45185cb5abbc30d7257104c434fe0b07e5a195a6847506c074527aa599ec/Click-7.0-py2.py3-none-any.whl (81kB)
         |################################| 81kB 5.5MB/s
    Requirement already satisfied: opencv-python in c:/anaconda3/envs/py35tf11/lib/site-packages (from baselines==0.1.6) (4.1.2.30)
    Requirement already satisfied: pyglet<=1.3.2,>=1.2.0 in c:/anaconda3/envs/py35tf11/lib/site-packages (from gym<0.16.0,>=0.15.4->baselines==0.1.6) (1.3.2)
    Requirement already satisfied: numpy>=1.10.4 in c:/anaconda3/envs/py35tf11/lib/site-packages (from gym<0.16.0,>=0.15.4->baselines==0.1.6) (1.18.1)
    Requirement already satisfied: six in c:/anaconda3/envs/py35tf11/lib/site-packages (from gym<0.16.0,>=0.15.4->baselines==0.1.6) (1.14.0)
    Requirement already satisfied: future in c:/anaconda3/envs/py35tf11/lib/site-packages (from pyglet<=1.3.2,>=1.2.0->gym<0.16.0,>=0.15.4->baselines==0.1.6) (0.18.2)
    Installing collected packages: tqdm, joblib, click, baselines
      Running setup.py develop for baselines
    Successfully installed baselines click-7.0 joblib-0.14.1 tqdm-4.41.1
    ```

### 다관절접합 ( MuJoCo: MUlti-JOint dynamics in COntact )

기준선 본보기의 일부는 [다관절접합(MuJoCo: MUlti-JOint dynamics in COntact)](http://www.mujoco.org) 물리 모사기(Simulator)를 사용한다, 이것은 소유권이 있다 그리고 바이너리와 권한(30일 사용권한은 [www.mujoco.org](http://www.mujoco.org)에서 얻을수 있다)이 필요하다. 다관절접합(MuJoCo) 설정에 대한 지침은 [여기](https://github.com/openai/mujoco-py)에서 찾을수 있다.  

## 설치 평가

All unit tests in baselines can be run using pytest runner:
기준선의 모든 단위 평가는 `pytest` 실행기를 사용하여 실행할수 있다:

```
pip install pytest
pytest
```

- 윈도우10 에서 `pip install pytest`실행결과  

```shell
Collecting pytest
  Downloading https://files.pythonhosted.org/packages/e6/94/29b5a09edc970a2f24742e6186dfe497cd6c778e60e54693215a36852613/pytest-5.3.3-py3-none-any.whl (235kB)
     |################################| 245kB 252kB/s
Collecting more-itertools>=4.0.0
  Downloading https://files.pythonhosted.org/packages/bc/e2/3206a70758a21f9878fcf9478282bb68fbc66a5564718f9ed724c3f2bb52/more_itertools-8.1.0-py3-none-any.whl (41kB)
     |################################| 51kB 3.2MB/s
... ~ ...
Collecting zipp>=0.5
  Downloading https://files.pythonhosted.org/packages/f4/50/cc72c5bcd48f6e98219fc4a88a5227e9e28b81637a99c49feba1d51f4d50/zipp-1.0.0-py2.py3-none-any.whl
Requirement already satisfied: six in c:/anaconda3/envs/py35tf11/lib/site-packages (from pathlib2>=2.2.0; python_version < "3.6"->pytest) (1.14.0)
Collecting pyparsing>=2.0.2
  Downloading https://files.pythonhosted.org/packages/5d/bc/1e58593167fade7b544bfe9502a26dc860940a79ab306e651e7f13be68c2/pyparsing-2.4.6-py2.py3-none-any.whl (67kB)
     |################################| 71kB 2.2MB/s
Installing collected packages: more-itertools, zipp, importlib-metadata, atomicwrites, wcwidth, pathlib2, py, colorama, pluggy, pyparsing, packaging, attrs, pytest
Successfully installed atomicwrites-1.3.0 attrs-19.3.0 colorama-0.4.3 importlib-metadata-1.4.0 more-itertools-8.1.0 packaging-20.0 pathlib2-2.3.5 pluggy-0.13.1 py-1.8.1 pyparsing-2.4.6 pytest-5.3.3 wcwidth-0.1.8 zipp-1.0.0
```

- 윈도우10 에서 `pytest`를 실행한 결과 오류 발생및 발생오류 제거... 

```shell
pytest
================================== test session starts =============================
platform win32 -- Python 3.5.6, pytest-5.3.3, py-1.8.1, pluggy-0.13.1
rootdir: G:/Work_GaeBal/baselines
collected 125 items / 1 skipped / 124 selected

baselines/bench/test_monitor.py F                                             [  0%]
baselines/common/test_mpi_util.py s                                           [  1%]
baselines/common/tests/test_cartpole.py ssssss                                [  6%]
baselines/common/tests/test_doc_examples.py s                                 [  7%]
baselines/common/tests/test_env_after_learn.py FF.FFF                         [ 12%]
baselines/common/tests/test_fixed_sequence.py ss                              [ 13%]
baselines/common/tests/test_identity.py ssssssssssssss                        [ 24%]
baselines/common/tests/test_mnist.py ssssss                                   [ 29%]
baselines/common/tests/test_plot_util.py .                                    [ 30%]
baselines/common/tests/test_schedules.py ..                                   [ 32%]
baselines/common/tests/test_segment_tree.py .....                             [ 36%]
baselines/common/tests/test_serialization.py ...FF....FF.FFFFFFFFFFFF......   [ 60%]
baselines/common/tests/test_tf_util.py ..                                     [ 61%]
baselines/common/tests/envs/identity_env_test.py ..                           [ 63%]
baselines/common/vec_env/test_vec_env.py ................s                    [ 76%]
baselines/common/vec_env/test_video_recorder.py FFFFFFFFFFFFFFFFFFFFFFFF      [ 96%]
baselines/ddpg/test_smoke.py ....                                             [ 99%]
baselines/ppo2/test_microbatches.py .                                         [100%]

================================== FAILURES ========================================

========= 17 failed, 77 passed, 32 skipped, 31 warnings in 233.77s (0:03:53) =======
```

- 윈도우10 에서 `pytest`실행결과  

```shell
E   ImportError: No module named 'matplotlib'
E   ImportError: No module named 'pandas'
E   ImportError: No module named 'filelock'
```

- 위는 `matplotlib`, `pandas` 설치가 안되었다는 오류임. 다음처럼 설치함

```shell
pip install matplotlib  # ==3.0.3
pip install pandas      # ==0.25.3
pip install filelock    # ==3.0.12
```

- win10 에 필요한 gym 놀이는 추가로 설치해야 한다...!

```shell
pip install git+https://github.com/Kojoley/atari-py.git

# 파이선에서 정상설치 확인
>>> import gym
>>> env = gym.make('CartPole-v1')         # CartPole-v1
>>> env = gym.make('PongNoFrameskip-v4')  # PongNoFrameskip-v4
```

- win10 `ffmpeg` 또는 `avconv` 미설치 오류 

```shell
# win10 `ffmpeg` 또는 `avconv` 미설치 오류

E           gym.error.DependencyNotInstalled: Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.

# win10 `ffmpeg` 설치명령
conda install -c conda-forge ffmpeg

# win10 `ffmpeg` 설치결과
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.7.12
  latest version: 4.8.1

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: C:\Anaconda3\envs\py35tf11

  added / updated specs:
    - ffmpeg


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    ffmpeg-4.2                 |       h6538335_0        23.4 MB  conda-forge
    ------------------------------------------------------------
                                           Total:        23.4 MB

The following NEW packages will be INSTALLED:

  ffmpeg             conda-forge/win-64::ffmpeg-4.2-h6538335_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
ffmpeg-4.2           | 23.4 MB   | ############################################################################ | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

- `PermissionError: [WinError 32]`

```shell
E       PermissionError: [WinError 32] 다른 프로세스가 파일을 사용 중이기 때문에 프로세스가 액세스 할 수 없습니다: '/tmp/baselines-test-43bc8f94-76b2-4bd8-9ff5-af31e52ab1e4.monitor.csv'

baselines/bench/test_monitor.py:31: PermissionError
```

## 벼림 모형

기준선 저장소 알고리즘 대부분은 다음처럼 사용된다:

```bash
python -m baselines.run --alg=<알고리즘 이름> --env=<환경_고유번호> [추가 결정고유값]
```

### 본보기 1: PPO 와 다관절접합 Humanoid

예를 들어, `PPO2`를 사용하여 전부-연결된 다관절접합 `humanoid` 제어 망을 2000만 보 훈련을 위해

```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7
```

다관절접합 환경은 전부연결된 망이 기본값임을 참고하라, 그래서 우리는 `--network=mlp`를 생략할수 있다. 
망과 벼림알고리즘 모두에 대한 잠정참여는 줄명령을 통해 제어할수 있다, 예를 들면:

```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```

엔트로피 계수(`ent_coef`)를 0.1로 설정, 그리고 3단의 각각에 은닉된 32개의 전부연결된 망을 구축한다, 그리고 가치 추정함수를 위한 망을 별도로 생성한다(그래서 이것들의 참여는 정책망과 공유되지 않는다, 하지만 이 구조는 같다).

[common/models.py](baselines/common/models.py)에서 모형의 각 유형에 대한 망 참여(parameter)를 기술한 설명글을 보라, 그리고 [baselines/ppo2/ppo2.py/learn()](baselines/ppo2/ppo2.py#L152)에서 `ppo2` 잠정참여(hyperparameter)를 기술한 설명글도.

### 본보기 2: 아타리 DQN

아타리 DQN은 현재는 고전적인 비교막대(benchmarks)다. 아타리 퐁(`Pong`)에서 DQN 구현 기준선(baseline)을 실행하기 위해:

```
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
```

## 모형을 저장, 탑재 그리고 보기

### 모형 저장과 탑재

순차 API 알고리즘은 아직 제대로 통합되지 않았다; 하지만, 벼림한 모형의 저장/복원하기 위한 간단한 기법이 있다. `--save_path` 와 `--load_path` 줄명령 선택은 벼림전에 주어진 경로에서 텐서플로우 상태를 탑재한다, 그리고 벼림후에 저장한다, 각각. 아타리 퐁()을 `ppo2`로 벼림한다고 가정하자, 모형을 저장한다 그런다음 벼림한 것을 보여준다.

```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2
```

이것은 약 20토막 당 평균포상을 가진다. 모형을 탑재하고 보여주기 위해, 우리는 다음을 수행한다 - 모형 탑재, 0보를 벼림한다, 그런다음 보여준다:

```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play
```

*알림:* 다관절접합 환경이 잘 작동하기 위해서는 고루기(normalization)가 필요하다, 그래서 우리는 `VecNormalize` 싸개로 싼다. 현재, 고루기계수(normalization coefficient)로 고룬(normalization) 모형저장(그래서 벼림된 모형을 복원하고 추가벼림없이 실행할수 있다)을 보장하기 위해 텐서플로우 변수로 저장된다. 이것은 성능이 다소 줄어들수 있다, 그래서 만약 다관절접합을 포함하여 많은 처리량 보가 필요하고 모형의 저장과 복원이 필요없다면, 대신에 넘피 고루기(normalization)를 사용하는 것이 좋다. 그렇게 하려면, [baselines/run.py](baselines/run.py#L116)에서 116줄을 `use_tf=False`로 설정한다.

### 학습 곡선과 그 외 벼림지표를 기록하고 보여준다

기본적으로, 요약 자료, 진행율 포함, 표준출력 모두는, 임시폴더 안에 고유 디렉토리에 저장된다, 파이썬에서 [tempfile.gettempdir()](https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir) 호출에 의해 지정된. 디렉토리는 `--log_path` 줄명령 선택으로 변경할수 있다.

```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2 --log_path=~/logs/Pong/
```

*알림:* 기록기는 존재하는 디렉토리에 같은 이름의 파일을 덮어쓸수 있다는 것을 알아둬라, 따라서 기록의 덮어쓰기를 방지하기 위해 고유한 특정시간을 부여하는 것을 권장한다.

다른 방법은 임시디렉토리를 변경할수 있는 `$OPENAI_LOGDIR` 환경변수를 사용하는 것이다.

벼림자료 탑재와 표시 방법에 대한 본보기는, [여기](docs/viz/viz.ipynb)를 보라.

## 보조 패키지 (Subpackage)

- [A2C](baselines/a2c): 강한소행평가(Advantage Actor Critic)  
- [ACER](baselines/acer): 경험재생 소행평가(Actor-Critic with Experience Replay)  
- [ACKTR](baselines/acktr): 크로네커-지수 신뢰영역을 사용하여 소행평가(Actor Critic using Kronecker-factored Trust Region)  
- [DDPG](baselines/ddpg): 깊은 결정적인 정책기울기(Deep Deterministic Policy Gradient)  
- [DQN](baselines/deepq): 깊은 가치 망(Deep Quality Network)  
- [GAIL](baselines/gail): (Generative Adversarial Imitation Learning)  
- [HER](baselines/her): 뒤늦은 경험재생(Hindsight Experience Replay)  
- [PPO1](baselines/ppo1) (사용되지 않는 판, 임시로 여기에 남음)  
- [PPO2](baselines/ppo2): 근사(근위)정책 최적화(Proximal Policy Optimization)  
- [TRPO](baselines/trpo_mpi): 신뢰영역 정책 최적화(Trust Region Policy Optimization)  

## 비교막대 (Benchmark)

다관절접합(1백만 보)과 아타리(1천만 보) 각각의 가능한 비교막대 결과:  

- [이것은 다관절접합](./benchmarks_mujoco1M.htm)  
- [이것은 아타리](./benchmarks_atari10M.htm)  

이 결과는 최신코드의 결과가 아닐수 있음을 알린다, 얻은 결과를 포함한 특정 지르기 해시는 비교막대(benchmark) 페이지에 지정되어 있다.

출판물에 이 저장소를 인용하려면:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai and Zhokhov, Peter},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }

