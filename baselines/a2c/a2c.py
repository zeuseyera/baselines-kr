import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner
from baselines.ppo2.ppo2 import safemean
from collections import deque

from tensorflow import losses

class Model(object):

    """
    이 클래스를 사용하여 :
        __init__:
        - step_model(표집모형) 을 생성한다
        - train_model(벼림모형) 을 생성한다

        train():
        - 벼림하는 부분을 만든다(순전파와 기울기 역전파)

        save/load():
        - 모형 저장과 탑재
    """
    def __init__( self
            , policy
            , env
            , nsteps
            , ent_coef          = 0.01  # 엔트로피계수
            , vf_coef           = 0.5   # 가치계수
            , max_grad_norm     = 0.5
            , lr                = 7e-4
            , alpha             = 0.99  # 벼림비 에누리
            , epsilon           = 1e-5  # 
            , total_timesteps   = int( 80e6 )
            , lrschedule        = 'linear' ):

        sess    = tf_util.get_session()
        nenvs   = env.num_envs
        nbatch  = nenvs*nsteps

        with tf.variable_scope( 'a2c_model', reuse=tf.AUTO_REUSE ):
            # step_model 은 표집을 위해 사용한다
            step_model  = policy( nenvs, 1, sess )

            # train_model 은 망을 벼림하기위해 사용한다
            train_model = policy( nbatch, nsteps, sess )

        A   = tf.placeholder( train_model.action.dtype, train_model.action.shape )
        ADV = tf.placeholder( tf.float32, [nbatch] )
        R   = tf.placeholder( tf.float32, [nbatch] )
        LR  = tf.placeholder( tf.float32, [] )

        # 손실(loss)을 계산한다
        # Total loss = Policy gradient loss - entropy coefficient * entropy + Value coefficient * value loss
        # 총손실 = 정책 기울기손실 - 엔트로피계수 * 엔트로피 + 가치계수 * 가치손실

        # 정책 손실(Policy loss)
        neglogpac = train_model.pd.neglogp( A )
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # 엔트로피는 부적절한(suboptimal) 정책으로 조기수렴하는 것을 제한하여 탐사를 개선하는데 사용한다.
        entropy = tf.reduce_mean( train_model.pd.entropy() )

        # 가치 손실(Value loss)
        vf_loss = losses.mean_squared_error( tf.squeeze(train_model.vf), R )

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # 손실(loss)을 사용하여 참여값(parameter: 가중값, 편향값)을 갱신한다
        # 1. 모형의 참여값(parameter)을 가져온다
        params = find_trainable_variables( "a2c_model" )

        # 2. 기울기(gradient)를 계산한다
        grads = tf.gradients( loss, params )

        if max_grad_norm is not None:
            # 기울기를 제한한다: Clip the gradients ( normalize )
            grads, grad_norm = tf.clip_by_global_norm( grads, max_grad_norm )

        # zip 은 참여값(parameter)에 관련된 각각의 기울기를 합산(aggregate)한다.
        # 예를들어 zip(ABCD, xyza) => Ax, By, Cz, Da
        grads = list( zip(grads, params) )

        # 3. A2C 정책과 가치 갱신 한단계에 대한 동작을 만든다.
        trainer = tf.train.RMSPropOptimizer( learning_rate=LR, decay=alpha, epsilon=epsilon )

        _train = trainer.apply_gradients( grads )

        lr = Scheduler( v=lr, nvalues=total_timesteps, schedule=lrschedule )

        def train( obs, states, rewards, masks, actions, values ):
            # 여기에서 강점을 계산한다: advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values

            for step in range( len(obs) ):
                cur_lr = lr.value()

            td_map = { train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr }

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _ = sess.run( [pg_loss, vf_loss, entropy, _train]
                                                                , td_map )

            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
      network
    , env
    , seed          = None
    , nsteps        = 5
    , total_timesteps= int( 80e6 )
    , vf_coef       = 0.5
    , ent_coef      = 0.01
    , max_grad_norm = 0.5
    , lr            = 7e-4
    , lrschedule    = 'linear'
    , epsilon       = 1e-5
    , alpha         = 0.99
    , gamma         = 0.99
    , log_interval  = 100
    , load_path     = None
    , **network_kwargs ):

    '''
    A2C 알고리즘에 대한 주 진입지점. `a2c` 알고리즘을 사용하여 주어진 환경에서 주어진 망으로 정책을 벼림한다.

    Parameters:
    -----------

    network:            정책망 구조. 표준망 구조를 지정하는 문자열(mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small
                        , conv_only - 전체목록을 보려면 baselines.common/models.py를 보라), 또는 입력으로 텐서플로우
                        텐서를 가지고 출력 텐서는 망 마지막단 출력쌍(output_tensor, extra_feed)을 반환하는 함수,
                        , extra_feed 는 feed-forward 를 위해서는 None 다, 그리고 extra_feed 는 재사용신경망을 위한
                        망으로 상태를 사비하는 방법을 설명하는 목록(dictionary)이다. 정책에서 재사용신경망 사용에 대한
                        자세한 내용은 baselines.common/policies.py/lstm 을 보라.

    env:                강화각흡 환경. VecEnv(baselines.common/vec_env)와 비슷한 전달기를 구현하거나
                        DummyVecEnv(baselines.common/vec_env/dummy_vec_env.py)로 싸야 한다.

    seed:               알고리즘에서 뿌림수 순서를 복제하기 위한 씨알이다. 기본적으로 None 이다, 이것은 씨스템
                        노이즈생성기가 씨알임을 의미한다(복제하지 않는다)

    nsteps:             int, 환경을 배열의 보수 마다 갱신한다(즉, 사리수(batch size)는 nsteps * nenv 이다 여기에서
                        nenv 는 병렬로 모사한 환경을 복사한 개수다.)

    total_timesteps:    int, 벼림하기 위한 총 보수 (기본값: 80M)

    vf_coef:            float, 총손실 함수에서 가치함수 손실 앞의 계수 (기본값: 0.5)

    ent_coef:           float, 총손실 함수에서 정책 엔트로피 앞의 계수 (기본값: 0.01)

    max_gradient_norm:  float, 기울기(gradient)는 전역(global) L2 보다 크지않은 값으로 제한(clipped)한다 (기본값: 0.5)

    lr:                 float, RMSProp 을 위한 벼림비(현재 구현은 RMSProp 에서 강제(hardcoded)한다) (기본값: 7e-4)

    lrschedule:         벼림비 계획. 'linear', 'constant', 또는 [0..1] -> [0..1] 함수로 할수 있다, 이것은 벼림진행의
                        일부를 입력으로 취하여 출력으로 벼림비(lr 로 지정) 부분을 반환한다.

    epsilon:            float, RMSProp epsilon (RMSProp 갱신 분모로 제곱근 계산을 정상화 한다) (기본값: 1e-5)

    alpha:              float, RMSProp 에누리 참여값(decay parameter) (기본값: 0.99)

    gamma:              float, 포상 에누리 참여값(reward discounting parameter) (기본값: 0.99)

    log_interval:       int, 얼마나 자주 기록을 인쇄하는지 지정한다 (기본값: 100)

    **network_kwargs:   정책/망 작성기에 대한 열쇄글 결정고유값(arguments). baselines.common/policies.py/build_policy와
                        망의 특정 유형에 대한 결정고유값(arguments)을 봐라. 예를들어, 'mlp' 망 구조는 num_hidden 와
                        num_layers 의 결정고유값(arguments)을 가진다.

    '''

    set_global_seeds( seed )

    # 환경의 개수를 가져온다(Get the nb of env)
    nenvs   = env.num_envs
    policy  = build_policy( env, network, **network_kwargs )

    # 모형개체 대리자 (step_model(표집모형) 와 train_model(벼림모형)을 생성한다)
    model = Model( policy           = policy
                , env               = env
                , nsteps            = nsteps
                , ent_coef          = ent_coef
                , vf_coef           = vf_coef
                , max_grad_norm     = max_grad_norm
                , lr                = lr
                , alpha             = alpha
                , epsilon           = epsilon
                , total_timesteps   = total_timesteps
                , lrschedule        = lrschedule )

    if load_path is not None:
        model.load( load_path )

    # 실행개체 대리자(Instantiate the runner object)
    runner      = Runner( env, model, nsteps=nsteps, gamma=gamma )
    epinfobuf   = deque( maxlen=100 )

    # 사리수(batch_size) 계산
    nbatch = nenvs*nsteps

    # 전체타이머 시작
    tstart = time.time()

    for update in range( 1, total_timesteps//nbatch+1 ):
        # 경험의 작은 덩이를 가져온다. Get mini batch of experiences
        obs, states, rewards, masks, actions, values, epinfos = runner.run()
        epinfobuf.extend(epinfos)

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart

        # fps 계산 (frame per second)
        fps = int((update*nbatch)/nseconds)
        
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()

    return model

