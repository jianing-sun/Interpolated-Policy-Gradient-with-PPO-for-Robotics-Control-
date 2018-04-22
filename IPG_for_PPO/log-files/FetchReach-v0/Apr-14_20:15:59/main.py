import argparse
import random
import datetime

import gym
import numpy as np
import scipy.signal
import tensorflow as tf

from IPG_for_PPO.nn_value_function import ValueFncNN
from IPG_for_PPO.utils import Scaler, Logger, Plotter
from IPG_for_PPO.PPO import OnPolicyPPO
from IPG_for_PPO.replay_buffer import Buffer


# TODO: integrate this method within the replay buffer class
def BatchSample(current_buffer, size):
    return np.copy(np.reshape(np.array(random.sample(current_buffer, size)), [size, 3]))


def run_policy(env, policy, scaler, logger, plotter, episodes, plot=True):

    """ Run policy and collect data for a minimum of min_steps and min_episodes
    Everytime we call this method will trigger 50 episodes training, and will get 50 trajectories in total. Every
    trajectory is a dict with observes, actions, rewards, and unsclaed_obs. append these 50 trajectory lead to a big
    list with 50 dict inside named trajectories.

    :returns
    trajectories, for on-policy training
    episode_experiences, this is the same with trajectories but with a different type and shape used to fit the requirement
    of the batch data saved into the big replay buffer.
    """

    total_steps = 0
    trajectories = []
    episode_experiences = []
    success_rates = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs, episode_experience, success_rate = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        episode_experiences.append(episode_experience)
        success_rates.append(success_rate)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations

    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    if plot:
        plotter.updateMeanR(np.mean([t['rewards'].sum() for t in trajectories]))
        plotter.updateSuccessR(np.mean(success_rates))

    return trajectories, episode_experiences


def run_episode(env, policy, scaler, animate=False):

    """ Run single episode with option to animate.
    This method is triggered inside run_policy, this will form a trajectory with 50 timesteps for each. Every time we
    will sample an action based on the PPO algorithm policy distribution, we will take that action and get corresponding
    next observes and rewards. observes are appended with the value of time steps (start from 0, every step increase 1e-3).

    :returns
    concatenate all the observes, sampled actions, rewards, and unscaled_obs after taking 50 sampled actions during the
    trajectory.  current buffer saves current episode experience, we append 50 current buffer in the run_policy method.
    """

    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    current_buffer = []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        temp_obs = obs
        observes.append(obs)        # center and scale observations
        action = policy.sample(obs).reshape((1, -1)).astype(np.float64)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature
        current_buffer.append((temp_obs, action, reward))

    success_rate = rewards.count(-0.0) / len(rewards)
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs), current_buffer, success_rate)


def compute_vvalue(trajectories, val_func):

    """evaluate the values for all the trajectories in current big episode.
    The size of the values should be the batch_size (15) * total timesteps for each episode (50)
    Calculate the values by using the ValueFncNN class, and save that into the trajectory dict
    """

    for on_trajectory in trajectories:  # 15 trajectories, each with 50 time steps
        observes = on_trajectory['observes']
        values = val_func.predict(observes)
        on_trajectory['values'] = values


def critic_compute_vvalue(dict_states, val_func):

    """the critic neural network is the same structure of the value function neural network used to count the advantages,
    but there are TWO neural networks with different shape of input, so for interpolated policy gradient, here I used
    this critic nn to compute the off-policy TD target based on random samples from the replay buffer.
    This is a medium step to compute the TD error. As the input shape is different we can't use the same one to predict.
    """

    values = val_func.predict(dict_states['states'])
    dict_states['values'] = values


def discount(x, gamma):

    """ Calculate discounted forward sum of a sequence at each point """

    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def compute_advantages(trajectories, gamma, lam):

    """ This is used to calculate advantage functions for all the trajectories based on the baseline neural network not
    the critic nn. This is actually Monte Carlo advantages as we use complete whole trajectory.
    """

    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages
    # return trajectories['advantages']


def add_disc_sum_rew(trajectories, gamma):

    """ This is used to calculate the expected true value of the target for the updating. Target is Gt, the error here
    is the MC error: Gt - Vt
    """

    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def build_train_set(trajectories):

    """Connect all the trainings into a big set.
    :returns
    We need all the observations, actions, advantages to feed into main ppo policy neural network.
    Plus we need the discounted sum reward (Gt) for each trajectory (15*50)
    """

    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    learning_signals = np.zeros((len(trajectories), len(trajectories[0]['advantages'])))

    # TODO：check if it is something wrong for computing learning signals
    for i in range(0, learning_signals.shape[0]):
        for j in range(0, learning_signals.shape[1]):
            learning_signals[i][j] = trajectories[i]['advantages'][j]
    return observes, actions, advantages, learning_signals, disc_sum_rew


def TD(env, dict_states, policy, critic, gamma=0.995):

    """Compute one step temporal difference error.
    Formula: Rt+1 + gamma * Vt+1 - Vt
    :param dict_states: dict_states save the observations from samples (D=S1:m)
    :param policy: taking a step based on random samples following current policy
    :param critic:
    :return: return TD errors and targets for all random samples from replay buffer
    """

    states = dict_states['states']
    states_ = []
    rewards_ = []
    for state in states:
        # action = policy.sample(np.array(state).reshape(1, env.observation_space.shape[0]+1)).reshape((1, -1)).astype(np.float64)
        action = policy.getMean(np.array(state).reshape(1, env.observation_space.shape[0]+1)).reshape((1, -1)).astype(np.float64)
        state_, reward, done, _ = env.step(action)
        state_ = np.append(state_, [state[-1]+0.001])      # TODO: what if the timestep is the final step in an episode?
        states_.append(state_)
        rewards_.append(reward)
    # compute one-step forward values_
    values_ = critic.predict(states_)
    dict_states['values_'] = values_
    td_targets = rewards_ + gamma * values_
    td_errors = td_targets - dict_states['values']

    return td_errors, td_targets


def main(num_episodes, gamma, lam, kl_targ, batch_size, env_name):

    """ main function for the overall process of interpolated policy gradient (off-line update)
    :param num_episodes: total episodes numbers
    :param batch_size: in every big episode, after batch_size times episodes, we update the policy and neural networks
    """

    # initialize gym environment and get observations and actions
    env = gym.make(env_name)
    gym.spaces.seed(1234)
    env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal'])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # parameters
    time_steps = 50  # T, time steps in every episode
    userCV = False
    interpolate_ratio = 0.2  # set v
    samples_size = 64

    # logger and plotter from utilies
    now = (datetime.datetime.utcnow() - datetime.timedelta(hours=4)).strftime(
        "%b-%d_%H:%M:%S")  # create dictionaries based on ETS time
    logger = Logger(logname=env_name, now=now)
    plotter = Plotter(plotname=env_name+"-Fig", now=now)

    # add 1 to obs dimension for time step feature (see run_episode())
    obs_dim += 1
    scaler = Scaler(obs_dim)

    # initialize three neural network, on for the ppo policy, one for the value function baseline used to compute
    # advantages, and one is critic
    baseline = ValueFncNN(obs_dim, name='baseline')
    critic = ValueFncNN(obs_dim, name='critic')
    on_policy = OnPolicyPPO(obs_dim, act_dim, kl_targ)

    # initialize replay buffer
    buff = Buffer(1000000)

    # run 5 episodes to initialize scaler
    run_policy(env, on_policy, scaler, logger, plotter, episodes=5, plot=False)
    episode = 0

    # start training
    with on_policy.sess as sess:
        while episode < num_episodes:

            """experience replay: there are two buffers, one is replay buffer which 
            keep expanding with new experiences (off-policy); one is current buffer 
            ("play" buffer) which only contains current experience (on-policy)
            """
            # roll-out pi for initial_buff_size episodes, T (50) time step each to
            # collect a batch of data to R (replay buffer)
            current_buffer = []
            trajectories, episode_experiences = run_policy(env, on_policy, scaler, logger,
                                                           plotter, episodes=batch_size, plot=True)
            episode += len(trajectories)
            plotter.updateEpisodes(episode)

            for i in range(0, batch_size):
                for j in range(0, time_steps):
                    state, action, reward = episode_experiences[i][j]
                    buff.add(np.reshape([state, action, reward], [1, 3]))  # add to replay buffer
                    current_buffer.append(np.reshape([state, action, reward], [1, 3]))

            # current i don't use the control variate, so no need to compute Q value here
            # """fit Qw through off-policy (use replay buffer)"""
            # off_trajectories = buff.sample(batch_size*time_steps)  # numpy array
            # q_values = compute_q_value(off_trajectories, off_policy, gamma)

            """fit baseline V() through on-policy (use current trajectories)"""
            compute_vvalue(trajectories, baseline)
            # print(trajectories)

            """compute Monte Carlo advantage estimate advantage (on-policy)"""
            compute_advantages(trajectories, gamma, lam)
            # here as we don't use control variate, learning_signals equal advantages but with a different shape
            # to facilitate next step of the algorithm
            # so in the on-policy advantages I just input with the advantages which is wrong in the strict sense
            # TODO: change the advantages as the form of learning signal
            add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            observes, on_actions, advantages, learning_signals, sum_dis_return = build_train_set(trajectories)
            log_batch_stats(observes, on_actions, advantages, logger, sum_dis_return, episode)

            """different situations based on if we use control variate: if useCV=True, then compute
            critic-based advantage estimate using current buffer, Q and policy
            if useCV=False, then just center the learning signals lt,e=At,e
            """
            # if userCV:
            #     pass
            # else:
            #     # center the learning signals = advantages, and set b = v
            #     learning_signals = advantages
            #     b = interpolate_ratio

            # multiply learning signals by (1-v)
            learning_signals *= (1 - interpolate_ratio)

            """sample D=S1:M from replay buffer or current buffer based on beta (M=40)"""
            if buff.buffer_size < len(current_buffer):
                # using on-policy samples to compute loss and optimize policy
                samples = BatchSample(current_buffer, samples_size)
            else:
                # using off-policy samples to compute loss and optimize policy (always go here)
                # TODO: what's the condition to change?
                samples = buff.sample(samples_size)

            """compute loss function"""
            states, actions, rewards = [np.squeeze(elem, axis=1) for elem in np.split(samples, 3, 1)]
            states = np.array([s for s in states])
            states = np.squeeze(states)

            # compute PPO loss (first term in the IPO algorithm loss function)
            # with on_policy.sess as sess:
            on_feed_dict = {on_policy.obs_ph: observes,
                            on_policy.act_ph: on_actions,
                            on_policy.advantages_ph: advantages,
                            on_policy.beta_ph: on_policy.beta,
                            on_policy.eta_ph: on_policy.eta,
                            on_policy.lr_ph: on_policy.lr * on_policy.lr_multiplier}
            old_means_np, old_log_vars_np = sess.run([on_policy.means, on_policy.log_vars], feed_dict=on_feed_dict)
            on_feed_dict[on_policy.old_log_vars_ph] = old_log_vars_np
            on_feed_dict[on_policy.old_means_ph] = old_means_np

            sess.run(on_policy.train_op, on_feed_dict)

            # compute loss
            on_policy_loss = sess.run(on_policy.loss, feed_dict=on_feed_dict)

            # times (1/ET)
            # on_policy_loss = (1 / (time_steps * batch_size)) * on_policy_loss
            surr_loss = on_policy_loss

            # compute off-policy loss (second term in the IPG algorithm loss function)
            """
            consider using temporal difference as the critic, then delta Q = Rt+1 + gamma * Q(St+1, At+1) - Q(St, At)
            then the loss is the sum over all the batch samples
            """
            # dict_states is a dict for random samples from replay buffer, not for trajectory
            dict_states = {'states': states}
            # evaluate values (Vt) for samples and add them to   the dict by using the critic neural network
            critic_compute_vvalue(dict_states, critic)
            # compute (td target - current values) as delta Qw(Sm) under PPO policy
            b = interpolate_ratio
            # compute Rt+1 + gamma * Q(St+1, At+1)
            off_policy_loss, td_targets = TD(env, dict_states, on_policy, critic)
            off_policy_loss = (b / samples_size) * np.sum(off_policy_loss)
            plotter.updateOffPolicyLoss(off_policy_loss)
            surr_loss += off_policy_loss

            print("on_policy_loss: {}. Off_policy_loss: {}. Total Loss: {}".format(on_policy_loss, off_policy_loss, surr_loss))
            print("")

            """update current policy based on current observes, actions, advantages"""
            on_feed_dict[on_policy.loss] = tf.reduce_sum(surr_loss)
            on_policy.update(surr_loss, observes, on_actions, advantages, old_means_np, old_log_vars_np, logger, plotter)

            """update baseline and critic"""
            # observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
            # with baseline.sess as sess:
            baseline.fit(observes, sum_dis_return, logger, plotter, id="BaselineLoss")  # update value function

            # with critic.sess as sess:
            critic.fit(states, td_targets, logger, plotter, id="CriticLoss")
            logger.write(display=True)

    """record"""
    logger.close()
    plotter.plot()

    """close sessions"""
    on_policy.close_sess()
    baseline.close_sess()


def log_batch_stats(observes, actions, advantages, logger, disc_sum_rew, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name', default="Hopper-v2")
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)

    args = parser.parse_args()

    main(**vars(args))
