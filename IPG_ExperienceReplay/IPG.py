import argparse
import random

import gym
import numpy as np
import scipy.signal
import tensorflow as tf
from sklearn.utils import shuffle


class CriticNN(object):
    """ NN-based state-value function """

    # TODO: so this is not a state-action function but is a state-value function nn??
    def __init__(self, obs_dim, name):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
        """
        with tf.variable_scope(name):
            self.replay_buffer_x = None
            self.replay_buffer_y = None
            self.obs_dim = obs_dim
            self.epochs = 10
            self.lr = None  # learning rate set in _build_graph()
            self._build_graph()
            self.sess = tf.Session(graph=self.g)
            self.sess.run(self.init)

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
            hid1_size = self.obs_dim * 10  # 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)                  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat) / np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))  # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func
        return loss

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means


class OnPolicyPPO(object):
    """ NN-based policy approximation """

    def __init__(self, obs_dim, act_dim, kl_targ):
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ  # KL target
        self.epochs = 20
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()  # create a new empty graph
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')  # = shape(?, 17)
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * 10  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name="h1")  # mean=0, standard deviation = np.sqrt(1/self.obs_dim)
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        self.means = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)), name="means")

        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48  # integer division  = 35
        # print('logvar_speed',logvar_speed)  # 35
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))  # 0

        self.log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0  # [-1, -1, -1......, -1]   17 dimension in total

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1, name="logp")

        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        with tf.name_scope('KL'):
            self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                           tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                         tf.exp(self.log_vars), axis=1) - self.act_dim)
        with tf.name_scope('Entropy'):
            self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                                  tf.reduce_sum(self.log_vars))

    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))  # add_5:0

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        with tf.name_scope('loss'):
            loss1 = -tf.reduce_mean(self.advantages_ph *
                                    tf.exp(self.logp - self.logp_old), name="PG")
            loss2 = tf.reduce_mean(self.beta_ph * self.kl, name="KL")
            loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ),
                                            name="HingeLoss")  # quadratically smooth
            self.loss = tf.reduce_sum(loss1 + loss2 + loss3, name="TotalLoss")

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, loss, observes, actions, advantages):
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}

        optimizer = tf.train.AdamOptimizer(self.lr_ph)  # gradient descent with Adam optimizer
        self.train_op = optimizer.minimize(loss)

        kl, entropy = 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train_op, feed_dict)
            kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()


# Experience replay buffer
class Buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[int(0.0001 * self.buffer_size):]

    def sample(self, size):
        if len(self.buffer) >= size:
            experience_buffer = self.buffer
        else:
            experience_buffer = self.buffer * size
        return np.copy(np.reshape(np.array(random.sample(experience_buffer, size)), [size, 3]))


def BatchSample(current_buffer, size):
    return np.copy(np.reshape(np.array(random.sample(current_buffer, size)), [size, 3]))


def run_policy(env, policy, scaler, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes
    """
    total_steps = 0
    trajectories = []
    episode_experiences = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs, episode_experience = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        episode_experiences.append(episode_experience)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations

    return trajectories, episode_experiences


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate
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
        observes.append(obs) # center and scale observations
        action = policy.sample(obs).reshape((1, -1)).astype(np.float64)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature
        current_buffer.append((temp_obs, action, reward))

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs), current_buffer)


# def compute_qvalue(off_trajectories, val_func, gamma):
#     for off_trajectory in off_trajectories:
#         observes = off_trajectories[0]
#         values = val_func.predict(observes)
#         off_trajectory.append(values)
#         if gamma < 0.999:  # don't scale for gamma ~= 1
#             rewards = off_trajectory[2] * (1 - gamma)
#         else:
#             rewards = off_trajectory[2]
#         q_value = rewards + np.append(values[1:] * gamma, 0)
    # return q_value


def compute_vvalue(trajectories, val_func):
    for on_trajectory in trajectories:  # 15 trajectories, each with 50 time steps
        observes = on_trajectory['observes']
        values = val_func.predict(observes)
        on_trajectory['values'] = values


def critic_compute_vvalue(dict_states, val_func):
    """
    compute V(st) for next step td error calculation
    :param dict_states:
    :param val_func:
    :return:
    """
    values = val_func.predict(dict_states['states'])
    dict_states['values'] = values


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def compute_advantages(trajectories, gamma, lam):
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
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    learning_sginals = np.zeros((len(trajectories), len(trajectories[0]['advantages'])))
    for i in range(0, learning_sginals.shape[0]):
        for j in range(0, learning_sginals.shape[1]):
            learning_sginals[i][j] = trajectories[i]['advantages'][j]
    return observes, actions, advantages, learning_sginals, disc_sum_rew


def main(num_episodes, gamma, lam, kl_targ, batch_size, env_name):
    # initialize gym environment and get observations and actions
    env = gym.make(env_name)
    env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal', 'achieved_goal'])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    time_steps = 50  # T, time steps in every episode
    userCV = False
    interpolate_ratio = 0.2  # set v
    samples_size = 40

    # add 1 to obs dimension for time step feature (see run_episode())
    obs_dim += 1
    scaler = Scaler(obs_dim)
    baseline = CriticNN(obs_dim, name='baseline')
    critic = CriticNN(obs_dim, name='critic')
    on_policy = OnPolicyPPO(obs_dim, act_dim, kl_targ)

    # initialize replay buffer
    buff = Buffer(1000000)

    # run 5 episodes to initialize scaler
    run_policy(env, on_policy, scaler, episodes=5)
    episode = 0

    # start training
    # initial_buff_size = 15
    while episode < num_episodes:
        """experience replay: there are two buffers, one is replay buffer which 
        keep expanding with new experiences (off-policy); one is current buffer 
        ("play" buffer) which only contains current experience (on-policy)
        """
        # roll-out pi for initial_buff_size episodes, T (50) time step each to
        # collect a batch of data to R (replay buffer)
        current_buffer = []
        trajectories, episode_experiences = run_policy(env, on_policy, scaler, episodes=batch_size)
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
        with on_policy.sess as sess:
            on_feed_dict = {on_policy.obs_ph: observes,
                            on_policy.act_ph: on_actions,
                            on_policy.advantages_ph: advantages,
                            on_policy.beta_ph: on_policy.beta,
                            on_policy.eta_ph: on_policy.eta,
                            on_policy.lr_ph: on_policy.lr * on_policy.lr_multiplier}
            old_means_np, old_log_vars_np = sess.run([on_policy.means, on_policy.log_vars], feed_dict=on_feed_dict)
            on_feed_dict[on_policy.old_log_vars_ph] = old_log_vars_np
            on_feed_dict[on_policy.old_means_ph] = old_means_np
            # compute loss
            on_policy_loss = sess.run(on_policy.loss, feed_dict=on_feed_dict)
            print(on_policy_loss)
            # times (1/ET)
            on_policy_loss = (1 / (time_steps * batch_size)) * on_policy_loss

            # compute off-policy loss (second term in the IPO algorithm loss function)
            """
            consider using Sarsa as the critic, then delta Q = Rt+1 + gamma * Q(St+1, At+1) - Q(St, At)
            then the loss is the sum over all the batch samples
            """

            # target = td_target(env, states, on_policy)  # compute Rt+1 + gamma * Q(St+1, At+1)
            dict_states = {'states': states}
            # add values evaluation for current states
            critic_compute_vvalue(dict_states, critic)
            # compute (td target - current values) as delta Qw(Sm) under PPO policy
            b = interpolate_ratio
            off_policy_loss = TD(env, dict_states, on_policy, critic)
            off_policy_loss = (b / samples_size) * np.sum(off_policy_loss)

            loss = on_policy_loss + off_policy_loss

        """update policy using interpolated policy gradient loss function"""
        on_policy.update(loss, observes, actions, advantages)

        """update baseline and critic"""
        # observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        baseline.fit(observes, sum_dis_return)  # update value function

    """close sessions"""
    on_policy.close_sess()
    baseline.close_sess()


def TD(env, dict_states, policy, critic, gamma=0.995):
    states = dict_states['states']
    states_ = []
    rewards_ = []
    for state in states:
        action = policy.sample(np.array(state).reshape(1, 17)).reshape((1, -1)).astype(np.float64)
        state_, reward, done, _ = env.step(action)
        state_ = np.append(state_, [state[-1]+0.001])
        # dict_temp = {'state_': state_,
        #              'r_': reward}
        # dict_states.append(dict_temp)
        states_.append(state_)
        rewards_.append(reward)
    values_ = critic.predict(states_)
    dict_states['values_'] = values_
    td_errors = rewards_ + gamma * values_ - dict_states['values']

    return td_errors


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
                        default=15)

    args = parser.parse_args()

    main(**vars(args))
