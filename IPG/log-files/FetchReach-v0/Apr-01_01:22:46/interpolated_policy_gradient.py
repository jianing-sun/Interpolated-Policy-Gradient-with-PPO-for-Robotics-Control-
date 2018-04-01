import gym
import numpy as np
import tensorflow as tf

from IPG.experience_replay import SimpleReplayPool


class PolicyIPG:
    def __init__(self, obs_dim, act_dim, kl_target, val_fc):
        self.kl_targ = kl_target
        self.beta = 1.0
        self.eta = 50
        self.lr_multiplier = 1.0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr_multiplier = 1.0
        self.v = 0.2
        self.epochs = 20
        self.replay_pool_size = 1000000

        self.val_fc = val_fc
        self.e_qval = []
        self.pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=16,  # for FetchReach, the length of observation is 10, total maybe 10+3+3
            action_dim=4,
            env=gym.wrappers.
                FlattenDictWrapper(gym.make('FetchReach-v0'), ['observation', 'desired_goal', 'achieved_goal'])
        )
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph
            Initialize graph with all functions beginning with a dash
            except sample() and update are used in reach.py for sampling and updating
        """

        self.g = tf.Graph()     # create a new empty graph
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

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

                Calculates log probabilities using previous step's model parameters and
                new parameters being trained.
                """
        logp = -0.5 * tf.reduce_sum(self.log_vars)  # -0.5*-17=8.5
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1, name="logp")

        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _sample(self):
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))  # add_5:0

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

    def _kl_entropy(self):
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

    def _loss_train_op(self):
        with tf.name_scope('loss'):

            loss1 = -tf.reduce_mean(self.advantages_ph *
                                    tf.exp(self.logp - self.logp_old), name="PG")
            loss2 = tf.reduce_mean(self.beta_ph * self.kl, name="KL")
            loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ), name="HingeLoss")  # quadratically smooth
            self.loss = tf.reduce_sum(loss1 + loss2 + loss3, name="TotalLoss")

            # Interpolated Policy Gradient Mixture Loss with on- and off-policy
            self.loss *= (1 - self.v)
            # self.e_qval = self.val_fc.predict(self.obs_ph)
            # a = tf.reduce_mean(self.e_qval)
            # print(tf.reduce_sum(a * self.v))
            self.loss -= self.v * tf.reduce_mean(self.e_qval)

            optimizer = tf.train.AdamOptimizer(self.lr_ph)  # gradient descent with Adam optimizer
            self.train_op = optimizer.minimize(self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr_ph)  # gradient descent with Adam optimizer
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)

        # TensorBoard
        # writer = tf.summary.FileWriter('tensorboard/', self.sess.graph)

        self.sess.run(self.init)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages):

        base_batch_size = len(observes) + len(actions) + len(advantages)
        batch_size = base_batch_size
        if self.pool.size < batch_size:
            ac_obs = observes
        else:
            batch_data = self.pool.random_batch(batch_size=batch_size)
            ac_obs = batch_data["observation"]
        print(observes.shape)
        print(ac_obs.shape)
        observes += ac_obs
        self.e_qval = self.val_fc.predict(observes)

        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.e_qval: self.e_qval,
                     self.lr_ph: self.lr * self.lr_multiplier}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0

        for e in range(self.epochs):
            # # Interpolated Policy Gradient Mixture Loss with on- and off-policy
            # self.loss *= (1 - self.v)

            # self.loss -= self.v * tf.reduce_mean(e_qval)
            #
            # optimizer = tf.train.AdamOptimizer(self.lr_ph)  # gradient descent with Adam optimizer
            # self.train_op = optimizer.minimize(self.loss)

            self.sess.run(self.train_op, feed_dict)

            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
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

        self.val_fc.fit()






