import numpy as np
import tensorflow as tf
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer

from IPG_for_PPO.Critic.QFunction import compile_function


class CriticEval:
    def __init__(self,
                 qf,
                 policy,
                 min_pool_size=10000,
                 replay_pool_size=1000000,
                 replacement_prob=1.0,
                 qf_batch_size=32,
                 qf_weight_decay=0.,
                 qf_update_method='adam',
                 qf_learning_rate=1e-3,
                 qf_use_target=True,
                 soft_target_tau=0.001,
                 ):

        self.soft_target_tau = soft_target_tau
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.replacement_prob = replacement_prob
        self.qf_batch_size = qf_batch_size
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = FirstOrderOptimizer(update_method=qf_update_method,
                                                    learning_rate=qf_learning_rate)
        self.qf_use_target = qf_use_target
        self.discount = 0.99
        self.qf = qf
        self.policy = policy

        self.qf_loss_averages = []
        self.q_averages = []
        self.y_averages = []

    def init_opt_critic(self, obs_dim, act_dim):

        target_qf = self.qf
        extra_dims = 1

        obs = tf.placeholder(tf.float32, shape=[None] * extra_dims + list([obs_dim]), name='qf_obs')
        action = tf.placeholder(tf.float32, shape=[None] * extra_dims + list([act_dim]), name='qf_action')

        # obs = tf.placeholder(tf.float32, shape=list([obs_dim] + [None] * extra_dims), name='qf_obs')
        # action = tf.placeholder(tf.float32, shape=list([act_dim] + [None] * extra_dims), name='qf_action')

        yvar = tf.placeholder(dtype=tf.float32, shape=[None], name='ys')

        # qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
        #                        sum([tf.reduce_sum(tf.square(param)) for param in
        #                             self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = tf.reduce_mean(tf.square(yvar - qval))
        qf_input_list = [yvar, obs, action]
        qf_output_list = [qf_loss, qval]

        # qf_reg_loss = qf_loss + qf_weight_decay_term
        qf_reg_loss = qf_loss

        self.qf_update_method.update_opt(
            loss=qf_reg_loss, target=self.qf, inputs=qf_input_list)
        # qf_output_list += [self.qf_update_method._train_op]

        f_train_qf = compile_function(inputs=qf_input_list, outputs=qf_output_list, sess=tf.get_default_session())

        self.opt_info_critic = dict(
            f_train_qf=f_train_qf,
            target_qf=target_qf,
        )

    def do_critic_training(self, batch):

        obs = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['states_']
        terminals = batch['terminals']

        target_qf = self.opt_info_critic["target_qf"]

        target_policy = self.policy
        next_qvals = target_qf.get_e_qval(next_obs, target_policy)

        ys = rewards + (1. - terminals) * self.discount * next_qvals
        inputs = (ys, obs, actions)

        qf_outputs = self.opt_info_critic['f_train_qf'](*inputs)
        qf_loss = qf_outputs.pop(0)
        qval = qf_outputs.pop(0)

        if self.qf_use_target:
            target_qf.set_param_values(
                target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
                self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

    def optimize_critic(self, batch, batch_size):
        """ Train the critic for batch sampling-based policy optimization methods
        :param samples:
        :param batch_size:
        :param policy:
        :return:
        """
        qf_updates_ratio = 1
        qf_itrs = float(batch_size) * qf_updates_ratio
        qf_itrs = int(np.ceil(qf_itrs))
        for i in range(qf_itrs):
            # Train critic
            self.do_critic_training(batch)
