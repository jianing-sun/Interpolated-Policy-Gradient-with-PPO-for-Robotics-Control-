import tensorflow as tf
from IPG_for_PPO.Critic.QFunction import compile_function


class CriticEval:
    """
    Base class defining methods for policy evaluation.
    """
    def init_critic(self,
            min_pool_size=10000,
            replay_pool_size=1000000,
            replacement_prob=1.0,
            qf_batch_size=32,
            qf_weight_decay=0.,
            qf_update_method=tf.train.AdamOptimizer(1e-3),
            qf_use_target=True,
            qf_mc_ratio = 0,
            qf_residual_phi = 0,
            soft_target_tau=0.001,
            **kwargs):
        self.soft_target_tau = soft_target_tau
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.replacement_prob = replacement_prob
        self.qf_batch_size = qf_batch_size
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = qf_update_method
        self.qf_use_target = qf_use_target
        self.discount = 0.99

        self.qf_mc_ratio = qf_mc_ratio
        if self.qf_mc_ratio > 0:
            self.mc_y_averages = []

        self.qf_residual_phi = qf_residual_phi
        if self.qf_residual_phi > 0:
            self.residual_y_averages = []
            self.qf_residual_loss_averages = []

        self.qf_loss_averages = []
        self.q_averages = []
        self.y_averages = []

    def init_opt_critic(self):
        target_qf = self.qf

        obs = self.qf.env_spec.observation_space.new_tensor_variable(
            'qf_obs',
            extra_dims=1,
        )
        action = self.qf.env_spec.action_space.new_tensor_variable(
            'qf_action',
            extra_dims=1,
        )
        yvar = tf.placeholder(dtype=tf.float32, shape=[None], name='ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([tf.reduce_sum(tf.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = tf.reduce_mean(tf.square(yvar - qval))
        qf_input_list = [yvar, obs, action]
        qf_output_list = [qf_loss, qval]

        # set up residual gradient method
        if self.qf_residual_phi > 0:
            next_obs = self.qf.env_spec.observation_space.new_tensor_variable(
                'qf_next_obs',
                extra_dims=1,
            )
            rvar = tf.placeholder(dtype=tf.float32, shape=[None], name='rs')
            terminals = tf.placeholder(dtype=tf.float32, shape=[None], name='terminals')
            discount = tf.placeholder(dtype=tf.float32, shape=(), name='discount')
            qf_loss *= (1. - self.qf_residual_phi)
            next_qval = self.qf.get_e_qval_sym(next_obs, self.policy)
            residual_ys = rvar + (1.-terminals)*discount*next_qval
            qf_residual_loss = tf.reduce_mean(tf.square(residual_ys-qval))
            qf_loss += self.qf_residual_phi * qf_residual_loss
            qf_input_list += [next_obs, rvar, terminals, discount]
            qf_output_list += [qf_residual_loss, residual_ys]

        # set up monte carlo Q fitting method
        if self.qf_mc_ratio > 0:
            mc_obs = self.qf.env_spec.observation_space.new_tensor_variable(
                'qf_mc_obs',
                extra_dims=1,
            )
            mc_action = self.qf.env_spec.action_space.new_tensor_variable(
                'qf_mc_action',
                extra_dims=1,
            )
            mc_yvar = tf.placeholder(dtype=tf.float32, shape=[None], name='mc_ys')
            mc_qval = self.qf.get_qval_sym(mc_obs, mc_action)
            qf_mc_loss = tf.reduce_mean(tf.square(mc_yvar - mc_qval))
            qf_loss = (1.-self.qf_mc_ratio)*qf_loss + self.qf_mc_ratio*qf_mc_loss
            qf_input_list += [mc_yvar, mc_obs, mc_action]

        qf_reg_loss = qf_loss + qf_weight_decay_term
        self.qf_update_method.update_opt(
            loss=qf_reg_loss, target=self.qf, inputs=qf_input_list)
        qf_output_list += [self.qf_update_method._train_op]

        f_train_qf = compile_function(inputs=qf_input_list, outputs=qf_output_list)

        self.opt_info_critic = dict(
            f_train_qf=f_train_qf,
            target_qf=target_qf,
        )

    def do_critic_training(self, itr, batch, policy):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        target_qf = self.opt_info_critic["target_qf"]

        target_policy = policy
        next_qvals = target_qf.get_e_qval(next_obs, target_policy)

        ys = rewards + (1. - terminals) * self.discount * next_qvals
        inputs = (ys, obs, actions)

        qf_outputs = self.opt_info_critic['f_train_qf'](*inputs)
        qf_loss = qf_outputs.pop(0)
        qval = qf_outputs.pop(0)
        if self.qf_residual_phi:
            qf_residual_loss = qf_outputs.pop(0)
            residual_ys = qf_outputs.pop(0)
            self.qf_residual_loss_averages.append(qf_residual_loss)
            self.residual_y_averages.append(residual_ys)

        if self.qf_use_target:
            target_qf.set_param_values(
                target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
                self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)
