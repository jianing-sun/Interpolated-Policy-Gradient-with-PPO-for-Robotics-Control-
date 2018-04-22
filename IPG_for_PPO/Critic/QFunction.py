import numpy as np
import tensorflow as tf

import IPG_for_PPO.Critic.layers as L


def compile_function(inputs, outputs):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))

    return run


class ContinuousQFunction:
    def __init__(self, obs_dim, act_dim):
        self.name = 'qnet'
        self.hidden_sizes = (32, 32)
        self.action_merge_layer = -1
        self.hidden_nonlinearity = tf.nn.relu
        self.batch_norm = False
        self.eqf_use_full_qf = False
        self.eqf_sample_size = 1
        self.n_itr = 500
        self.discount = 0.99

        with tf.variable_scope(self.name):
            l_obs = L.InputLayer(shape=(None, obs_dim), name="obs")
            l_action = L.InputLayer(shape=(None, act_dim), name="actions")
            n_layers = len(self.hidden_sizes) + 1
            if n_layers > 1:
                self.action_merge_layer = \
                    (self.action_merge_layer % n_layers + n_layers) % n_layers
            else:
                self.action_merge_layer = 1
            l_hidden = l_obs
            for idx, size in enumerate(self.hidden_sizes):
                if self.batch_norm:
                    l_hidden = L.batch_norm(l_hidden)
                if idx == self.action_merge_layer:
                    l_hidden = L.ConcatLayer([l_hidden, l_action])
                l_output = L.DenseLayer(
                    l_hidden,
                    num_units=1,
                    nonlinearity=None,
                    name="output"
                )
                output_var = L.get_output(l_output, deterministic=True)
                output_var = tf.reshape(output_var, (-1,))

                self._f_qval = compile_function([l_obs.input_var, l_action.input_var], output_var)
                self._output_layer = l_output
                self._obs_layer = l_obs
                self._action_layer = l_action
                self._output_nonlinearity = None

    def get_qval(self, observations, actions):
        return self._f_qval(observations, actions)

    def get_e_qval_sym(self, obs_var, policy, **kwargs):
        return self._get_e_qval_sym(obs_var, policy, **kwargs)[0]

    def _get_e_qval_sym(self, obs_var, policy, **kwargs):
        [mean_var, log_std_var] = policy.getMeanAndLogVar(obs_var)
        # mean_var, log_std_var = agent_info['mean'], agent_info['log_std']
        return self.get_qval_sym(obs_var, mean_var, **kwargs), mean_var

    def get_e_qval(self, observations, policy):
        [means, log_stds] = policy.getMeanAndLogVar(observations)
        # means, log_stds = agent_info['mean'], agent_info['log_std']
        if self.eqf_use_full_qf and self.eqf_sample_size > 1:
            observations = np.repeat(observations, self.eqf_sample_size, axis=0)
            means = np.repeat(means, self.eqf_sample_size, axis=0)
            stds = np.repeat(np.exp(log_stds), self.eqf_sample_size, axis=0)
            randoms = np.random.randn(*(means))
            actions = means + stds * randoms
            all_qvals = self.get_qval(observations, actions)
            qvals = np.mean(all_qvals.reshape((-1, self.eqf_sample_size)), axis=1)
        else:
            qvals = self.get_qval(observations, means)

        return qvals

    def get_qval_sym(self, obs_var, action_var, **kwargs):
        qvals = L.get_output(
            self._output_layer,
            {self._obs_layer: obs_var, self._action_layer: action_var},
            **kwargs
        )
        return tf.reshape(qvals, (-1,))
