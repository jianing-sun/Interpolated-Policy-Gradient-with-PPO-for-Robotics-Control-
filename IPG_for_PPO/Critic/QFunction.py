import itertools

import numpy as np
import tensorflow as tf
from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable

import IPG_for_PPO.Critic.layers as L


def compile_function(inputs, outputs):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))

    return run


class QFunction(Parameterized):

    def __init__(self, env_spec):
        Parameterized.__init__(self)
        self._env_spec = env_spec

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def env_spec(self):
        return self._env_spec


class LayersPowered(Parameterized):

    def __init__(self, output_layers, input_layers=None):
        self._output_layers = output_layers
        self._input_layers = input_layers
        Parameterized.__init__(self)

    def get_params_internal(self, **tags):
        layers = L.get_all_layers(self._output_layers, treat_as_input=self._input_layers)
        params = itertools.chain.from_iterable(l.get_params(**tags) for l in layers)
        return L.unique(params)


class ContinuousQFunction(QFunction, LayersPowered, Serializable):
    def __init__(
            self,
            obs_dim,
            act_dim,
            name='qnet',
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            action_merge_layer=-2,
            eqf_use_full_qf=False,
            eqf_sample_size=1,
            bn=False):
        Serializable.quick_init(self, locals())
        # self.name = 'qnet'
        # self.hidden_sizes = (32, 32)
        # self.action_merge_layer = -1
        # self.hidden_nonlinearity = tf.nn.relu
        self.batch_norm = False
        self.eqf_use_full_qf = False
        # self.eqf_sample_size = 1
        self.n_itr = 500
        self.discount = 0.99
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            l_obs = L.InputLayer(shape=(None, obs_dim), name="obs")
            l_action = L.InputLayer(shape=(None, act_dim), name="actions")

            n_layers = len(hidden_sizes) + 1

            if n_layers > 1:
                action_merge_layer = \
                    (action_merge_layer % n_layers + n_layers) % n_layers
            else:
                self.action_merge_layer = 1

            l_hidden = l_obs

            for idx, size in enumerate(hidden_sizes):
                if bn:
                    l_hidden = L.batch_norm(l_hidden)
                if idx == action_merge_layer:
                    l_hidden = L.ConcatLayer([l_hidden, l_action])

                l_hidden = L.DenseLayer(
                    l_hidden,
                    num_units=size,
                    nonlinearity=hidden_nonlinearity,
                    name="h%d" % (idx + 1)
                )

            if action_merge_layer == n_layers:
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

            self.eqf_use_full_qf = eqf_use_full_qf
            self.eqf_sample_size = eqf_sample_size

            LayersPowered.__init__(self, [l_output])

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
