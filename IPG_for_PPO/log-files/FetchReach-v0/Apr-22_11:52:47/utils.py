import csv
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np


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


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, now):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        path = os.path.join('log-files', logname, now)
        os.makedirs(path)

        filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        for filename in filenames:     # for reference
            shutil.copy(filename, path)
        path = os.path.join(path, 'log.csv')

        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
                                                               log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()


class Plotter:
    def __init__(self, plotname, now):
        self.path = os.path.join('plot-files', plotname, now)
        os.makedirs(self.path)

        self.plot_dict = {}
        self.pMeanRewards = []
        self.pEpisodes = []
        self.pSuccessRate = []
        self.pPolicyEn = []
        self.pKL = []
        self.pBeta = []
        self.pTotalLoss = []
        self.pBaselineLoss = []
        self.pCriticLoss = []
        self.pOffPolicyLoss = []

    def updateOffPolicyLoss(self, item):
        self.pOffPolicyLoss.append(item)

    def updateMeanR(self, item):
        self.pMeanRewards.append(item)

    def updateEpisodes(self, item):
        self.pEpisodes.append(item)

    def updateSuccessR(self, item):
        self.pSuccessRate.append(item)

    def updatePolicyEn(self, item):
        self.pPolicyEn.append(item)

    def updateBeta(self, item):
        self.pBeta.append(item)

    def updateTotalLoss(self, item):
        self.pTotalLoss.append(item)

    def updateKL(self, item):
        self.pKL.append(item)

    def updateBaselineLoss(self, item):
        self.pBaselineLoss.append(item)

    def updateCriticLoss(self, item):
        self.pCriticLoss.append(item)

    def makeDict(self):
        plot_dict = {}
        plot_dict['Mean Rewards'] = self.pMeanRewards
        plot_dict['Episodes'] = self.pEpisodes
        plot_dict['KL'] = self.pKL
        plot_dict['Total Loss'] = self.pTotalLoss
        plot_dict['Policy Entropy'] = self.pPolicyEn
        plot_dict['Beta'] = self.pBeta
        plot_dict['Success Rate'] = self.pSuccessRate
        plot_dict['Baseline Loss'] = self.pBaselineLoss
        plot_dict['Critic Loss'] = self.pCriticLoss
        plot_dict['Off-Policy Loss'] = self.pOffPolicyLoss
        return plot_dict

    def plot(self):
        plot_dict = self.makeDict()
        plot_keys = [k for k in plot_dict.keys()]
        i = 0
        for key in plot_keys:
            if key != 'Episodes':
                plt.figure(i)
                i += 1
                plt.plot(plot_dict['Episodes'], plot_dict[key], label=key)
                plt.xlabel("Episodes")
                plt.ylabel(key)
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(self.path, key))



