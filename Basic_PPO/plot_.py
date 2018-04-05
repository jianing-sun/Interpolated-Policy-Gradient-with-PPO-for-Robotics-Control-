import os
import matplotlib.pyplot as plt


class Plot:
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
        self.pPolicyLoss = []
        self.pValFcLoss = []

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

    def updatePolicyLoss(self, item):
        self.pPolicyLoss.append(item)

    def updateKL(self, item):
        self.pKL.append(item)

    def updateValFcLoss(self, item):
        self.pValFcLoss.append(item)

    def makeDict(self):
        plot_dict = {}
        plot_dict['Mean Rewards'] = self.pMeanRewards
        plot_dict['Episodes'] = self.pEpisodes
        plot_dict['KL'] = self.pKL
        plot_dict['Policy Loss'] = self.pPolicyLoss
        plot_dict['Policy Entropy'] = self.pPolicyEn
        plot_dict['Beta'] = self.pBeta
        plot_dict['CriticLoss'] = self.pValFcLoss
        plot_dict['Success Rate'] = self.pSuccessRate
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












