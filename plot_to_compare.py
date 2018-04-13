import matplotlib.pyplot as plt
import csv
import os

basicppo = './Basic_PPO/log-files/FetchReach-v0/Apr-12_22:02:13-Seed1234/log.csv'
h_episode = '_Episode'
h_meanReward = '_MeanReward'

episode1 = []
meanReward1 = []
reader = csv.reader(open(basicppo, 'rU'), delimiter=',', dialect='excel')
hrow = next(reader)

idx1 = hrow.index(h_episode)
idx2 = hrow.index(h_meanReward)
for row in reader:
    episode1.append(row[idx1])
    meanReward1.append(row[idx2])

episode1 = list(map(int, episode1))
meanReward1 = list(map(float, meanReward1))
print(episode1)
print(meanReward1)


ipgppo = './IPG_for_PPO/log-files/FetchReach-v0/Apr-12_23:16-Seed1234/log.csv'
h_episode = '_Episode'
h_meanReward = '_MeanReward'

episode2 = []
meanReward2 = []
reader = csv.reader(open(ipgppo, 'rU'), delimiter=',', dialect='excel')
hrow = next(reader)

idx3 = hrow.index(h_episode)
idx4 = hrow.index(h_meanReward)
for row in reader:
    episode2.append(row[idx3])
    meanReward2.append(row[idx4])

episode2 = list(map(int, episode2))
meanReward2 = list(map(float, meanReward2))
print(episode2)
print(meanReward2)

plt.plot(episode1, meanReward1, label='PPO')
plt.plot(episode2, meanReward2, label='IPG')
plt.legend()


path = os.path.join('./Results')
plt.savefig(os.path.join(path, '0412'))

