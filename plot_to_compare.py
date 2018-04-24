import matplotlib.pyplot as plt
import csv
import os

#PPO
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

# IPG
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

#HER
ipgppo = './Hindsight_ExperienceReplay/log-files/FetchReach-v0/Apr-23_21:13:47-Pre/log.csv'
h_episode = '_Episode'
h_meanReward = '_MeanReward'

episode3 = []
meanReward3 = []
reader = csv.reader(open(ipgppo, 'rU'), delimiter=',', dialect='excel')
hrow = next(reader)

idx5 = hrow.index(h_episode)
idx6 = hrow.index(h_meanReward)
for row in reader:
    episode3.append(row[idx5])
    meanReward3.append(row[idx6])

episode3 = list(map(int, episode3))
meanReward3 = list(map(float, meanReward3))
print(episode3)
print(meanReward3)

plt.plot(episode1, meanReward1, label='PPO')
plt.plot(episode2, meanReward2, label='IPG')
plt.plot(episode3, meanReward3, label='HER+IPG')
plt.xlabel('Mean Reward')
plt.ylabel('Episodes')
plt.legend()


path = os.path.join('./Results')
plt.savefig(os.path.join(path, 'Apr-12_23:16-Seed1234'))

