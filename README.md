- Some Results based on FetchReach-v0 environment between IPG and PPO:

A [slide](https://github.com/jianing-sun/Interpolated-Policy-Gradient-with-PPO-for-Robotics-Control-/blob/master/RL2018.pdf) for initial result and comparison among PPO, IPG and HER+IPG based on multi-goal RL environment (FetchReach-v0 from Robotic of Gym).

<img src="https://github.com/jianing-sun/Interpolated-Policy-Gradient-with-PPO-for-Robotics-Control-/blob/master/Results/Apr-12_23:16-Seed1234.png" width="500px" />

(./Results/Apr-12_23:16-Seed1234.png)

- TODO:


- [x] Combine with Experience Replay!!
- [x] Clean up current neural networks settings and 
- [x] Check if it's under correct formula from the paper
- [x] Combine with basic IPG
- [x] More debug, verify if the algorithm is working as it was totally created based on my understanding of the algorithm diagram from the initial paper (it works)
- [ ] More experiments on pushing, sliding, pick&place
- [x] Generalize to hindsight experience replay and compare with experience replay 
- [x] Change tanh to ReLU
- [x] Change mini batch size to 256 (so far is 500)
- [x] For pushing, sliding tasks, need to train more episodes (in original papar, they train for 50 epochs (one epoch consists of 19 · 2 · 50 = 1 900 full episodes, which amounts to a total of 4.75 · 106 timesteps). And also **improve the time steps to 2500 per episode** for these tasks. (Should not change the time steps configuration in Gym, but must train with much more episodes, in Multi-goal Reinforcement learning (no.5 reference), they train more than 50000 episodes to see little improvement).
- [x] Try to delete coefficient (1/ET) for on-policy loss
- [ ] Add a stochastic target policy for critic of IPG
- Reference:
  - [Proximal Policy Gradient](https://arxiv.org/pdf/1707.02286.pdf)
  - [Trust Region Policy Gradient](https://arxiv.org/pdf/1502.05477.pdf)
  - [Interpolated Policy Gradient](https://arxiv.org/pdf/1706.00387.pdf)
  - [Hindsight Experience Replay](http://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf)
  - [Generalized Advantage Estimator](https://arxiv.org/pdf/1506.02438.pdf)
  - [Multi-goal Reinforcement Learning](https://d4mucfpksywv.cloudfront.net/research-covers/ingredients-for-robotics-research/technical-report.pdf)





```Python
 "CODE IS FAR AWAY FROM BUG WITH THE ANIMAL PROTECTING"
 
 *          ##2      ##2
 *       ┏-##1　  ┏-##1
 *    ┏_┛ ┻---━┛_┻━━┓
 *    ┃　　　        ┃　　　　 
 *    ┃　　 ━　      ┃　　　 
 *    ┃ @^　  @^    ┃　　 
 *    ┃　　　　　　  ┃
 *    ┃　　 ┻　　　 ┃
 *    ┃_　　　　　 _┃
 *     ┗━┓　　　┏━┛
 *    　　┃　　　┃神兽保佑
 *    　　┃　　　┃永无BUG！
 *    　　┃　　　┗━━━┓----|
 *    　　┃　　　　　　　  ┣┓}}}
 *    　　┃　　　　　　　  ┏┛
 *    　　┗┓&&&┓-┏&&&┓┏┛-|
 *    　　　┃┫┫　 ┃┫┫
 *    　　　┗┻┛　 ┗┻┛
 *
 *
 "CODE IS FAR AWAY FROM BUG WITH THE ANIMAL PROTECTING"
```

​	

Jianing Sun

Last modified: April 23th, 2018