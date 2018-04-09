## Interpolated Policy Gradient for PPO

- Interpolated Policy Gradient is an approach to merging on- and off-policy updates for deep reinforcement learning.
- The main algorithm used here is Proximal Policy Gradient (PPO) for continuous control task. Basic environment from a newly introduced environment - Robotics. It is a multi-goal environment from OpanAI Gym and use the MuJoCo physic engine for fast and accurate simulation. 
- Mainly based on Fetch environment with four task and start from FetchReach-v0, the other three are FetchPush, FetchSlide, FetchPickAndPlace. 
- Parameters changing process during experiments will be saved at **plot-files**. So far *Success Rate*, *Mean Rewards*, *Policy Entropy* have obvious regular changing and they are important results/parameters.
- **Problems so far:**
  - [ ] **Not always have good results, sometimes it would be even worse sometimes it would be similar with the original PPO experiments. One of the reasons I think it's the experiment environment/task is too simple (simple reaching task), other reasons might be about the hyperparameters.**
  - [ ] The learning process has higher variance, in some episodes the mean reward would jump from -26 to -34 which cause some "burrs" for figures.


- TODO:
  - [x] Combine with Experience Replay!!
  - [x] Clean up current neural networks settings and 
  - [x] Check if it's under correct formula from the paper
  - [x] Combine with basic IPG
  - [x] More debug, verify if the algorithm is working as it was totally created based on my understanding of the algorithm diagram from the initial paper (it works)
  - [ ] More experiments on pushing, sliding, pick&place
  - [ ] Generalize to hindsight experience replay and compare with experience replay 
  - [ ] Change tanh to ReLU
  - [x] For pushing, sliding tasks, need to train more episodes (in original papar, they train for 50 epochs (one epoch consists of 19 · 2 · 50 = 1 900 full episodes, which amounts to a total of 4.75 · 106 timesteps). And also **improve the time steps to 2500 per episode** for these tasks. (Should not change the time steps configuration in Gym, but must train with much more episodes, in Multi-goal Reinforcement learning (no.5 reference), they train more than 50000 episodes to see little improvement).
- Reference:
  - [Proximal Policy Gradient](https://arxiv.org/pdf/1707.02286.pdf)
  - [Trust Region Policy Gradient](https://arxiv.org/pdf/1502.05477.pdf)
  - [Interpolated Policy Gradient](https://arxiv.org/pdf/1706.00387.pdf)
  - [Hindsight Experience Replay](http://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf)
  - [Generalized Advantage Estimator](https://arxiv.org/pdf/1506.02438.pdf)
  - [Multi-goal Reinforcement Learning](https://d4mucfpksywv.cloudfront.net/research-covers/ingredients-for-robotics-research/technical-report.pdf)



- Pseduo-code for Interpolated Policy Gradient:

![](https://ws3.sinaimg.cn/large/006tNc79gy1fq1pkn1c05j314i0jqgr1.jpg)

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