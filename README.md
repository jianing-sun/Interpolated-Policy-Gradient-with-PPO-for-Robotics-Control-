## Interpolated Policy Gradient for PPO

- The main algorithm used here is Proximal Policy Gradient (PPO) for continuous control task. Basic environment from a newly introduced environment - Robotics. It is a multi-goal environment from OpanAI Gym and use the MuJoCo physic engine for fast and accurate simulation. 
- Mainly based on Fetch environment with four task and start from FetchReach-v0, the other three are FetchPush, FetchSlide, FetchPickAndPlace. 
- Parameters changing process during experiments will be saved at **plot-files**. So far *Success Rate*, *Mean Rewards*, *Policy Entropy* have obvious regular changing and they are important results/parameters.


- TODO:
  - [x] Combine with Experience Replay!!
  - [x] Clean up current neural networks settings and 
  - [ ] Check if it's under correct formula from the paper
  - [x] Combine with basic IPG
  - [ ] More debug, verify if the algorithm is working as it was totally created based on my understanding of the algorithm diagram from the initial paper
  - [ ] More experiments on pushing, sliding, pick&place
  - [ ] Generalize to hindsight experience replay and compare with experience replay 
  - [ ] Change tanh to ReLU
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

