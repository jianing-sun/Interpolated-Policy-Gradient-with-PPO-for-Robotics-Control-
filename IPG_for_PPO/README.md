## Interpolated Policy Gradient

- Interpolated Policy Gradient is an approach to **merging on- and off-policy updates for deep reinforcement learning.**

- The main algorithm used here is **Proximal Policy Gradient (PPO)** for continuous control tasks. Basic environment is a newly-introduced environment - **Robotics**. It is a multi-goal environment from OpanAI Gym and use the MuJoCo physic engine for fast and accurate simulation. 

- Mainly based on Fetch environment with four task and start experiments with FetchReach-v0, going to extend to other three tasks in the future: FetchPush, FetchSlide, FetchPickAndPlace, however, these tasks have a high requirement for the hardware (num of cpu cores, num of episodes)

- Parameters changing process during experiments will be saved at **plot-files**. So far *Success Rate*, *Mean Rewards*, *Policy Entropy* have obvious regular changing and they are important results/parameters.

- **Problems so far:**

  - [ ] **Not always have good results, sometimes it would be even worse sometimes it would be similar with the original PPO experiments. One of the reasons I think it's the experiment environment/task is too simple (simple reaching task), other reasons might be about the hyperparameters.**
  - [ ] The learning process has higher variance, in some episodes the mean reward would jump from -26 to -34 which cause some "burrs" for figures.
  - [ ] For complicated tasks it has a bad performance.
  - [ ] Change the critic to state-action based instead only state based.

- Pseudo-code for IPG:

  ![](https://ws3.sinaimg.cn/large/006tNc79gy1fq1pkn1c05j314i0jqgr1.jpg)