## Hindsight_ExperienceReplay

- Hindsight Experience Replay allows sample-efficient learning from rewards which are sparse and binary and therefore avoid the need for complicated reward engineering. Basically HER adds more samples to the replay buffer R. The `desired_goal` of these samples are substitute by a set of random selected `achieved_goal` from a random time step in a random episode/trajectories. That means, instead only regard the object position as the target/goal, HER adds some timesteps' current achieved goal as the desired goal for earlier samples.

- Pseudo-code for HER based on DDPG (in my experiment I use IPG with PPO)

  ![](https://ws4.sinaimg.cn/large/006tKfTcgy1fq83lve7xoj318u0zcalx.jpg)