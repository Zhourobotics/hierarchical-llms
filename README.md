# Hierarchical LLMs In-the-loop Optimization

A hierarchical Large Language Models (LLMs) framework for real-time multi-robot task allocation and target tracking with unknown hazards.


## 1. About 

__Authors__: [Yuwei Wu](https://github.com/yuwei-wu), [Yuezhan Tao](https://sites.google.com/view/yuezhantao/home), [Peihan Li](https://scholar.google.com/citations?user=Qg7-Gr0AAAAJ&hl=en), [Guangyao Shi](https://guangyaoshi.github.io/), Gaurav S. Sukhatme, and Vijay Kumar, and [Lifeng Zhou](https://zhourobotics.github.io/)

__Video Links__:  [Youtube](https://youtu.be/282BHEHNBq8)


__Related Paper__: Yuwei Wu, Yuezhan Tao, Peihan Li, Guangyao Shi, Gaurav S. Sukhatme, Vijay Kumar,  Lifeng Zhou, "Hierarchical LLMs In-the-loop Optimization for Real-time Multi-Robot Target Tracking under Unknown Hazards". 2024


__System Architecture__: 

<p align="center">
  <img src="docs/llm.png" />
</p>



## 2. Prerequisites

- [ROS] (optional) (https://wiki.ros.org/ROS/Installation): Our framework has been tested in ROS Noetic.

- [Forces Pro](https://www.embotech.com/products/forcespro/overview/): You can request an academic license from [here](https://www.embotech.com/products/forcespro/licensing/).

- [openai](https://platform.openai.com/docs/overview): Install by 

```
pip install openai
```


## 3. Setups

### 3.1 Runing mode

We have different setups for runing in numerial simulation, ros simulation and hardwares. As hardware and ros setup may depend on different simulator used. Here we provide a numerical experiments, which can be easily integrated into simulators and hardware platform. 


To perform numerical simulation, set the mode in "src/tracker/config/${your experiment file}" as simulation

```
exp: "simulation"  # 0: "simulation", 1: "ros simulation", 2: "ros real"
```

### 3.2 OpenAI API key

You will need to create a file in "src/tracker/secrets.yml" and set your OpenAI API key with the format:

```
api_keys:
    0: ""
```

### 3.3 LLM-related parameters

- LLM related paramters. Set true to run LLMs. The duration is to control the calling frequency. (will change to callback later)

```
llm_inner_dur: 2
llm_outer_dur: 10
llm_on: True
```

- Use steps and dt to control the experiment duration.
```
steps: 100
Problem: dt: 0.2
```

- Use task ability to change the maximum number of targets a robot can track
```
task_ability: 1
```

- The initial task assignment is given for the setup.


To check more scenarios and experiment settings, please refer to the files in "src/tracker/config/".


### Run with Numerical Simulation


```
python tracker_server.py exp3
```

- During the initial run, ForcesPro will generate solvers that can be reused for future runs.  

- You can modify `exp` to test different settings.

- You may need to clean the solver folder when you change some setups from the problem and danger zone.


## Maintaince

For any technical issues, please contact Yuwei Wu (yuweiwu@seas.upenn.edu).
