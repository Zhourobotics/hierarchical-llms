# llm_target_tracking


### Prerequisites

- [ROS](https://wiki.ros.org/ROS/Installation): Our framework has been tested in ROS Noetic.

- [Forces Pro](https://www.embotech.com/products/forcespro/overview/): You can request an academic license from [here](https://www.embotech.com/products/forcespro/licensing/).

- [openai](https://platform.openai.com/docs/overview)


```
pip install openai
```


### run 

```
python tracker_server.py exp1
```
change exp to test different settings.

- You may need to clean the solver folder when you change some setups from problem and dangerzone.



### config

- LLM related paramters. Set true to run LLMs. The duration is to control the calling frequency. (will change to callback later)

```
llm_inner_dur: 2
llm_outer_dur: 10
llm_on: True
```

- Use steps and dt to control the experiment duation.
```
steps: 100
Problem: dt: 0.2
```

- Use task ability to change the maximum number of target a robot can track
```
task_ability: 1
```




### folder

```
└── tracker
  ├── CMakeLists.txt
  ├── config
  │   ├── exp1.yaml
  │   ├── exp2.yaml
  │   ├── exp3.yaml
  │   ├── exp4.yaml
  │   ├── exp5.yaml
  │   ├── exp6.yaml
  │   ├── exp7.yaml
  │   └── sim.rviz
  ├── data
  │   ├── 0.pcd
  │   ├── 1.pcd
  │   └── source
  │       └── WMSC_points.pcd
  ├── launch
  │   └── vis_sim.launch
  ├── package.xml
  └── script
      ├── llm_adaptive_server.py
      ├── model
      │   ├── config_loader.py
      │   ├── danger_zones.py
      │   ├── dynamics.py
      │   ├── forcepro_centralized.py
      │   ├── forcepro_single.py
      │   ├── problem.py
      │   └── tracker_manger.py
      ├── tracker_server.py
      └── utils
          ├── keys.py
          ├── pcd_segmentation.py
          ├── publish_pcd.py
          ├── visualizer.py
          └── visualizer_ros.py
```
