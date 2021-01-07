# TUD_PDM_FianlProject_group10

## Brief description
This repository mainly shows the source files of the PDM final project, which uses RRT* and MPC algorithm to solve the automatic parking problem in the parking lot.

The code mainly depend on two parts the source code [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics): "RRTStarReedsShepp" and "model_predictive_speed_and_steer_control".

## Code part
There are three parts:
- RRT* + Reeds-Shep path: `rrt_star_reeds_shepp.py`, `rrt_star.py`, `rrt.py`, `reeds_shepp_path_planning.py`
- MPC: `model_predictive_speed_and_steer_control.py`, `cubic_spline_planner.py`
- Main code that combine them into this application: `Find_Path_Park.py`

The simulation can be started by just run the `Find_Path_Park.py` code.

The example video can be found in `simulation video` folder.

