# MPC-Based Control using MuJoCo and Julia

Julia implementation of Model Predictive Path Integral (MPPI) control for the Unitree Go1 quadruped robot and cartpole using the MuJoCo physics engine. The code is designed to simulate and control the robot's movement while optimizing for specific cost functions.

# go1 walking


[![dog go brr](https://img.youtube.com/vi/td9j1vgA25c/0.jpg)](https://www.youtube.com/watch?v=td9j1vgA25c0)

## Overview

The goal of this project is to simulate and control the Unitree Go1 robot using a Model Predictive Control (MPC) approach with the MPPI algorithm. The control loop optimizes the robot's trajectory by minimizing a cost function that includes terms for height tracking, orientation, velocity, control effort, and symmetry.

## Dependencies

The following Julia packages are required to run the code:

- `MuJoCo.jl`: For interfacing with the MuJoCo physics engine.
- `LinearAlgebra`: For linear algebra operations.
- `Random`: For generating random numbers.
- `Statistics`: For statistical operations.
- `Base.Threads`: For multithreading support.

## MPPI Control

The MPPI (Model Predictive Path Integral) control algorithm is used to optimize the control inputs over a finite horizon. The algorithm samples control sequences, evaluates their costs, and updates the control inputs based on the weighted average of the sampled sequences.
