# HumbleRL
Straightforward reinforcement learning Python framework. It will provide all the boilerplate code needed to implement RL logic (see diagram below) with different publicly available environments and own agents (plus e.g. logging).

<p align="center"><img src ="misc/rl_diagram.png" /></p>

It's not a deep learning framework! It's designed to work with them to build agents (e.g. PyTorch or TensorFlow).

**Work in progress!** It's not officially released yet. Contributions are welcome :smile:

## How to run?
### Dependencies:
* Tested on python 3.6.4. It _should_ work with Python 2.7 too. 
* See `requirements.txt` for rest of dependencies (note that `pytest` is needed only to run tests).

## Samples:
We are currently working on research project "Transfer Learning in Reinforcement Learning" and we are developing this small tool as we go. We will publish sample code (how we use this tool) some time in the future. You can expect [AlphaZero](http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/) and [World Model](https://worldmodels.github.io) implementations in this framework.

## What we are currently working on?
The most important things now are to improve logging and visualization capabilities, but also add support for more environments. Visualizing and training supervision should be easy-peasy and one should be able to run experiments in many many environments! We are waiting for you contribution :smile:
