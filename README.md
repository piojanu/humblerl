# HumbleRL
Straightforward reinforcement learning Python framework. It will provide all the boilerplate code needed to implement RL logic (see diagram below) with different publicly available environments and own agents (plus e.g. logging).

<p align="center"><img src ="misc/rl_diagram.png" /></p>

It's not a deep learning framework! It's designed to work with them to build agents (e.g. PyTorch or TensorFlow).

**Work in progress!** It's not officially released yet. Contributions are welcome :smile:

## How to run?
### Dependencies:
* Tested on python 3.6.4. It _should_ work with Python 2.7 too. 
* numpy - https://scipy.org/install.html
* tqdm - https://github.com/tqdm/tqdm

## Samples:
We are currently working on research project "Transfer Learning in Reinforcement Learning" and we are developing this small tool as we go. We will publish sample code (how we use this tool) some time in the future. You can expect AlphaZero implementation in this framework. We will move some generic code from that implementation to this framework.

## What we are currently working on?
The most important thing now is to improve logging and visualization capabilities. So visualizing and training supervision will be easy-peasy! Idea is to use something in shape of Keras metrics and TensorBoard, but we are still working it out. We are waiting for you contribution :smile:
