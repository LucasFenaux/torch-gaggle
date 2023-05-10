# Gaggle
Genetic Algorithms on the GPU

This is the version of gaggle used for the experiments in the paper: Gaggle: Genetic Algorithms on the GPU using 
Pytorch.

The code can be also accessed as a package by installing it with pip:

<code>pip install torch-gaggle</code>

To run the code during review, use the environment provided in the parent directory of the experiment code.

## Example

We provided a simple training example script in the examples folder.

It can be run in the following way from the examples folder:

<pre><code>python3 train.py --config_path ../configs/train_mnist_lenet.yml </code></pre>

## Tutorials

Two tutorials can be found in the tutorials folder. 
The first one: <code>introduction.ipynn</code> covers using the GASupervisor to solve pre-built problems and get a high level overview of using Genetic Algorithms to solve problems.
The second one: <code>research_mode.ipynb</code> goes into a lot more depth and covers each of the main components of the inner workings of Gaggle to allow for configuration file support, reproducible experiments and custom code integration.

## Paper Experiments

The paper experiment code can be found at [Gaggle Experiment Code](https://github.com/LucasFenaux/gaggle-benchmarking).
