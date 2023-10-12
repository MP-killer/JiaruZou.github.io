---
layout: page
title:  Mini Bloom
description: a High Frequency Trading project
img: assets/img/proj_1/cover.jpg
importance: 1
category: work
---

**Team: Jiaru Zou, Haojiong Zhang, Eric Liro, Naren Alluri**
\\
Note: The open source code can be found [here](https://gitlab.engr.illinois.edu/ie421_high_frequency_trading_fall_2022/ie421_hft_fall_2022_group_06/mini_bloom)

#### **Introduction**

###### **1.Project Description:**

This is a semester long project for IE 421 - High Frequency Trading Technology instructed by [Professor David Lariviere](https://davidl.web.illinois.edu/) The main goal of the project is to evaluate the performance on training ML models using multiple acceleration infrastructures (Apple's M1, M2 chips, Google Colab's Nvidia's A100, T4 GPU, local Nvidia 1660ti, Goole Cloud Platform, NCSA HAL Cluster) and test the high accuracy training and low cost inference. We also tested the differences in performance between pytorch and tensorflow implementation. 

###### **2.Main Objectives:**

- To design a parser and search engien to create a dataset for model training, validating, and testing.
- To develop a computationally intensive model for testing large amounts of High Frequency Trading (HFT) data
- To have this model satisfy technical requirements allowing it to be hardware accelerated
- To test this model by using multiple hardware accelerators, including on the cloud

###### **3.Tools and sources for developing the data, model and benchmarking:**

***Data source**:* \\
The main data source for this project is through IEX, an exchange which generates vast amounts of HFT data. We have adapted Professor Lariviere’s IEX data downloader and parser for this project.

***Strategy Development**:* \\
We researched multiple models for testing the data and decided on using Long Short-Term Memory (LSTM) as the model. This serves as a computationally intensive model based on time-series data that can be hardware accelerated and can provide an output loss. This loss can be used for comparison purposes, as well as the time it takes to run the model.

***Visualizing**:* \\
We executed the LSTM model on two separate Machine Learning (ML) kernels; these being PyTorch and TensorFlow. Both of these kernels can provide hardware acceleration, which we ran locally on various machines, as well as on the cloud. The benchmarking resulted in loss scores and time to completion for the model testing. 

#### **Description of Project:**

###### **1.LSTM Model:**
Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) that is capable of learning and remembering long-term dependencies in sequential data. RNNs are neural networks that have a "memory" that allows them to process sequences of inputs, such as text, speech, or time series data. However, traditional RNNs are not able to effectively handle long-term dependencies due to the vanishing and exploding gradients problem.

LSTMs were designed to overcome this problem by introducing a new kind of neuron called a "memory cell" that can retain information for a longer period of time. The memory cell is a simple unit that can store and retrieve information, and it is connected to three different gates: an input gate, an output gate, and a forget gate. These gates are used to control the flow of information into and out of the memory cell.
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/neuron.png" title="neuron image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The basic type of an artificial nueron in LSTM model in shown above.
</div>

The input gate regulates the flow of new information into the memory cell, while the forget gate controls the removal of outdated information from the memory cell. The output gate determines the value that is passed to the next layer of the neural network. By controlling these gates, LSTMs are able to selectively remember and forget information, which allows them to effectively learn long-term dependencies.Using neurons with sigmoid threshold functions, these neural networks are able to express non-linear decision surfaces.
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/LSTM_Cell.svg.png" title="LSTM_Cell.svg image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/multilayer.png" title="multilayer image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An LSTM Cell that Processes Data Sequentially and Keeps a Hidden State.(left) <br>
    A multilayer feed-forward neural network with one input layer, twohidden layers, and an output layer.(right)
</div>

LSTMs have been used in a wide range of applications, including natural language processing, speech recognition, and time series forecasting. They have also been used in sequential decision-making tasks, such as controlling robots and autonomous vehicles.

One of the benefits of LSTMs is that they are able to handle sequences of varying lengths and can be trained using standard backpropagation algorithms. Additionally, LSTMs can be stacked to form deeper architectures, which allows them to learn even more complex patterns in the data.

Despite the many advantages of LSTMs, they also have some limitations. One of the main challenges with LSTMs is that they can require a large amount of data to train effectively, which can make them impractical for some applications. Additionally, LSTMs can be computationally expensive to train and use, which can be a problem for some applications.

Effectively, LSTM is a powerful type of RNN that is capable of learning and remembering long-term dependencies in sequential data. It has been widely used in various applications and has shown promising results. However, LSTMs also have some limitations, such as the need for large amounts of data and high computational costs. Despite these limitations, LSTMs remain a popular choice for many sequential learning tasks.

In out Project,the model is used to evaluate the data and determine losses over the course of multiple epochs. Specifically, both PyTorch and TensorFlow offer LSTM functionality in each of their respective frameworks allowing for the analysis of our data to be performed. However, there still needed to be hyperparamter tuning in order to result in relevant results and useful information. 

###### **2.Accelerators:**

***Google Colab**:*\\
Specific Settings:
- Standard: NVIDIA T4 Tensor Core with 789GB [specs](https://www.nvidia.com/en-us/data-center/tesla-t4/) or K80 with 1197GB [specs](https://www.nvidia.com/en-gb/data-center/tesla-k80/)
- Premium: NVIDIA V100 [specs](https://www.nvidia.com/en-us/data-center/v100/)/ A100 Tensor Core [specs](https://www.gigabyte.com/Solutions/nvidia-a100)
The premium also comes with TPU v4 with currently supports TensorFlow, Pytorch, and JAX with setup [here](https://cloud.google.com/tpu/docs/quick-starts)

NVIDIA Tensor Cores are specialized hardware components designed for deep learning and artificial intelligence workloads. They were first introduced with NVIDIA's Volta architecture and are also available in their subsequent Turing and Ampere architectures.

Tensor Cores accelerate matrix multiplication operations commonly used in deep learning, allowing for faster training and inference times. They can perform mixed-precision operations, which allow for more efficient use of memory and faster training times.

In mixed-precision operations, the Tensor Cores use lower-precision (such as half-precision) calculations for most of the matrix multiplication, but then use higher-precision (such as single-precision) calculations for the final result. This reduces memory usage and allows for faster computations.

***Nvidia GPUs**:*\\
Locally we were able to run with the NVIDIA GTX 1660 Ti [specs](https://www.asus.com/motherboards-components/graphics-cards/dual/dual-gtx1660ti-o6g/techspec/) (There might be performance decrease given it was purchased four years ago)
The newer NVIDIA GTX cards come with Tensor Cores which were made specifically to accelerate machine learning however the 1660 Ti does not have these chips

***Apple M1/M2**:* \\
M1:
It includes an 8-core CPU with 4 high-performance cores and 4 high-efficiency cores, an 8-core GPU, and a 16-core Neural Engine. The Neural Engine is specifically designed for machine learning tasks and is capable of performing up to 11 trillion operations per second. It includes specialized hardware for matrix multiplication and convolution operations commonly used in deep learning.

In addition, the M1 includes a unified memory architecture that allows the CPU, GPU, and Neural Engine to access the same pool of memory, providing faster data transfer and reducing the need to copy data between different memory banks. Overall, the M1's specs make it a capable processor for machine learning tasks, particularly for small to medium-sized models. However, for larger models, it may be outperformed by dedicated GPUs or specialized processors such as the NVIDIA Tensor Cores.

M2:
The CPU and GPU is 20% and 28% faster compared to the M1 on ResNet50 and BERT for one epoch. M2 improves upon the M1 even further with an 18 percent faster CPU, a 35 percent more powerful GPU, and a 40 percent faster Neural Engine with process speeds up to 15.8 trillion operations per second. The chip delivers 50 percent more memory bandwidth compared to M1 and up to 24 GB of fast unified memory

***NCSA HAL**:*\\
HAL system provides GPU profile functionality via NVIDIA Nsight System CLI and NVIDIA Nsight Compute CLI. Users can generate the profile result files on HAL system and then download them to their local machine to visualize. Hardware equipements in HAL:
* 16 IBM AC922 nodes
* IBM 8335-GTH AC922 server
* 2x 20-core IBM POWER9 CPU @ 2.4GHz
* 256 GB DDR4
* 4x NVIDIA V100 GPUs
* 5120 cores
* 16 GB HBM 2
* 2-Port EDR 100 Gb/s IB ConnectX-5 Adapter
* 1 IBM 9006-22P storage node
* 72TB Hardware RAID array
* NFS
* 3 DDN GS400NVE Flash Arrays
* 360 TB usable, NVME SSD-based storage
* Spectrum Scale File System

NVIDIA Nsight Systems is a low overhead performance analysis tool designed to provide insights developers need to optimize their software. Unbiased activity data is visualized within the tool to help users investigate bottlenecks, avoid inferring false-positives, and pursue optimizations with higher probability of performance gains. Users will be able to identify issues, such as GPU starvation, unnecessary GPU synchronization, insufficient CPU parallelizing, and even unexpectedly expensive algorithms across the CPUs and GPUs of their target platform. In addition, NVIDIA Nsight Compute CLI (nv-nsight-cu-cli) provides a non-interactive way to profile applications from the command line. It can print the results directly on the command line or store them in a report file. It can also be used to simply launch the target application and later attach with NVIDIA Nsight Compute or another nv-nsight-cu-cli instance.

***Campus Cluster**:*\\
The CampusCluster is a link between many different servers maintained by various colleges, departments, labs and individuals at the University of Illinois at Urbana-Champaign. The central idea is as follows: Many different groups need server cycles, but most of them don’t always need that much power. In fact, most of the time those servers are inactive! The campuscluster is a server network that not only allows users to run programs on their own servers, but also allows users to request cycles from other people’s inactive servers.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/cc.png" title="campus cluster image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The basic framework of Campus Cluster.
</div>
As shown in the image, each server is connected to two types of queues, a primary and secondary. The primary queue is specific to that server and access is typically limited to the individuals/groups that own that server. The server itself priortizes taking jobs from the primary queue basically giving its owners dibs on the server time. This gives groups an incentive to buy and connect servers to the CC since their server will always priotize their work.

###### **3.Benchmarking Procedure and Results:**
***Procedure**:*\\
The procedure for benchmarking the model is relatively straightforward and repetitive. Starting with Google Colab, since they have default Jupyter Notebook functionality, they provide a means of testing directly on their GPUs and TPUs, allowing for benchmarking to occur. Additionally, they have packages that can link to Google Drive, which provides 15 GB of free storage, which can allow for the data to be stored there with ease. This connectivity then allows the user to run through the notebook, run the cells, and output results while changing very little.

However, with Google, it isn't exactly clear the exact computer components that are being worked with, and while they claim a GPU is accelerated the computation even on the free version of Colab, it is quite slow compared to the paid version, taking almost 45 minutes per epoch for the run. This is somewhat akin to testing locally, where laptop graphics processing units aren't nearly as fast as cloud platform ones. However, once opting for the paid version, the per epoch time is much faster.

While Google Colab provides ease, AWS offers many more instance types, at the cost of more complication and generally being more expensive. For instance, one cannot read the data through just Sagemaker, requiring spinning an S3 bucket and connecting to this within the instance itself. This is done with relative ease, however, similar to how Google Colab allows for connectivity to S3. It should be noted that Sagemaker does not allow connectivity to Google Drive, somewhat understandably. 

Afterwards, it is also unfortunately prohibitive to spin an instance on Sagemaker in the Ohio region, which is closest to Illinois. The default instance allowance is 0, meaning any instance that one is interested in must go through a service level agreement to increase the limit beyond 0. This is somewhat cumbersome, because it takes multiple days for approval and processing, resulting in lost time. Due to this, these preliminary results feature only three accelerated instance types using both PyTorch and TensorFlow kernels. However, once the instances are spun, they follow the same procedure as Google Colab; using a typical .ipynb Jupyter notebook resulting in ease of running and waiting. After waiting an appropriate amount of time, the results are presented.

***Results**:*

<table>
  <tr>
    <th>Platform</th>
    <th>Instance Type</th>
    <th>Kernel</th>
    <th>Price ($/hour)</th>
    <th>vCPU</th>
    <th>Memory</th>
    <th>Epoch Count</th>
    <th>Batch/Hidden Layer Size</th>
    <th>Time to Completion (s)</th>
    <th>Final Loss</th>
    <th>Mean Absolute Error</th>
  </tr>
  <tr>
    <td>AWS</td>
    <td>ml.g4dn.xlarge</td>
    <td>TensorFlow</td>
    <td>0.736</td>
    <td>4</td>
    <td>16 GiB</td>
    <td>10</td>
    <td>32</td>
    <td>4627</td>
    <td>0.0000052</td>
    <td>0.0013</td>
  </tr>
  <tr>
    <td>AWS</td>
    <td>ml.g4dn.xlarge</td>
    <td>PyTorch</td>
    <td>0.736</td>
    <td>4</td>
    <td>16 GiB</td>
    <td>10</td>
    <td>32</td>
    <td>4465</td>
    <td>1.421</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>AWS</td>
    <td>ml.g4dn.2xlarge</td>
    <td>TensorFlow</td>
    <td>0.94</td>
    <td>8</td>
    <td>32 GiB</td>
    <td>10</td>
    <td>32</td>
    <td>4219</td>
    <td>0.0000052</td>
    <td>0.0013</td>
  </tr>
  <tr>
    <td>AWS</td>
    <td>ml.g4dn.2xlarge</td>
    <td>PyTorch</td>
    <td>0.94</td>
    <td>8</td>
    <td>32 GiB</td>
    <td>10</td>
    <td>32</td>
    <td>4460</td>
    <td>1.096</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>AWS</td>
    <td>ml.g4dn.4xlarge</td>
    <td>TensorFlow</td>
    <td>1.505</td>
    <td>16</td>
    <td>64 GiB</td>
    <td>10</td>
    <td>32</td>
    <td>4162</td>
    <td>0.0000050</td>
    <td>0.0013</td>
  </tr>
  <tr>
    <td>AWS</td>
    <td>ml.g4dn.4xlarge</td>
    <td>PyTorch</td>
    <td>1.505</td>
    <td>16</td>
    <td>64 GiB</td>
    <td>10</td>
    <td>32</td>
    <td>4411</td>
    <td>0.883</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>GCP</td>
    <td>TPU</td>
    <td>TensorFlow</td>
    <td>0</td>
    <td>4</td>
    <td>12 GiB</td>
    <td>1</td>
    <td>32</td>
    <td>3758</td>
    <td>0.0000594</td>
    <td>0.0024</td>
  </tr>
</table>

\\
***Conclusion**:*\\
Accelerated computing for machine learning is a necessary procedure to hasten the time it takes to run models. Furthermore, there is certainly, at the very least, an academic market for implemeting machine learning in the high-frequency trading realm. However, there isn't much insight into how various accelerated computing instances affect the output speeds and results of ML based models for HFT. This project sought to provide preliminary results for thus, creating more talking points into how the cloud can be used for HFT machine learning models. 

It is of particular interest to compare different cloud platforms between one another, with the main three being AWS, GCP, and Azure, as well as testing locally to see what the cost-benefit analysis is of spending more on instances and if it provides better and faster results. For example, if an accelerated instance costs twice as much as another but only provides a 10% increase in speed, then it is certainly not worth the higher price. This is in fact what the preliminary results show. It can be seen from the table that the 4xlarge instance costs approximately twice as much as the xlarge one, but only provides a few minutes of faster analysis. Interestingly though, this also provides lower losses across the board for both PyTorch and Tensorflow, which may be of particular interest for accurate results. If one seeks the most minimal losses, then evidently more expensive instances do in fact have some benfit toward this, as well as providing faster results. While these results are preliminary with only a single run each at 10 epochs, they do provide some insight into what other accelerated computing types can provide. 

####  **Appendix:**

###### **1.ml.g4dn.xlarge:**
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/xlarge_tensorflow.PNG" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    TensorFlow.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/xlarge_PyTorch.PNG" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PyTorch.
</div>
###### **2.ml.g4dn.2xlarge:**
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/2xlarge_tensorflow.PNG" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    TensorFlow.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/2xlarge_PyTorch.PNG" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PyTorch.
</div>
###### **3.ml.g4dn.4xlarge:**
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/4xlarge_tensorflow.PNG" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    TensorFlow.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/4xlarge_PyTorch.PNG" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PyTorch.
</div>
###### **4.GCP-TPU:**

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/GCP_TensorFlow.PNG" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    TensorFlow.
</div>
###### **5.Overall Performance:**
<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/1_epoch.PNG" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_1/50_epoch.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
#### **Refencences:**
1.LSTM pages on Pytorch:
[https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

2.TensorFlow:
[https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)