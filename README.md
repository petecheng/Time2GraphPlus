# Time2GraphPlus (Time2Graph+)

This project implements the Time2Graph+ model<sup>[1]</sup>, an extension of [Time2Graph](https://petecheng.github.io/Time2Graph/)<sup>[2]</sup> which focuses on time series modeling with dynamic shapelets via graph attentions. See [project homepage](https://petecheng.github.io/Time2GraphPlus/) for model details.

## Quick Links

- [Building and Testing](#building-and-testing)
- [Usage](#usage)
- [Performance](#performance)
- [Reference](#reference)

## Building and Testing

This project is implemented primarily in Python 3.6, with several dependencies listed below. We have tested the framework on Ubuntu 16.04.5 LTS with kernel 4.4.0, and it is expected to easily build and run under a regular Unix-like system.

### Dependencies

- [Python 3.7](https://www.python.org).
  Version 3.7.0 has been tested. Higher versions are expected be compatible with current implementation, while there may be syntax errors or conflicts under python 2.x.

- [PyTorch](https://pytorch.org). 

  Version 1.7.0 has been tested. You can find installation instructions [here](https://pytorch.org/get-started/locally/). Note that the GPU support is **ENCOURAGED** as it greatly boosts training efficiency.

- [XGBoost](https://github.com/dmlc/xgboost)

  Version 1.3.3 has been tested. You can find installation instructions [here](https://xgboost.readthedocs.io/en/latest/build.html).

- [Other Python modules](https://pypi.python.org). Some other Python module dependencies are listed in ```requirements.txt```, which can be easily installed with pip:

  ```bash
  pip install -r requirements.txt
  ```

  Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.

### Building the Project

Before building the project, we recommend switching the working directory to the project root directory. Assume the project root is at ``<time2graphplus_root>``, then run command

```bash
cd <time2graphplus_root>
```

Note that we assume ``<time2graphplus_root>`` as your working directory in all the commands presented in the rest of this documentation. Then make sure that the environment variable `` PYTHONPATH``  is properly set, by running the following command (on a Linux distribution):

```bash
export PYTHONPATH=`readlink -f ./`
```

### Testing the Project (Reproducibility)

A test script ```scripts/benchmark_test.py``` is available for reproducibility on the benchmark datasets:

```markdown
python . -h
 
usage: . [-h] [--dataset] [--n_splits] [--model_cache] [--shapelet_cache] [--gpu_enable]

optional arguments:
  -h, --help        show this help message and exit
  --dataset         str, one of `ucr-Earthquakes`, `ucr-WormsTwoClass` and `ucr-Strawberry`, 
                    which we have set the optimal parameters after fine-tuning. 
                    (default: `ucr-Earthquakes`)
  --n_splits        int, number of splits in cross-validation. (default: 5)
  --model_cache	    bool, whether to use a pretrained model.(default: False)
  --shapelet_cache  bool, whether to use a pretrained shapelets set.(default: False)
  --gpu_enable      bool, whether to enable GPU usage. (default: False)
```

## Usage

Given a set of time series data and the corresponding labels, the **Time2Graph+** framework aims to learn the representations of original time series, and conduct time series classifications under the setting of supervised learning.

### Input Format 

The input time series data and labels are expected to be ```numpy.ndarray```:

```markdown
Time_Series X: 
    numpy.ndarray with shape (N x L x data_size),
    where N is the number of time series, L is the time series length, 
    and data_size is the data dimension.
Labels Y:
    numpy.ndarray with shape (N x 1), with 0 as negative, and 1 as positive samples.
```

We organize the preprocessing codes that load the *UCR* dataset in the `archive/` repo, and if you want to utilize the framework on other datasets, just preprocess the original data as the abovementioned format. 

### Main Script

Now that the input data is ready, the main script `scripts/run.py` is a pipeline example to train and test the whole framework. Firstly you need to modify the codes in the following block (*line 46-51*) to load your datasets, by reassigning `x_train, y_train, x_test, y_test` respectively.

```python
if args.dataset.startswith('ucr'):
    dataset = args.dataset.rstrip('\n\r').split('-')[-1]
    x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
    fname=dataset, length=args.seg_length * args.num_segment)
else:
    raise NotImplementedError()
```

The help information of the main script `scripts/run.py` is listed as follows:

```markdown
python . -h
 
usage: .[-h] [-- dataset] [--K] [--C] [--num_segment] [--seg_length] [--data_size] 
        [--n_splits] [--njobs] [--optimizer] [--alpha]  [--beta] [--init] 
        [--gpu_enable] [--opt_metric] [--cache] [--embed] [--embed_size] [--warp] 
        [--cmethod] [--kernel] [--percentile] [--measurement] [--batch_size] 
        [--tflag] [--scaled] [--norm] [--no_global]

optional arguments:
  -h, --help        show this help message and exit
  --dataset         str, indicate which dataset to load; 
                    need to modify the codes in line 46-51.
  --K               int, number of shapelets that try to learn
  --C               int, number of shapelet candidates used for learning shapelets
  --num_segment     int, number of segment that a time series have
  --seg_length      int, the segment length, 
                    so the length of a time series is num_segment * seg_length
  --data_size       int, the dimension of time series data
  --n_splits        int, number of cross-validation, default 5.
  --njobs           int, number of threads if using multiprocessing.
  --optimizer       str, optimizer used for learning shapelets, default `Adam`.
  --alpha           float, penalty for local timing factor, default 0.1.
  --beta            float, penalty for global timing factor, default 0.05.
  --init            int, init offset for time series, default 0.
  --gpu_enable      bool, whether to use GPU, default False.
  --opt_metric      str, metric for optimizing out-classifier, default `accuracy`.
  --cache           bool, whether to save model cache, defualt False.
  --wrap            int, warp size in greedy-dtw, default 2.
  --cmethod         str, candidate generation method, one of `cluster` and `greedy`
  --kernel          str, choice of outer-classifer, default `xgb`.
  --percentile      int, distance threshold (percentile) in graph construction, default 10
  --measurement     str, distance measurement,default `gdtw`.
  --batch_size      int, batch size, default 50
  --tflag           bool, whether to use timing factors, default True.
```

Some of the arguments may require further explanation:

- ``--K/--C``: the number of shapelets should be carefully selected, and it is highly related with intrinsic properties of the dataset. And in our extensive experiments, `C` is often set 10 or 20 times of `K` to ensure that we can learn from a large pool of candidates.
- ``--percentile`` , ``--alpha`` and `--beta`: we have conduct fine-tuning on several datasets, and in most cases we recommend the default settings, although modifying them may bring performance increment, as well as drop.

### Demo

We include all three benchmark *UCR* datasets in the ``dataset`` directory, which is a subset of *UCR-Archive* time series dataset. See [Data Sets](#data-sets) for more details. Then a demo script is available by calling `scripts/run.py`, as the following:

```shell
python scripts/run.py --dataset ucr-Earthquakes --K 50 --C 500 
--num_segment 21 --seg_length 24 --data_size 1 --gpu_enable
```

## Evaluation

### Data Sets

The three benchmark datasets reported in <sup>[1]</sup> was made public by [UCR](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), and detailed descriptions can be referred in [Time2Graph](https://github.com/petecheng/Time2Graph).
Furthermore, we apply the proposed *Time2Graph* model on three real-world scenarios: Electricity Consumption Records (**ECR**) and Elderly Electricity Records (**EER**) provided by State Grid of China, and Network Traffic Flow (**NTF**) from China Telecom. Detailed dataset descriptions can be found in our paper. The performance increment compared with existing models clearly demonstrate the effectiveness of the framework, and below we list the final results along with several popular baselines.

### Performance

| Accuracy on UCR(%) | Earthquakes | WormsTwoClass | Strawberry |
| :----------------: | :---------: | :-----------: | :--------: |
|       NN-DTW       |    70.31    |     68.16     |   95.53    |
|        TSF         |    74.67    |     68.51     |   96.27    |
|         FS         |    74.66    |     70.58     |   91.66    |
|      Time2Graph     |    79.14    |     72.73     |   96.76    |
|     Time2Graph+     |  **77.70**  |   **71.43**   | **96.49**  |

| Performance on ECR(%) | Precision |  Recall   |    F1     |
| :-------------------: | :-------: | :-------: | :-------: |
|        NN-DTW         |   15.52   |   18.15   |   16.73   |
|          TSF          |   26.32   |   2.02    |   3.75    |
|          FS           |   10.45   |  79.84*   |   18.48   |
|      Time2Graph       |   30.10   |  40.26    |   34.44 |
|      Time2Graph+       | **35.94** | **44.81** | **39.88** |

| Performance on NTF(%) | Precision |  Recall   |    F1     |
| :-------------------: | :-------: | :-------: | :-------: |
|        NN-DTW         |   33.20   |   43.75   |   37.75   |
|          TSF          |   57.52   |   33.85   |   42.62   |
|          FS           |   63.55   |   35.42   |   45.49   |
|      Time2Graph       |   71.52   |   56.25   |   62.97   |
|      Time2Graph+       | **97.62** | **48.81** | **65.08** |

| Performance on NTF(%) | Precision |  Recall   |    F1     |
| :-------------------: | :-------: | :-------: | :-------: |
|        NN-DTW         |   33.20   |   43.75   |   37.75   |
|          TSF          |   57.52   |   33.85   |   42.62   |
|          FS           |   63.55   |   35.42   |   45.49   |
|      Time2Graph       |  71.52    |   56.25   |   62.97   |
|      Time2Graph+       | **32.80** | **66.19** | **43.87** |


Please refer to our paper <sup>[1]</sup> for detailed information about the experimental settings, the description of unpublished data sets, the full results of our experiments, along with ablation and observational studies.
Last but not least, we have deployed Time2Graph+ model in a real-world application, elderly recognition, cooperated with State Grid of China, Jinhua Zhejiang. See [Project Homepage](https://petecheng.github.io/Time2GraphPlus/) and our paper for details.

## Reference
[1] Cheng, Z; Yang, Y; Jiang, S; Hu, W; Ying, Z and Chai, Z, 2021, Time2Graph: Bridging Time Series and Graph Representation Learning via Multiple Attentions, under review.

[2] Cheng, Z; Yang, Y; Wang, W; Hu, W; Zhuang, Y and Song, G, 2020, Time2Graph: Revisiting Time Series Modeling with Dynamic Shapelets, In AAAI, 2020

```
@inproceedings{cheng2020time2graph,
  title = "{Time2Graph: Revisiting Time Series Modeling with Dynamic Shapelets}", 
  author = {{Cheng}, Z. and {Yang}, Y. and {Wang}, W. and {Hu}, W. and {Zhuang}, Y. and {Song}, G.}, 
  booktitle={Proceedings of Association for the Advancement of Artificial Intelligence (AAAI)},
  year = 2020, 
} 
```