# Controllable Decision Transformer for Recommendation

现已可在Movielens-1M数据集上运行。
步骤如下：



#### 1. 将本库下载到本地
如果下载到实验室服务器，不要下载在/home/你自己/目录下，这个目录不适合存放大文件，放不下项目数据。
请用`df -h`查看磁盘空间，每个服务器都挂载了额外硬盘，一般在/data/你自己/, 如果/data/下没有“你自己”目录，请联系管理员创造一个。

此时，在服务器上的/data/你自己/下，下载本库：
```shell
git clone https://github.com/chongminggao/CRL.git
```

#### 2. 进入CRL目录，下载数据集
以下代码请逐行运行：
```shell
cd CRL/environments/ML_1M
mkdir "data"
cd data
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
mv ml-1m/* .
cd ../../.. # return to CRL directory
```

#### 3. 安装相应的环境：
首先创造一个conda环境，并激活
```shell
conda create -n CRL python=3.11
```
然后安装相应的库（已经用清华源对pip进行加速）
```shell
conda activate CRL
sh install.sh  # 清华源加速
```

#### 4. 运行
```shell
python run_CRL.py
```
说明：
现在的run_CRL的参数是对于ML-1M数据集的，如果要运行其他数据集，需要修改run_CRL.py中的参数。
我写的比较结构化，借鉴了DeepCTR库中的feature_column的设计，以及tianshou库中的buffer设计。都已经自我包含了（不需要额外下载其他库），需要先读懂整个逻辑。
评估与训练是交替进行的。

## TODO:

- 快速在MovieLens调出性能。
- 将该逻辑实现在公开数据集[ZhihuRec](https://github.com/THUIR/ZhihuRec-Dataset?tab=readme-ov-file)以及[KuaiRand-Pure](https://kuairand.com/)数据集上。逻辑均与MovieLens处理一致。
- 现在KuaiRand-1K数据进行了简单的分析，但KuaiRand-1K数据集实在太大了，id太多跑不动，放弃。不用管1K的数据。需要单独下载KuaiRand-Pure数据集重新处理。
- 实现对比算法。

需要快速整理related work, 整理出适合作为对比算法的methods，到网上找寻相关代码。related work包括的方向：
   1. 序列推荐算法，比如SASRec，GRU4REC，参照杨正一、辛鑫论文里的实验对比算法。
   2. multi-objective推荐算法，包括:
      - 一定要引用以及读的近期综述，从这些综述中找到一些常见的且方便实现的方法实现。
        - [Multi-Objective Recommender Systems: Survey and Challenges](https://arxiv.org/pdf/2210.10309.pdf)
        - [A survey on multi-objective recommender systems](https://www.frontiersin.org/articles/10.3389/fdata.2023.1157899/full)
        - [Multi-Objective Recommendations: A Tutorial](https://arxiv.org/pdf/2108.06367.pdf)
        - [Multi-Task Deep Recommender Systems: A Survey](https://arxiv.org/pdf/2302.03525.pdf)
      - 传统在输出head端进行加权的，或者multi intent或者multi objective，看这篇论文的2.3~2.4节：[Intent-aware Ranking Ensemble for Personalized Recommendation](https://arxiv.org/pdf/2304.07450.pdf)
      - 偏策略学习（强化学习）的包括：
        - 刘子儒WWW24投稿的Multi-task Sequential Recommendation with Decision Transformer
        - 刘子儒的[Multi-Task Recommendations with Reinforcement Learning](https://arxiv.org/pdf/2302.03328.pdf)
        - 以及蔡庆芃的：[Two-Stage Constrained Actor-Critic for Short Video](https://arxiv.org/pdf/2302.01680.pdf)
        - [Prompting Decision Transformer for Few-Shot Policy Generalization](https://proceedings.mlr.press/v162/xu22g/xu22g.pdf)
   3. 可控的算法，包括传统帕累托优化的一些方法，这些方法没有强调强化学习，例如：
      - [Personalized Approximate Pareto-Efficient Recommendation](https://nlp.csai.tsinghua.edu.cn/~xrb/publications/WWW-21_PAPERec.pdf)
      - [Multi-Task Learning as Multi-Objective Optimization](https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf)
      - [A Pareto-Eficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation](http://ofey.me/papers/Pareto.pdf)
      - [Pareto Self-Supervised Training for Few-Shot Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pareto_Self-Supervised_Training_for_Few-Shot_Learning_CVPR_2021_paper.pdf)
      - 关键：[Controllable Multi-Objective Re-ranking with Policy Hypernetworks](https://arxiv.org/pdf/2306.05118.pdf)
   
   4. 已有策略算法
      - [User Retention-oriented Recommendation with Decision Transformer](https://arxiv.org/pdf/2303.06347.pdf)
      - [Causal Decision Transformer for Recommender Systems via Offline Reinforcement Learning](https://arxiv.org/pdf/2304.07920.pdf)
   

    

