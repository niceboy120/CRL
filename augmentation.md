# 数据增强说明

数据增强的代码位于`environments/base_data.py`中，主要有两个函数：
- `augment_matrix`: 直接在评分矩阵上进行增强，根据每个用户原有的交互数据，人为构造新的交互数据。 增强得到的pandas Dataframe将会被保存。
    - random: 为每个用户，随机从其原有交互数据中采样交互组成新的序列（未去重）
    - rating：为每个用户挑选评分最高的item，重新组成新的序列（未去重）
    - diversity：通过贪心的方式，每次选择拥有当前未见过的tags的item来组成序列，如果所有tags都已见过，则随机采样一个item然后重复生成
- `augment_sequence`: 为现有的状态添加更优的决策序列，从而得到增强的序列数据，序列构造的方式与上述方式一致，但都经过了**去重处理**

运行数据增强的参数有三个：
- `augment_type : "mat" or "seq"`: 选择上述两种增强方式的一种
- `augment_rate : float`: 选择增强数据的比例，对于matrix可以选择5~10倍，对于sequence可以选择0.1~0.5
- `augment_strategies : list of "random", "rating" and "diversity"`: 选择数据增强的内容，可以多选

> 注意: 运行数据增强的代码之前，应该先用未增强的参数（即`augment_rate = 0`）跑一遍，从而避免增强的数据对帕累托前沿、评分矩阵等的影响

运行示例：
```shell
# run ML-1M with matrix augmentation
python run_CRL.py --env "ml-1m" --cuda_id 1 --augment_type mat --augment_rate 1 --augment_strategies random rating diversity --epoch 50

# run KuaiRand-Pure with sequence augmentation
python run_CRL.py --env "KuaiRand-Pure" --cuda_id 2 --augment_type seq --augment_rate 0.1 --augment_strategies random rating diversity --epoch 50
```