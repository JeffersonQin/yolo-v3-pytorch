# 实验记录

从 `resnet18-voc-zqy-seed-sgd` 开始记录，之前的 ... 懒了，忘记了。

## resnet18-voc-zqy-seed-sgd [baseline]

* 日期：2022/06/03 凌晨
* 实验目的及方法
  * 换种子，找 zqy 随便要了几个种。
  * 首次记录，作为 baseline 调试。
* 改进
  * 换种子
* 效果
  * max AP@.5: 52.25% @ epoch 149
  * max VOCmAP: 52.17% @ epoch 140
* 结论
  * 低于之前的初始版本，未达到 AP@.5:58%
* 调参建议
  * 叠 epoch，两次 cosine annealing
  * 逐步修改回之前的参数：
    * lambda_scale = [1, 1, 1]
    * lambda_obj = 20.0
    * lr_linear_max = 0.1
* <details>
  <summary>参数</summary>
  <pre>
  # define hyper parameters
  # batch & gradient accumulation
  batch_size_train = 32
  batch_size_test = 64
  accum_batch_num = 2
  # epoch
  num_epoch = 160
  multi_scale_epoch = 150
  output_scale_S = 13
  # optimizer
  weight_decay = 0.0005
  momentum = 0.9
  # mix precision
  mix_precision = True
  # gradient clipping
  clip_max_norm = 20.0
  # lambda scale
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  # loss
  lambda_coord = 10.0
  lambda_noobj = 1.0
  lambda_obj = 50.0
  lambda_class = 10.0
  lambda_prior = 0.1
  epoch_prior = 60
  IoU_thres = 0.7
  scale_coord = True
  eps = 1e-6
  no_obj_v3 = True
  # learning rate scheduler
  lr_linear_max = 0.5
  lr_warmup_epoch = 30
  lr_T_half = 130
  # conf thres
  conf_thres = 0.01
  conf_ratio_thres = 0.05
  # test strategy
  test_pr_after_epoch = 10
  test_pr_batch_ratio = 1.0
  </pre>
  </details>
* 显著参数变化：
  ```python
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  lambda_obj = 50.0
  lr_linear_max = 0.5
  lr_cosine_max_1 = 0.5
  ```
* 流程
  | epoch | load_epoch | seed  |       reason       |
  | :---: | :--------: | :---: | :----------------: |
  |   0   |     -1     |  29   |    normal start    |
  |   8   |     7      |  28   |        nan         |
  |  21   |     20     |  31   |        nan         |
  |  62   |     61     |  30   | unexpected restart |
  |  64   |     63     |  37   |      test nan      |
  |  94   |     93     |  32   |        nan         |
  |  113  |    112     |  33   |  cuda async error  |
  |  130  |    129     |  47   |        nan         |
  |  133  |    132     |  36   |      test nan      |

## resnet18-voc-double-cosine-sgd

* 日期：2022/06/04 23:15
* 实验目的及方法
  * 对 `resnet18-voc-zqy-seed-sgd` 叠 epoch
  * cosine annealing 之后再叠一个 cosine annealing，看看能不能跳出 local optima 提高 mAP
  * 从 `resnet18-voc-zqy-seed-sgd` @ epoch-150 (multi-scale 结束) 开始 restore
* 改进
  * 叠 epoch, double consine annealing
  * 增加了 auto-restore 功能，不需要再手动 restore
* 效果
  * max AP@.5: 52.43% @ epoch 200, + 0.18% [baseline]
  * max VOCmAP: 52.26% @ epoch 200, + 0.11% [baseline]
* 结论
  * 与 `resnet18-voc-zqy-seed-sgd` 持平。先下降后陡增。
* 调参建议
  * 叠 epoch，上第三次 cosine annealing，降低初始学习率
  * 与 `resnet18-voc-zqy-seed-sgd` 第二点一致
* <details>
  <summary>参数</summary>
  <pre>
  # define hyper parameters
  # batch & gradient accumulation
  batch_size_train = 32
  batch_size_test = 64
  accum_batch_num = 2
  # epoch
  num_epoch = 210
  multi_scale_epoch = 200
  output_scale_S = 13
  # optimizer
  weight_decay = 0.0005
  momentum = 0.9
  # mix precision
  mix_precision = True
  # gradient clipping
  clip_max_norm = 20.0
  # lambda scale
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  # loss
  lambda_coord = 10.0
  lambda_noobj = 1.0
  lambda_obj = 50.0
  lambda_class = 10.0
  lambda_prior = 0.1
  epoch_prior = 60
  IoU_thres = 0.7
  scale_coord = True
  eps = 1e-6
  no_obj_v3 = True
  # learning rate scheduler
  lr_linear_max = 0.5
  lr_warmup_epoch = 30
  lr_cosine_max_1 = 0.5
  lr_T_half_1 = 130
  lr_cosine_max_2 = 0.5
  lr_T_half_2 = 50
  # conf thres
  conf_thres = 0.01
  conf_ratio_thres = 0.05
  # test strategy
  test_pr_after_epoch = 10
  test_pr_batch_ratio = 1.0
  </pre>
  </details>
* 显著参数变化
  ```python
  lr_linear_max = 0.5
  lr_warmup_epoch = 30
  lr_cosine_max_1 = 0.5
  lr_T_half_1 = 130
  lr_cosine_max_2 = 0.5
  lr_T_half_2 = 50
  ```
* 流程
  | epoch | load_epoch | seed  |    reason    |
  | :---: | :--------: | :---: | :----------: |
  |  150  |    149     |  13   | normal start |

## resnet18-voc-triple-cosine-sgd

* 日期：2022/6/5 13:45
* 实验目的及方法
  * 对 `resnet18-voc-double-cosine-sgd` 叠 epoch
  * 继续 fine tuning
  * 从 `resnet18-voc-double-cosine-sgd` @ epoch-200 (multi-scale 结束) 开始 restore
* 改进
  * 叠 epoch, triple cosine annealing
* 效果
  * max AP@.5: 53.33% @ epoch 265, + 1.08% [baseline]
  * max VOCmAP: 53.16% @ epoch 263, + 0.99% [baseline]
* 结论
  * 可以发现，适当的初始值可以提升一个点左右
  * 但是总体来说用处不大，不会起到决定性的区别，可以作为一个 trick 使用
* 调参建议
  * 与 `resnet18-voc-zqy-seed-sgd` 第二点一致
* <details>
  <summary>参数</summary>
  <pre>
  # define hyper parameters
  # batch & gradient accumulation
  batch_size_train = 32
  batch_size_test = 64
  accum_batch_num = 2
  # epoch
  num_epoch = 270
  multi_scale_epoch = 260
  output_scale_S = 13
  # optimizer
  weight_decay = 0.0005
  momentum = 0.9
  # mix precision
  mix_precision = True
  # gradient clipping
  clip_max_norm = 20.0
  # lambda scale
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  # loss
  lambda_coord = 10.0
  lambda_noobj = 1.0
  lambda_obj = 50.0
  lambda_class = 10.0
  lambda_prior = 0.1
  epoch_prior = 60
  IoU_thres = 0.7
  scale_coord = True
  eps = 1e-6
  no_obj_v3 = True
  # learning rate scheduler
  lr_linear_max = 0.5
  lr_warmup_epoch = 30
  lr_cosine_max_1 = 0.5
  lr_T_half_1 = 130
  lr_cosine_max_2 = 0.5
  lr_T_half_2 = 50
  lr_cosine_max_3 = 0.25
  lr_T_half_3 = 60
  # conf thres
  conf_thres = 0.01
  conf_ratio_thres = 0.05
  # test strategy
  test_pr_after_epoch = 10
  test_pr_batch_ratio = 1.0
  </pre>
  </details>
* 显著参数变化
  ```python
  lr_cosine_max_3 = 0.25
  lr_T_half_3 = 60
  ```
* 流程
  | epoch | load_epoch | seed  |    reason    |
  | :---: | :--------: | :---: | :----------: |
  |  200  |    199     |   0   | normal start |

## resnet18-voc-low-lr-sgd

* 日期：2022/6/6 12:25
* 实验目的及方法：
  * 对 `resnet18-voc-zqy-seed-sgd` 进行修改
  * 将最大学习率从 0.5 降低到 0.1，看看能不能比之前的 `resnet18-voc-cosine-obj-sgd` 高或者持平。以此来判断 `lambda_scale` 以及拉高 `lambda_obj` 的作用。
* 改进
  * 调参
* 效果
  * max AP@.5: 19.02% @ epoch 158, - 33.23% [baseline]
  * max VOCmAP: 21.7% @ epoch 154, - 30.47% [baseline]
* 结论
  * 在低学习率下，拉高 `lambda_scale` 和 `lambda_obj` 有明显的副作用，从涨势来看学习率起到了关键作用。但是还不能看出到底是 `lambda_scale` 和 `lambda_obj` 中的哪一个起到了决定性的影响。
* 调参建议
  * 继续调整 `lr`, `lambda_scale`, `lambda_obj`，观察在彼此不同取值下的区别。
* <details>
  <summary>参数</summary>
  <pre>
  # define hyper parameters
  # batch & gradient accumulation
  batch_size_train = 32
  batch_size_test = 64
  accum_batch_num = 2
  # epoch
  num_epoch = 160
  multi_scale_epoch = 150
  output_scale_S = 13
  # optimizer
  weight_decay = 0.0005
  momentum = 0.9
  # mix precision
  mix_precision = True
  # gradient clipping
  clip_max_norm = 20.0
  # lambda scale
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  # loss
  lambda_coord = 10.0
  lambda_noobj = 1.0
  lambda_obj = 50.0
  lambda_class = 10.0
  lambda_prior = 0.1
  epoch_prior = 60
  IoU_thres = 0.7
  scale_coord = True
  eps = 1e-6
  no_obj_v3 = True
  # learning rate scheduler
  lr_linear_max = 0.1
  lr_warmup_epoch = 30
  lr_cosine_max_1 = 0.1
  lr_T_half_1 = 130
  # conf thres
  conf_thres = 0.01
  conf_ratio_thres = 0.05
  # test strategy
  test_pr_after_epoch = 10
  test_pr_batch_ratio = 1.0
  </pre>
  </details>
* 显著参数变化
  ```python
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  lambda_obj = 50.0
  lr_linear_max = 0.1
  lr_cosine_max_1 = 0.1
  ```
* 流程
  | epoch | load_epoch | seed  |    reason    |
  | :---: | :--------: | :---: | :----------: |
  |   0   |     -1     |   0   | normal start |
  |  25   |     24     |  29   |     nan      |
  |  38   |     37     |  28   |     test     |
  |  40   |     39     |  30   |     test     |
  |  86   |     85     |  31   |   CPU OOM    |

## resnet18-voc-low-obj-sgd [Colab / Kaggle]

* 日期：2022/6/7
* 实验目的及方法：
  * 对 `resnet18-voc-zqy-seed-sgd` 进行修改
  * 保持其他参数不变，降低 `lambda_obj` 到 10，也就是 `class` 和 `coord` 的标准
  * 使用 Colab，所以 `batch_size=64`
* 改进
  * 调参
* 效果
  * max AP@.5: 55.68% @ epoch 150, + 0.2% [renaissance], + 3.43% [baseline]
  * max VOCmAP: 55.28% @ epoch 150, + 0.27% [renaissance], + 3.11% [baseline]
* 结论
  * 高于 baseline
  * 甚至高于 renaissance
  * 对比 [renaissance] 和 [baseline] 可得
    * `lambda_obj` 的提高有极大副作用
    * `lambda_scale` 和 `lr` 要同时提高
    * 结合其他实验可得单一使用 `lambda_scale`（`resnet18-voc-renaissance-scale-sgd`） 和单一提高 `lr`（`resnet18-voc-renaissance-high-obj-sgd`） 都会有副作用
  * 同时我们也要反思这里对比 baseline 如此巨大的提高，我们不仅要意识到这可能是 `lambda_obj` 的贡献，`batch_size` 也可能在里面起到了重要的作用。
* 调参建议
  * 证实了 `lambda_obj` 不应过高，可以继续尝试降低
* <details>
  <summary>参数</summary>
  <pre>
  # define hyper parameters
  # batch & gradient accumulation
  batch_size_train = 64
  batch_size_test = 64
  accum_batch_num = 1
  # epoch
  num_epoch = 160
  multi_scale_epoch = 150
  output_scale_S = 13
  # optimizer
  weight_decay = 0.0005
  momentum = 0.9
  # mix precision
  mix_precision = True
  # gradient clipping
  clip_max_norm = 20.0
  # lambda scale
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  # loss
  lambda_coord = 10.0
  lambda_noobj = 1.0
  lambda_obj = 20.0
  lambda_class = 10.0
  lambda_prior = 0.1
  epoch_prior = 60
  IoU_thres = 0.7
  scale_coord = True
  eps = 1e-6
  no_obj_v3 = True
  # learning rate scheduler
  lr_linear_max = 0.5
  lr_warmup_epoch = 30
  lr_cosine_max_1 = 0.5
  lr_T_half_1 = 130
  # lr_cosine_max_2 = 0.5
  # lr_T_half_2 = 50
  # lr_cosine_max_3 = 0.25
  # lr_T_half_3 = 60
  # conf thres
  conf_thres = 0.01
  conf_ratio_thres = 0.05
  # test strategy
  test_pr_after_epoch = 10
  test_pr_batch_ratio = 1.0
  </pre>
  </details>
* 显著参数变化：
  ```python
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  lambda_obj = 20.0
  lr_linear_max = 0.5
  lr_cosine_max_1 = 0.5
  ```
* 流程
  | epoch | load_epoch | seed  |      reason      |
  | :---: | :--------: | :---: | :--------------: |
  |   0   |     -1     |  31   |   normal start   |
  |   2   |     1      |  28   | Colab disconnect |
  |  12   |     11     |  29   |   Kaggle stop    |
  |  18   |     17     |  30   |   Kaggle stop    |
  |  68   |     67     |  33   |    Kaggle TLE    |
  |  121  |    120     |  27   |    Kaggle TLE    |

## resnet18-voc-renaissance-sgd

* 日期 2022/6/8 12:00
* 实验目的及方法：
  * 尝试重现 `resnet18-voc-cosine-obj-sgd`
  * 本次结果作为一个新的基准
* 改进
  * remake
* 效果
  * max AP@.5: 55.48% @ epoch 154, + 3.23% [baseline]
  * max VOCmAP: 55.01% @ epoch 154, + 2.84% [baseline]
* 结论
  * 果然高于 baseline
  * 仍未达到最开始的版本，推测原因有以下两个：
    * 当时 seed 并没有强制每次 restore 换 seed，而我们可以看到增长仍然是有可能的，所以可能即便没有换 seed，也没有过拟合，反而加速了收敛过程
    * 随机数问题
* 调参建议
  * 增加最大学习率 20% 至 50%
  * 增加 cosine annealing epoch，最大 epoch 数可以考虑突破 200
  * 更强的 data augmentation
    * 调节 augmentation 随机数
    * 增加更多 augmentation 策略
  * 进一步降低 `lambda_obj`
* <details>
  <summary>参数</summary>
  <pre>
  # define hyper parameters
  # batch & gradient accumulation
  batch_size_train = 32
  batch_size_test = 64
  accum_batch_num = 2
  # epoch
  num_epoch = 160
  multi_scale_epoch = 150
  output_scale_S = 13
  # optimizer
  weight_decay = 0.0005
  momentum = 0.9
  # mix precision
  mix_precision = True
  # gradient clipping
  clip_max_norm = 20.0
  # lambda scale
  lambda_scale_1 = 1
  lambda_scale_2 = 1
  lambda_scale_4 = 1
  # loss
  lambda_coord = 10.0
  lambda_noobj = 1.0
  lambda_obj = 20.0
  lambda_class = 10.0
  lambda_prior = 0.1
  epoch_prior = 60
  IoU_thres = 0.7
  scale_coord = True
  eps = 1e-6
  no_obj_v3 = True
  # learning rate scheduler
  lr_linear_max = 0.1
  lr_warmup_epoch = 30
  lr_cosine_max_1 = 0.1
  lr_T_half_1 = 130
  # lr_cosine_max_2 = 0.5
  # lr_T_half_2 = 50
  # lr_cosine_max_3 = 0.25
  # lr_T_half_3 = 60
  # conf thres
  conf_thres = 0.01
  conf_ratio_thres = 0.05
  # test strategy
  test_pr_after_epoch = 10
  test_pr_batch_ratio = 1.0
  </pre>
  </details>
* 显著参数变化：
  ```python
  lambda_scale_1 = 1
  lambda_scale_2 = 1
  lambda_scale_4 = 1
  lambda_obj = 20.0
  lr_linear_max = 0.1
  lr_cosine_max_1 = 0.1
  ```
* 流程
  | epoch | load_epoch | seed  |    reason    |
  | :---: | :--------: | :---: | :----------: |
  |   0   |     -1     |  28   | normal start |
  |  103  |    102     |  31   |   CPU OOM    |

## resnet18-voc-renaissance-high-obj-sgd [Colab / Kaggle]

* 日期：2022/6/8
* 实验目的及方法：
  * 在 `resnet18-voc-renaissance` 的基础上调大 `lambda_obj`
* 改进
  * 调参
* 效果
  * EARLY STOP @ epoch 50
  * max AP@.5 6.1963% @ epoch 49
  * max VOCmAP 7.97582% @ epoch 48
* 结论
  * 进一步证明 `lambda_obj` 应该压低
* 调参建议
  * 略，见 `renaissance`
* <details>
  <summary>参数</summary>
  <pre>
  # define hyper parameters
  # batch & gradient accumulation
  batch_size_train = 64
  batch_size_test = 64
  accum_batch_num = 1
  # epoch
  num_epoch = 160
  multi_scale_epoch = 150
  output_scale_S = 13
  # optimizer
  weight_decay = 0.0005
  momentum = 0.9
  # mix precision
  mix_precision = True
  # gradient clipping
  clip_max_norm = 20.0
  # lambda scale
  lambda_scale_1 = 1
  lambda_scale_2 = 1
  lambda_scale_4 = 1
  # loss
  lambda_coord = 10.0
  lambda_noobj = 1.0
  lambda_obj = 50.0
  lambda_class = 10.0
  lambda_prior = 0.1
  epoch_prior = 60
  IoU_thres = 0.7
  scale_coord = True
  eps = 1e-6
  no_obj_v3 = True
  # learning rate scheduler
  lr_linear_max = 0.1
  lr_warmup_epoch = 30
  lr_cosine_max_1 = 0.1
  lr_T_half_1 = 130
  # lr_cosine_max_2 = 0.5
  # lr_T_half_2 = 50
  # lr_cosine_max_3 = 0.25
  # lr_T_half_3 = 60
  # conf thres
  conf_thres = 0.01
  conf_ratio_thres = 0.05
  # test strategy
  test_pr_after_epoch = 10
  test_pr_batch_ratio = 1.0
  </pre>
  </details>
* 显著参数变化：
  ```python
  lambda_scale_1 = 1
  lambda_scale_2 = 1
  lambda_scale_4 = 1
  lambda_obj = 50.0
  lr_linear_max = 0.1
  lr_cosine_max_1 = 0.1
  ```
* 流程
  | epoch | load_epoch | seed  |      reason      |
  | :---: | :--------: | :---: | :--------------: |
  |   0   |     -1     |  28   |   normal start   |
  |  24   |     23     |  31   | Colab disconnect |
  |  25   |     24     |  29   | Colab disconnect |
  |  43   |     42     |  30   | Colab disconnect |

## resnet18-voc-renaissance-scale-sgd

* 日期：2022/6/10 11:30
* 实验目的及方法：
  * 给 `resnet18-voc-renaissance` 加上 `lambda_scale` 试试水
* 改进
  * 增加 `lambda_scale`
* 效果
  * EARLY STOP @ epoch 25
  * max AP@.5 1.702% @ epoch 24
  * max VOCmAP 1.984% @ epoch 24
* 结论
  * renaissance 条件下，即低学习率条件下，`lambda_scale` 会对效果产生极大的副作用
* 调参建议
  * 不建议在低学习率下使用 `lambda_scale`
  * 可以考虑在拉高学习率的同时再试试 `lambda_scale`
* <details>
  <summary>参数</summary>
  <pre>
  # define hyper parameters
  # batch & gradient accumulation
  batch_size_train = 32
  batch_size_test = 64
  accum_batch_num = 2
  # epoch
  num_epoch = 160
  multi_scale_epoch = 150
  output_scale_S = 13
  # optimizer
  weight_decay = 0.0005
  momentum = 0.9
  # mix precision
  mix_precision = True
  # gradient clipping
  clip_max_norm = 20.0
  # lambda scale
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  # loss
  lambda_coord = 10.0
  lambda_noobj = 1.0
  lambda_obj = 20.0
  lambda_class = 10.0
  lambda_prior = 0.1
  epoch_prior = 60
  IoU_thres = 0.7
  scale_coord = True
  eps = 1e-6
  no_obj_v3 = True
  # learning rate scheduler
  lr_linear_max = 0.1
  lr_warmup_epoch = 30
  lr_cosine_max_1 = 0.1
  lr_T_half_1 = 130
  # lr_cosine_max_2 = 0.5
  # lr_T_half_2 = 50
  # lr_cosine_max_3 = 0.25
  # lr_T_half_3 = 60
  # conf thres
  conf_thres = 0.01
  conf_ratio_thres = 0.05
  # test strategy
  test_pr_after_epoch = 10
  test_pr_batch_ratio = 1.0
  </pre>
  </details>
* 显著参数变化：
  ```python
  lambda_scale_1 = 16
  lambda_scale_2 = 4
  lambda_scale_4 = 1
  lambda_obj = 20.0
  lr_linear_max = 0.1
  lr_cosine_max_1 = 0.1
  ```
* 流程
  | epoch | load_epoch | seed  |    reason    |
  | :---: | :--------: | :---: | :----------: |
  |   0   |     -1     |  28   | normal start |