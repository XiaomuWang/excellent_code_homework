
一. 作业1在 cluster 文件夹中，看.py文件的名字即可

二. 作业2在lanenet文件中

    1. 运行 train_lanenet.py 训练模型

    2. 数据集默认在同级目录中，已设置数据集默认路径为'./dataset/train_set',
       也可通过命令行指定数据集位置

    3. 默认不加载预训练模型，要加载预训练模型可以为'--ckpt_path'指定默认参数，
       或命令行输入预训练模型位置。网络结构默认为基于resnet18的fcn的一个编码
       一个解码器结构，要改变模型请更改'--arch'默认参数或命令行输入，要更改编
       码器数量请给'--dual_decoder'任意输入

    4. 日志文件保存 ./summary 文件夹中，可通过tensorboard --logdir path查
       看训练过程日志。模型保存在 ./check_point 文件夹中