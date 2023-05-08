# GN-RRT

Training code is adapted from my repo [Torch-MTS](https://github.com/XDZhelheim/Torch-MTS).

注意数据的坐标系
- 在array和tensor中左上角是 (0, 0), 向下是第一维 (height维) 正方向, 向右是第二维 (weight维) 正方向
- 在画图的时候左下角是 (0, 0), 第一维画成了横轴, 向右是第一维正方向
- 实际上存数据的时候是一样的，只不过画图的时候把原点定在了左下角，所以一张图存的 (height, width) 但是画出来是 (width, height)
- 在做数据集的时候容易混乱, 然而就按 (H, W) 来就行, 不需要什么坐标转换; 不要惦记着画图是怎么画的, 容易把自己搞晕
