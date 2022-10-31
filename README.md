### torch_gpu_for_huge_data



一般情况下，使用 GPU 加速矩阵运算，如果矩阵太大，那么就会爆显存

我写了一个 BigMM 库，用于**自动拆分**大矩阵使之可以在 GPU 上加速运行，并完美支持多 GPU

使用了 **多线程技术** 和 **pytorch 单机多卡(GPU)技术**



## API  及 性能分析

测试平台：32v cpu + 4 * T4



### 矩阵乘法

```python
1.  BIGmm(mat1, mat2, Slice_size, gpu_ids=[0，1，2，3])
"""
	矩阵乘法运算,输入输出为 torch.cpu
    只有中间运算在 GPU 上
    具体思路是将两个相乘的大矩阵按照若干行列拆分成矩阵小块
   all the mat is in cpu memory, only calculate in gpu
   :param thread_num: number of threads
   :param mat1: [m, n]
   :param mat2: [n, p]
   :param Slice_size: [a, b]  divided to [a, n] * [n, b] => [a, b]
   :param gpu_ids: the id of your gpus [0,1,2,3]
   :return: mat_out [m, p]
 """

# eg
a = torch.ones([10000000, 10000000]).float() 
b = BIGmm(a, a, [100000, 100000], gpu_ids=[0,1,2,3])


```

| 序号 | 数据规模（mat_in, slice）         | 单元运算次数 | GPU数 | 时间 | 速率  |
| ---- | --------------------------------- | ------------ | ----- | ---- | ----- |
| 1    | [40000, 40000]， [10000, 20000]   | 8            | 1     | 42   | 1x    |
| 2    | [40000, 40000]， [10000, 20000]   | 8            | 2     | 23   | 1.82x |
| 3    | [40000, 40000]， [10000, 20000]   | 8            | 4     | 18   | 2.33x |
| 4    | [100000, 100000]， [10000, 20000] | 50           | 1     | 608  | 1x    |
| 5    | [100000, 100000]， [10000, 20000] | 50           | 4     | 188  | 3.23x |

性能分析说明：

均使用方阵乘法 （float32）

单元运算指一次 torch.mm 运算（分块后矩阵乘法），不同数据规模对应相同切片实际大小并不同

小规模数据更适用于少量 GPU ，多 GPU提升很少，看速率可知（尤其是第三组）