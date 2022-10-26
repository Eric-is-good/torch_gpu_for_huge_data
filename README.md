# torch_gpu_for_huge_data



一般情况下，使用 GPU 加速矩阵运算，如果矩阵太大，那么就会爆显存

我写了一个 BigMM 库，用于**自动拆分**大矩阵使之可以在 GPU 上加速运行

使用了 **多线程技术** 和 **pytorch 单机多卡(GPU)技术**



目前实现 API 及使用例子

```python
1.  BIGmm(mat1, mat2, Slice_size, thread_num=6, gpu_ids=[0,1,2,3])
"""
	矩阵乘法运算,输入输出为 torch.cpu
    只有中间运算在 GPU 上
    具体思路是将两个相乘的大矩阵按照行列拆分
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
b = BIGmm(a, a, [10000, 10000], thread_num=6, gpu_ids=[0,1,2,3])


```

