import torch
from concurrent.futures import ThreadPoolExecutor

mat_1 = None
mat_2 = None
mat_out = None


def meta_mm(a, b, da, db, gpudevice):
    global mat_1
    global mat_2
    global mat_out
    mat1 = mat_1[a:a + da].to(gpudevice)
    mat2 = mat_2[:, b:b + db].to(gpudevice)
    mat = torch.mm(mat1, mat2)
    mat_out[a:a + da, b:b + db] = mat.cpu()


def BIGmm(mat1, mat2, Slice_size, thread_num=6, gpu_num=1):
    """
    all the mat is in cpu memory, only calculate in gpu
    :param thread_num: number of threads
    :param mat1: [m, n]
    :param mat2: [n, p]
    :param Slice_size: [a, b]  divided to [a, n] * [n, b] => [a, b]
    :param gpu_num: number of gpu
    :return: mat_out [m, p]
    """
    global mat_1
    global mat_2
    global mat_out
    mat_1 = mat1
    mat_2 = mat2
    shape1 = mat1.shape
    shape2 = mat2.shape
    mat_out = torch.rand([shape1[0], shape2[1]])
    a_index = 0
    b_index = 0

    if gpu_num == 1:
        with ThreadPoolExecutor(max_workers=thread_num) as pool:
            for i in range(shape1[0] // Slice_size[0]):
                for j in range(shape2[1] // Slice_size[1]):
                    pool.submit(meta_mm, a_index, b_index, Slice_size[0], Slice_size[1], "cuda:0")
                    b_index += Slice_size[1]

                b_index = 0
                a_index += Slice_size[0]

            if shape1[0] % Slice_size[0] != 0 or shape2[1] % Slice_size[1] != 0:
                b_index = 0
                for i in range(shape2[1] // Slice_size[1]):
                    pool.submit(meta_mm, a_index, b_index, shape1[0] - a_index, Slice_size[1], "cuda:0")
                    b_index += Slice_size[1]

                a_index = 0
                for i in range(shape1[0] // Slice_size[0]):
                    pool.submit(meta_mm, a_index, b_index, Slice_size[0], shape2[1] - b_index, "cuda:0")
                    a_index += Slice_size[0]

                pool.submit(meta_mm, a_index, b_index, shape1[0] - a_index, shape2[1] - b_index, "cuda:0")


        return mat_out
