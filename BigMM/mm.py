from time import sleep

import torch
import threading

mat_1 = None
mat_2 = None
mat_out = None
okk = []


def meta_mm(a, b, da, db, gpudevice):
    global mat_1
    global mat_2
    global mat_out
    global okk
    mat1 = mat_1[a:a + da].cuda(gpudevice)
    mat2 = mat_2[:, b:b + db].cuda(gpudevice)
    mat = torch.mm(mat1, mat2)
    mat_out[a:a + da, b:b + db] = mat.cpu()
    okk.append("ok")


def BIGmm_on_one_device(para_list, device):
    for parameters in para_list:
        meta_mm(*parameters, device)


def BIGmm(mat1, mat2, Slice_size, gpu_ids=[0]):
    """
        all the mat is in cpu memory, only calculate in gpu
        :param thread_num: number of threads
        :param mat1: [m, n]
        :param mat2: [n, p]
        :param Slice_size: [a, b]  divided to [a, n] * [n, b] => [a, b]
        :param gpu_ids: the id of your gpus [0,1,2,3]
        :return: mat_out [m, p]
    """
    global mat_1
    global mat_2
    global mat_out
    global  okk
    mat_1 = mat1
    mat_2 = mat2
    shape1 = mat1.shape
    shape2 = mat2.shape
    mat_out = torch.empty([shape1[0], shape2[1]])
    a_index = 0
    b_index = 0
    id_len = len(gpu_ids)
    para_list = []

    # 安排任务表
    for i in range(shape1[0] // Slice_size[0]):
        for j in range(shape2[1] // Slice_size[1]):
            para_list.append([a_index, b_index, Slice_size[0], Slice_size[1]])
            b_index += Slice_size[1]

        b_index = 0
        a_index += Slice_size[0]

    if shape1[0] % Slice_size[0] != 0 or shape2[1] % Slice_size[1] != 0:
        b_index = 0
        for i in range(shape2[1] // Slice_size[1]):
            para_list.append([a_index, b_index, shape1[0] - a_index, Slice_size[1]])
            b_index += Slice_size[1]

        a_index = 0
        for i in range(shape1[0] // Slice_size[0]):
            para_list.append([a_index, b_index, Slice_size[0], shape2[1] - b_index])
            a_index += Slice_size[0]

        para_list.append([a_index, b_index, shape1[0] - a_index, shape2[1] - b_index])

    para_num = len(para_list)

    if id_len == 1:
        BIGmm_on_one_device(para_list, gpu_ids[0])
    else:
        thread_list = []
        single_batch = int(para_num / id_len)
        for i in range(id_len - 1):
            thread = threading.Thread(target=BIGmm_on_one_device, args=(para_list[i*single_batch:(i+1)*single_batch], gpu_ids[i]))
            thread_list.append(thread)
        thread = threading.Thread(target=BIGmm_on_one_device,
                                  args=(para_list[(id_len-1) * single_batch:], gpu_ids[id_len-1]))
        thread_list.append(thread)

        for thread in thread_list:
            thread.start()


    while len(okk) != para_num:
        sleep(0.01)

    return mat_out

