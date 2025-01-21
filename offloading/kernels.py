from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import custom_op
def init_to_zero(name):
    """
    Returns a function that initializes the specified tensor to zero.
    :param name: Name of the tensor to initialize.
    :return: A function that initializes the tensor to zero.
    """
    def init_fn(args):
        # args 是一个字典，包含内核的参数
        if name in args:
            args[name].zero_()  # 将指定张量初始化为零
    return init_fn
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)

@triton.jit
def gather_gemv_elemul_flag_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    X_1,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """
    Kernel for computing Y = A[IDX, :] @ X) * X_1, where A is a
    dense matrix with M rows and N columns.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, N)
    - Input X_1 has shape (BATCHSIZE, M)
    - A has shape (M, N)
    - IDX has shape (M), where M is the flag for non-zero rows in A
    - Output has shape (BATCHSIZE, M)
    """
    # EVEN_N is asserted to be true
    start_m = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A and B
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X_1 = X_1 + rm
    X = X + rn

    
    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float16)
    x1_0 = tl.load(X_1, mask=idx, other=0.0)
    i_mask = idx[:, None]
    for n in range(N, 0, -BLOCK_N):
        a = tl.load(A, mask=i_mask, other=0.0)
        x0 = tl.load(X)
        acc0 += tl.sum(a * x0[None, :], 1)
        A += BLOCK_N
        X += BLOCK_N
    acc0 = (acc0*tl.sigmoid(acc0.to(tl.float32)) * x1_0).to(tl.float16)
    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back result
    Y = Y + rm
    # acc = acc0 * X_1
    tl.store(Y, acc0, mask=rm < M)

@custom_op("mylib::gather_gemv_elemul_flag_3d", mutates_args=())
def gather_gemv_elemul_flag_3d(
    x: torch.Tensor,
    x_1: torch.Tensor,
    wup: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = activation(x @ wgate[idx, :].T) * (x @ wup[idx, :].T).
    :param x: input tensor, (batch, N)
    :param x_1: input tensor, (batch, Z)
    :param wup: up weigth matrix, (Z, N)
    :param idx: flags, (Z,)
    :return: result tensor, (batch, N)
    """
    Z, N = wup.shape
    beam_width, seq_len, _ = x.shape
    # assert x.shape == (batch, N)
    # assert x_1.shape == (batch, Z)
    # assert seq_len == 1
    # assert beam_width >= 1 and beam_width <= 4
    x = x.contiguous()
    x_1 = x_1.contiguous()
    # if wup.stride(1) > 1:
    #     wup = wup.contiguous()
    # assert (
    #     x.dtype == wup.dtype
    # ), f"Input and weight must have the same dtype, got {x.dtype} and {wup.dtype}"
    # output1 = torch.empty(beam_width, seq_len, Z, device=x.device, dtype=torch.float16)
    output = torch.empty(beam_width, seq_len, Z, device=x.device, dtype=torch.float16)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(Z, META["BLOCK_M"]),)  # noqa

    gather_gemv_elemul_flag_kernel[grid](
        output,  # data ptrs
        wup,
        x,
        x_1,
        idx,
        Z,  # shapes
        N,
        Z // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        wup.stride(0),  # strides
        beam_width,  # Can't use kwargs because auto-tuner requires args
    )
    return output.to(x.dtype)

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_transposed_gemv_flag_atomicadd_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    
    
    a = tl.load(A, mask=idx[:, None], other=0.0)
    x0 = tl.load(X)#, mask=idx, other=0.0) # if flag_gemv is correct, this will be unnecessary.
    acc0 = tl.sum(a * x0[:, None], 0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y, acc0, mask=rn < N)

@custom_op("mylib::gather_transposed_gemv_flag_3d", mutates_args=())
def gather_transposed_gemv_flag_3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    Z, N = weight.shape
    beam_width, seq_len, _ = x.shape
    assert x.shape[2] == Z
    x = x.contiguous()
    # if weight.stride(1) > 1:
    #     weight = weight.contiguous()

    output = torch.empty(
        beam_width,
        seq_len,
        N,
        device=x.device,
        dtype=torch.float16,
    )

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(Z, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )  # noqa
    kernel = gather_transposed_gemv_flag_atomicadd_kernel
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        Z,  # shapes
        N,
        Z // 128,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
        beam_width,  # can't use kwargs because auto-tuner requires args
    )
    # return output
    return output.to(dtype=weight.dtype)


@torch.library.register_fake("mylib::gather_gemv_elemul_flag_3d")
def gather_gemv_elemul_flag_3d_fake_impl(x, x_1, wup, idx):
    """
    Fake impl for 'mylib::gather_gemv_elemul_flag_3d'.
    只做输出形状/类型推断，不进行真实计算。
    """
    # 根据源码注释：
    # wup.shape == (Z, N)
    # x.shape == (beam_width, seq_len, ?)，最后一维通常是 Z
    beam_width, seq_len, _ = x.shape
    Z, N = wup.shape
    # 从您的注释中可知，最终 output.shape == (beam_width, seq_len, Z)
    out_shape = (beam_width, seq_len, Z)
    # 假设输出dtype与 x.dtype 一致
    # 注意：原代码中是 return output.to(x.dtype)
    # 这里可以选 x.dtype / wup.dtype 均可，看最终实现
    out_dtype = x.dtype
    # 返回一个“假”张量
    return x.new_empty(out_shape, dtype=out_dtype)

# 3. 为第二个算子(mylib::gather_transposed_gemv_fla)注册fake_impl
@torch.library.register_fake("mylib::gather_transposed_gemv_flag_3d")
def gather_transposed_gemv_fla_fake_impl(x, weight, idx):
    """
    Fake impl for 'mylib::gather_transposed_gemv_fla'.
    只做输出形状/类型推断，不进行真实计算。
    """
    # weight.shape == (Z, N)
    # x.shape == (beam_width, seq_len, Z)
    beam_width, seq_len, Z = x.shape
    # 最终 output.shape == (beam_width, seq_len, N)
    N = weight.shape[1]
    out_shape = (beam_width, seq_len, N)
    out_dtype = x.dtype  # 也可改为 weight.dtype, 取决于真实实现
    return x.new_empty(out_shape, dtype=out_dtype)
def gather_gemv_elemul_flag_3d_fake_autograd( x,grad,):
   
    return ( None,None,None,None,None,)
def gather_transposed_gemv_flag_fake_autograd(x,grad,):
    return (None,None,None,None,)
torch.library.register_autograd(
     "mylib::gather_gemv_elemul_flag_3d", gather_gemv_elemul_flag_3d_fake_autograd, 
)
torch.library.register_autograd(
     "mylib::gather_transposed_gemv_flag_3d", gather_transposed_gemv_flag_fake_autograd,
)