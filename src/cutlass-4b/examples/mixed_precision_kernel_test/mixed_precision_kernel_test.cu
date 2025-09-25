/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
  This example shows how to run matrix multiplication kernels using functions and data structures
  provided by CUTLASS using tensor cores; which we run on a NVIDIA Turing GPU.

  Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
  high performance kernels at scale which works for multiple problem sizes with good abstractions is
  really hard. CUTLASS solves this problem by providing simplified abstractions to compose
  multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of GPU
  easily.

  CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
  and thread-block level, they compute on their own tile-size with higher level of tile sizes being
  composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
  to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
  threadblock-tile (tile size computed by a threadblock).

  In thie example, we split variable initialization into
  1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
  can view them (logical to physical mapping)
  2. Setting up computation properties : describes how the above set matrices will be used to compute
  output of matrix multiplication.

  First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
  GEMM is D = alpha * A * B + beta * C. In CUTLASS, the kernels first compute A * B and leaves the
  rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
  operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
  alpha and beta to be equal to ElementComputeEpilogue = int32_t. As we want to use MMA instructions
  on Turing and they support 8-bit signed integer (half_t), we use data type for elements in input
  matrix A and B as half_t. Volta also supports accumulation of partial dot product to int32_t, which
  can store wider range of numbers, we use it as data type of output matrix elements and accumulation.
  We convey this to CUTLASS kernel by initializing template variables ElementAccumulator (int32_t),
  ElementComputeEpilogue (int32_t), ElementInputA (half_t), ElementInputB (half_t), ElementOutput
  (int32_t). Communicating just the data type is not enough. As the data is laid out linearly in
  memory, we have to convey the layout of matrices. We do that by initializing template variable
  LayoutInputA to column major cutlass variable, LayoutInputB to row major and LayoutOutput to row
  major. Next, we setup rules to compute alpha * X + beta * C which is called epilogue of the kernel.
  We initialize template variable EpilogueOp, which takes the data type of output ElementOutput
  (int32_t), the number of elements per vector memory access (16), data type of accumulator (int32_t)
  and data type of computation of linear combination (alpha * X + beta * C).

  Now that we setup the properties of data, we have to setup properties of computation.

  Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x256x64,
  64x64x16, 8x8x16 (MxNxK) respectively. When passed to instantiate CUTLASS GEMM kernel, it internally
  deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
  bank-conflict free manner, and ton of other variables required to compose, initialize and launch a
  high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from
  understanding and coding complicated hardware optimizations which can easily go wrong.

  CUTLASS also supports multiple MMA pipelines in a threadblock. What are MMA pipelines? MMA pipelines
  constitute the whole process of loading input data from global memory to shared memory, loading data
  from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
  sequence shows a typical mma pipeline.

  matrix in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
  output to global memory

  The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
  until the previous finished executing. There are stages in the pipeline which do not have fixed
  latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
  pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
  Finally, the pipeline in a kernel looks like

  (1) matrix in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
  mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) matrix in global
  memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
  (9) output to global memory

  This way, you can hide the second global memoroy load latency by doing computation on already loaded
  input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::Gemm template.

The next step is to initialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if
the output from CUTLASS kernel is same as reference GEMM kernel.
*/

#include <iostream>
#include <chrono>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"
#include "cutlass/numeric_types.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator_Int4 = int32_t;                 // <- data type of accumulator
using ElementAccumulator_Half = float;                 // <- data type of accumulator
using ElementComputeEpilogue_Int4 = ElementAccumulator_Int4;  // <- data type of epilogue operations
using ElementComputeEpilogue_Half = ElementAccumulator_Half;  // <- data type of epilogue operations
using ElementInputA_Int4 = cutlass::int4b_t;                       // <- data type of elements in input matrix A
using ElementInputB_Int4 = cutlass::int4b_t;                       // <- data type of elements in input matrix B
using ElementInputA_Half = cutlass::half_t;                       // <- data type of elements in input matrix A
using ElementInputB_Half = cutlass::half_t;                       // <- data type of elements in input matrix B
using ElementOutput_Half = float;                      // <- data type of elements in output matrix D
using ElementOutput_Int4 = int32_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Row Major for
// Matrix A, Column Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute

// int4 magic numbers from https://docs.nvidia.com/cutlass/media/docs/cpp/implicit_gemm_convolution.html#cutlass-device-level-convolution-operator
using ShapeMMAThreadBlock_Int4 =
cutlass::gemm::GemmShape<128, 128, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
                                          // This code section describes tile size a warp will compute
using ShapeMMAWarp_Int4 = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 64 
                                                                  // This code section describes the size of MMA op
using ShapeMMAOp_Int4 = cutlass::gemm::GemmShape<16, 8, 64>;  // <- MMA Op tile M = 8, N = 8, K = 16
using ShapeMMAThreadBlock_Half =
cutlass::gemm::GemmShape<128, 256, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
                                         // This code section describes tile size a warp will compute
using ShapeMMAWarp_Half = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64 
                                                                 // This code section describes the size of MMA op
using ShapeMMAOp_Half = cutlass::gemm::GemmShape<8, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp_Half = cutlass::epilogue::thread::LinearCombination<
ElementOutput_Half,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput_Half>::value,  // <- the number of elements per vectorized
                                                            // memory access. For a byte, it's 16
                                                            // elements. This becomes the vector width of
                                                            // math instructions in the epilogue too
    ElementAccumulator_Half,                                // <- data type of accumulator
    ElementComputeEpilogue_Half>;  // <- data type for alpha/beta in linear combination function

using EpilogueOp_Int4 = cutlass::epilogue::thread::LinearCombination<
ElementOutput_Int4,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput_Int4>::value,  // <- the number of elements per vectorized
                                                            // memory access. For a byte, it's 16
                                                            // elements. This becomes the vector width of
                                                            // math instructions in the epilogue too
    ElementAccumulator_Int4,                                // <- data type of accumulator
    ElementComputeEpilogue_Int4>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm_Int4 = cutlass::gemm::device::Gemm<ElementInputA_Int4,
      LayoutInputA,
      ElementInputB_Int4,
      LayoutInputB,
      ElementOutput_Int4,
      LayoutOutput,
      ElementAccumulator_Int4,
      MMAOp,
      SmArch,
      ShapeMMAThreadBlock_Int4,
      ShapeMMAWarp_Int4,
      ShapeMMAOp_Int4,
      EpilogueOp_Int4,
      SwizzleThreadBlock,
      NumStages>;

using Gemm_Half = cutlass::gemm::device::Gemm<ElementInputA_Half,
      LayoutInputA,
      ElementInputB_Half,
      LayoutInputB,
      ElementOutput_Half,
      LayoutOutput,
      ElementAccumulator_Half,
      MMAOp,
      SmArch>;

double run(int length_m_int4, int length_n_int4, int length_k_int4, int length_m_half, int length_n_half, int length_k_half, int gemm_type){

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size_int4(length_m_int4, length_n_int4, length_k_int4);  // <- problem size of matrix multiplication
    cutlass::gemm::GemmCoord problem_size_half(length_m_half, length_n_half, length_k_half);  // <- problem size of matrix multiplication

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA_Int4, LayoutInputA> tensor_a_int4(
            problem_size_int4.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB_Int4, LayoutInputB> tensor_b_int4(
            problem_size_int4.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput_Int4, LayoutOutput> tensor_c_int4(
            problem_size_int4.mn());  // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput_Int4, LayoutOutput> tensor_d_int4(
            problem_size_int4.mn());  // <- Create matrix D with dimensions M x N used to store output from
                                      // CUTLASS kernel
    cutlass::HostTensor<ElementOutput_Int4, LayoutOutput> tensor_ref_d_int4(
            problem_size_int4.mn());  // <- Create matrix D with dimensions M x N used to store output from
                                      // reference kernel

    cutlass::HostTensor<ElementInputA_Half, LayoutInputA> tensor_a_half(
            problem_size_half.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB_Half, LayoutInputB> tensor_b_half(
            problem_size_half.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput_Half, LayoutOutput> tensor_c_half(
            problem_size_half.mn());  // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput_Half, LayoutOutput> tensor_d_half(
            problem_size_half.mn());  // <- Create matrix D with dimensions M x N used to store output from
                                      // CUTLASS kernel
    cutlass::HostTensor<ElementOutput_Half, LayoutOutput> tensor_ref_d_half(
            problem_size_half.mn());  // <- Create matrix D with dimensions M x N used to store output from
                                      // reference kernel

                                      // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
            tensor_a_int4.host_view(),
            1,
            ElementInputA_Int4(4),
            ElementInputA_Int4(-4),
            0);  // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
            tensor_b_int4.host_view(),
            1,
            ElementInputB_Int4(4),
            ElementInputB_Int4(-4),
            0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
            tensor_c_int4.host_view(),
            1,
            ElementOutput_Int4(4),
            ElementOutput_Int4(-4),
            0);  // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
            tensor_d_int4.host_view());  // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
            tensor_ref_d_int4.host_view());  // <- fill matrix D for reference on host with zeros

    cutlass::reference::host::TensorFillRandomUniform(
            tensor_a_half.host_view(),
            1,
            ElementInputA_Half(4),
            ElementInputA_Half(-4),
            0);  // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
            tensor_b_half.host_view(),
            1,
            ElementInputB_Half(4),
            ElementInputB_Half(-4),
            0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
            tensor_c_half.host_view(),
            1,
            ElementOutput_Half(4),
            ElementOutput_Half(-4),
            0);  // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
            tensor_d_half.host_view());  // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
            tensor_ref_d_half.host_view());  // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a_int4.sync_device();
    tensor_b_int4.sync_device();
    tensor_c_int4.sync_device();
    tensor_d_int4.sync_device();
    tensor_ref_d_int4.sync_device();

    tensor_a_half.sync_device();
    tensor_b_half.sync_device();
    tensor_c_half.sync_device();
    tensor_d_half.sync_device();
    tensor_ref_d_half.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue_Int4 alpha_int4 = ElementComputeEpilogue_Int4(1);
    ElementComputeEpilogue_Int4 beta_int4 = ElementComputeEpilogue_Int4(0);

    ElementComputeEpilogue_Half alpha_half = ElementComputeEpilogue_Half(1);
    ElementComputeEpilogue_Half beta_half = ElementComputeEpilogue_Half(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm_Int4::Arguments arguments_int4{problem_size_int4,  // <- problem size of matrix multiplication
        tensor_a_int4.device_ref(),  // <- reference to matrix A on device
        tensor_b_int4.device_ref(),  // <- reference to matrix B on device
        tensor_c_int4.device_ref(),  // <- reference to matrix C on device
        tensor_d_int4.device_ref(),  // <- reference to matrix D on device
        {alpha_int4, beta_int4},          // <- tuple of alpha and beta
        split_k_slices};        // <- k-dimension split factor

    typename Gemm_Half::Arguments arguments_half{problem_size_half,  // <- problem size of matrix multiplication
        tensor_a_half.device_ref(),  // <- reference to matrix A on device
        tensor_b_half.device_ref(),  // <- reference to matrix B on device
        tensor_c_half.device_ref(),  // <- reference to matrix C on device
        tensor_d_half.device_ref(),  // <- reference to matrix D on device
        {alpha_half, beta_half},          // <- tuple of alpha and beta
        split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size_int4 = Gemm_Int4::get_workspace_size(arguments_int4);
    size_t workspace_size_half = Gemm_Half::get_workspace_size(arguments_half);

    // Allocate workspace memory
    cutlass::device_memory::allocation<cutlass::uint4b_t> workspace_int4(workspace_size_int4);
    cutlass::device_memory::allocation<cutlass::half_t> workspace_half(workspace_size_half);

    // Instantiate CUTLASS kernel depending on templates
    Gemm_Int4 gemm_op_int4;
    Gemm_Half gemm_op_half;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op_int4.can_implement(arguments_int4);
    CUTLASS_CHECK(status);
    status = gemm_op_half.can_implement(arguments_half);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op_int4.initialize(arguments_int4, workspace_int4.get());
    CUTLASS_CHECK(status);
    status = gemm_op_half.initialize(arguments_half, workspace_half.get());
    CUTLASS_CHECK(status);

    auto start = std::chrono::high_resolution_clock::now();

    // Launch initialized CUTLASS kernel
    if (gemm_type == 0)
    {
        status = gemm_op_int4();
        CUTLASS_CHECK(status);
        status = gemm_op_half();
        CUTLASS_CHECK(status);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        // std::cout << "Runtime of compute kernel with INT4 length_m : " << length_m_int4 << " length_n : " << length_n_int4 << " length_k : " << length_k_int4 
        //           << " half length_m : " << length_m_half << " length_n : " << length_n_half << " length_k : " << length_k_half
        //           << ": " << duration.count() << " seconds ("
        //           << duration_ms.count() << " milliseconds)" << std::endl;
        return duration_ms.count();
    }

    if (gemm_type == 1)
    {
        status = gemm_op_int4();
        CUTLASS_CHECK(status);
        status = gemm_op_int4();
        CUTLASS_CHECK(status);
    }

    if (gemm_type == 2)
    {
        status = gemm_op_half();
        CUTLASS_CHECK(status);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        // std::cout << "Runtime of compute kernel with INT4 length_m : " << length_m_int4 << " length_n : " << length_n_int4 << " length_k : " << length_k_int4 
        //           << " half length_m : " << length_m_half << " length_n : " << length_n_half << " length_k : " << length_k_half
        //           << ": " << duration.count() << " seconds ("
        //           << duration_ms.count() << " milliseconds)" << std::endl;
        return duration_ms.count();
    }


    // // Create instantiation for device reference gemm kernel
    // cutlass::reference::device::Gemm<ElementInputA,
    //                                  LayoutInputA,
    //                                  ElementInputB,
    //                                  LayoutInputB,
    //                                  ElementOutput,
    //                                  LayoutOutput,
    //                                  ElementComputeEpilogue,
    //                                  ElementComputeEpilogue>
    //     gemm_device;

    // // Launch device reference gemm kernel
    // gemm_device(problem_size,
    //             alpha,
    //             tensor_a.device_ref(),
    //             tensor_b.device_ref(),
    //             beta,
    //             tensor_c.device_ref(),
    //             tensor_ref_d.device_ref());

    // // Wait for kernels to finish
    // cudaDeviceSynchronize();

    // // Copy output data from CUTLASS and reference kernel to host for comparison
    // tensor_d.sync_host();
    // tensor_ref_d.sync_host();

    // // Check if output from CUTLASS kernel and reference kernel are equal or not
    // bool passed = cutlass::reference::host::TensorEquals(
    //   tensor_d.host_view(),
    //   tensor_ref_d.host_view());

    // // std::cout<<"Computed: "<<tensor_d.host_view();
    // // std::cout<<"Reference: "<<tensor_ref_d.host_view();

    // std::cout << (passed ? "Passed" : "Failed") << std::endl;

    // return (passed ? 0  : -1);
    return 0;
}

int main() {
    bool notSupported = false;

    // Turing Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 10.2. 
    //
    // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
        std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (!((props.major * 10 + props.minor) >= 75)) {
        std::cerr << "Turing Tensor Core operations must be run on a machine with compute capability at least 75."
            << std::endl;

        notSupported = true;
    }

    if (notSupported) {
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    // run(5120, 4096, 4096, 5120, 4096, 4096);
    // run(10240, 8192, 8192, 10240, 8192, 8192);
    double fp_time = 0;
    double int4_time = 0;
    for (int i = 0; i < 11; i++){
        fp_time += run(259, 256, 192, 768, 256, 192, 2);

    }
    for (int i = 0; i < 11; i++){
        int4_time += run(259, 256, 192, 509, 256, 192, 0);
    }
    std::cout<<"DeiT-Tiny speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    fp_time = 0;
    int4_time = 0;
    for (int i = 0; i < 11; i++){
        fp_time += run(259, 256, 768, 1536, 256, 768, 2);

    }
    for (int i = 0; i < 11; i++){
        int4_time += run(653, 256, 768, 883, 256, 768, 0);
    }
    std::cout<<"DeiT-Small speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    fp_time = 0;
    int4_time = 0;
    for (int i = 0; i < 11; i++){
        fp_time += run(259, 256, 768, 3072, 256, 768, 2);

    }
    for (int i = 0; i < 11; i++){
        int4_time += run(1978, 256, 768, 1094, 256, 768, 0);
    }
    std::cout<<"DeiT-Base speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    fp_time = 0;
    int4_time = 0;
    fp_time += run(0,3136,96,384,3136,96,2);
    fp_time += run(0,3136,96,384,3136,96,2);
    fp_time += run(0,784,192,768,784,192,2);
    fp_time += run(0,784,192,768,784,192,2);
    fp_time += run(0,196,384,1536,196,384,2);
    fp_time += run(0,196,384,1536,196,384,2);
    fp_time += run(0,196,384,1536,196,384,2);
    fp_time += run(0,196,384,1536,196,384,2);

    int4_time += run(217,3136,96,167,3136,96,0);
    int4_time += run(217,3136,96,167,3136,96,0);
    int4_time += run(435,784,192,333,784,192,0);
    int4_time += run(435,784,192,333,784,192,0);
    int4_time += run(869,196,384,667,196,384,0);
    int4_time += run(869,196,384,667,196,384,0);
    int4_time += run(869,196,384,667,196,384,0);
    int4_time += run(869,196,384,667,196,384,0);

    std::cout<<"swintiny speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    fp_time = 0;
    int4_time = 0;
    fp_time+=run(0,3136,96,384,3136,96,2);
    int4_time+=run(201,3136,96,183,3136,96,0);
    fp_time+=run(0,3136,96,384,3136,96,2);
    int4_time+=run(201,3136,96,183,3136,96,0);
    fp_time+=run(0,784,192,768,784,192,2);
    int4_time+=run(402,784,192,366,784,192,0);
    fp_time+=run(0,784,192,768,784,192,2);
    int4_time+=run(402,784,192,366,784,192,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,196,384,1536,196,384,2);
    int4_time+=run(803,196,384,733,196,384,0);
    fp_time+=run(0,64,768,3072,64,768,2);
    int4_time+=run(1607,64,768,1465,64,768,0);
    fp_time+=run(0,64,768,3072,64,768,2);
    int4_time+=run(1607,64,768,1465,64,768,0);

    std::cout<<"swinsmall speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    fp_time = 0;
    int4_time = 0;
    fp_time+=run(0,4096,96,384,4096,96,2);
    int4_time+=run(281,4096,96,103,4096,96,0);
    fp_time+=run(0,4096,96,384,4096,96,2);
    int4_time+=run(281,4096,96,103,4096,96,0);
    fp_time+=run(0,1024,192,768,1024,192,2);
    int4_time+=run(563,1024,192,205,1024,192,0);
    fp_time+=run(0,1024,192,768,1024,192,2);
    int4_time+=run(563,1024,192,205,1024,192,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1126,256,384,410,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1126,256,384,410,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1126,256,384,410,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1126,256,384,410,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1126,256,384,410,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1126,256,384,410,256,384,0);
    fp_time+=run(0,64,768,3072,64,768,2);
    int4_time+=run(2252,64,768,820,64,768,0);
    fp_time+=run(0,64,768,3072,64,768,2);
    int4_time+=run(2252,64,768,820,64,768,0);


    std::cout<<"swin2tiny speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    fp_time = 0;
    int4_time = 0;
    fp_time+=run(0,4096,96,384,4096,96,2);
    int4_time+=run(292,4096,96,92,4096,96,0);
    fp_time+=run(0,4096,96,384,4096,96,2);
    int4_time+=run(292,4096,96,92,4096,96,0);
    fp_time+=run(0,1024,192,768,1024,192,2);
    int4_time+=run(584,1024,192,184,1024,192,0);
    fp_time+=run(0,1024,192,768,1024,192,2);
    int4_time+=run(584,1024,192,184,1024,192,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,256,384,1536,256,384,2);
    int4_time+=run(1167,256,384,369,256,384,0);
    fp_time+=run(0,64,768,3072,64,768,2);
    int4_time+=run(2335,64,768,737,64,768,0);
    fp_time+=run(0,64,768,3072,64,768,2);
    int4_time+=run(2335,64,768,737,64,768,0);

    std::cout<<"swin2small speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    fp_time = 0;
    int4_time = 0;

    fp_time+=run(0,3136,128,512,3136,128,2);
    int4_time+=run(255,3136,128,257,3136,128,0);
    fp_time+=run(0,3136,128,512,3136,128,2);
    int4_time+=run(255,3136,128,257,3136,128,0);
    fp_time+=run(0,784,256,1024,784,256,2);
    int4_time+=run(511,784,256,513,784,256,0);
    fp_time+=run(0,784,256,1024,784,256,2);
    int4_time+=run(511,784,256,513,784,256,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,196,512,2048,196,512,2);
    int4_time+=run(1022,196,512,1026,196,512,0);
    fp_time+=run(0,64,1024,4096,64,1024,2);
    int4_time+=run(2044,64,1024,2052,64,1024,0);
    fp_time+=run(0,64,1024,4096,64,1024,2);
    int4_time+=run(2044,64,1024,2052,64,1024,0);

    std::cout<<"swinbase speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    fp_time = 0;
    int4_time = 0;

    fp_time+=run(0,4096,128,512,4096,128,2);
    int4_time+=run(275,4096,128,237,4096,128,0);
    fp_time+=run(0,4096,128,512,4096,128,2);
    int4_time+=run(275,4096,128,237,4096,128,0);
    fp_time+=run(0,1024,256,1024,1024,256,2);
    int4_time+=run(551,1024,256,473,1024,256,0);
    fp_time+=run(0,1024,256,1024,1024,256,2);
    int4_time+=run(551,1024,256,473,1024,256,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,256,512,2048,256,512,2);
    int4_time+=run(1102,256,512,946,256,512,0);
    fp_time+=run(0,64,1024,4096,64,1024,2);
    int4_time+=run(2204,64,1024,1892,64,1024,0);
    fp_time+=run(0,64,1024,4096,64,1024,2);
    int4_time+=run(2204,64,1024,1892,64,1024,0);

    std::cout<<"swin2base speed up INT4 over FP16: "<<fp_time/int4_time<<std::endl;

    return 0;

}
