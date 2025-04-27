> "知己知彼，百战不殆。"    —— 孙武

# GPU 架构浅析

理解 GPU 的硬件结构和内存工作方式是写出高性能 CUDA 代码的关键。就像了解战场地形对将军至关重要一样，了解 GPU 的流式多处理器（SM）、CUDA 核心以及不同类型的内存（如高速但容量小的共享内存和寄存器，以及容量大但访问速度较慢的全局内存）能帮助我们更好地组织线程、管理数据，从而充分发挥 GPU 的并行计算能力。

## 查询 GPU 设备属性 (query_device_properties.cu)
了解你正在使用的 GPU 的具体参数非常重要，例如它有多少个 SM、每个块最多支持多少线程等。CUDA 提供了一些函数来获取这些信息。

## 关键参数关系与实际应用
- SM 数量决定可并行调度的 Block 数，SM 越多并行能力越强。
- 每个 Block 的最大线程数（如 1024）限制了单个 Block 的规模。
- 每个 SM 的最大线程数（如 2048）影响同一时刻可运行线程总数。
- Warp 是最小调度单位，Block 线程数最好为其整数倍。
- 共享内存和寄存器资源决定 Block 能否被调度，资源不足会限制并发。

## 举例说明：3070 能否一次处理一张 1000 * 1000 的图片？

### 3070 参数（用下面的程序）
```
--- 设备 0: Intel(R) Xeon(R) E-2286M  CPU @ 2.40GHz ---
  计算能力: 7.0
  全局内存总量: 56489.25 MB
  每个块的最大线程数: 1024
  SM (流式多处理器) 数量: 16
  每个 SM 的最大 Warp 数: 2048
  Warp 大小: 8
  核心时钟频率: 2.40 GHz
  共享内存大小/块: 32768 Bytes
  寄存器数量/块: 16
  块的最大维度: (1024, 1024, 1024)
  网格的最大维度: (2147483647, 2147483647, 2147483647)
Exit status: 0
```

### 分析
首先说答案：不能。因为虽然我们希望“每个像素分配一个线程”，即总共需要 1000 × 1000 = 1,000,000 个线程，但 RTX 3070 的硬件并不能让所有线程“同时”运行。原因如下：
- 3070 的 SM（流式多处理器）数量有限，每个 SM 能同时调度的线程数也有限。例如，每个 SM 最多支持 2048 个线程，16 个 SM 总共最多支持 16 × 2048 = 32,768 个线程并发执行。
- 线程实际的最小调度单位是 Warp（如本例中为 8，常见为 32），只有这么多线程能真正“同时”在硬件上运行。其余线程会被分批次调度，等前一批线程执行完后再轮到下一批。

所以，虽然 CUDA 允许你一次性启动 1,000,000 个线程，但这些线程会被分批次（分多个 Warp）在硬件上轮流执行，而不是所有线程同时并行。3070 的硬件资源决定了它无法“真正同时”处理 1,000,000 个线程，但 CUDA 的调度机制会自动帮你完成所有像素的计算，程序员无需手动干预。

```c
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int dev_count;
    // 获取可用的 CUDA 设备数量
    cudaError_t count_err = cudaGetDeviceCount(&dev_count);
    if (count_err != cudaSuccess) {
        printf("获取设备数量失败: %s\n", cudaGetErrorString(count_err));
        return 1;
    }

    if (dev_count == 0) {
        printf("未找到可用的 CUDA 设备。\n");
        return 0;
    }

    printf("找到 %d 个 CUDA 设备:\n", dev_count);

    // 遍历每个设备并打印其属性
    for (int i = 0; i < dev_count; i++) {
        cudaDeviceProp dev_prop;
        // 获取指定设备的属性
        cudaError_t prop_err = cudaGetDeviceProperties(&dev_prop, i);
        if (prop_err != cudaSuccess) {
            printf("  获取设备 %d 的属性失败: %s\n", i, cudaGetErrorString(prop_err));
            continue;
        }

        printf("\n--- 设备 %d: %s ---\n", i, dev_prop.name);
        printf("  计算能力: %d.%d\n", dev_prop.major, dev_prop.minor);
        printf("  全局内存总量: %.2f MB\n", (float)dev_prop.totalGlobalMem / (1024.0f * 1024.0f));
        printf("  每个块的最大线程数: %d\n", dev_prop.maxThreadsPerBlock);
        printf("  SM (流式多处理器) 数量: %d\n", dev_prop.multiProcessorCount);
        printf("  每个 SM 的最大 Warp 数: %d\n", dev_prop.maxThreadsPerMultiProcessor / dev_prop.warpSize);
        printf("  Warp 大小: %d\n", dev_prop.warpSize);
        printf("  核心时钟频率: %.2f GHz\n", (float)dev_prop.clockRate / (1000.0f * 1000.0f));
        printf("  共享内存大小/块: %zu Bytes\n", dev_prop.sharedMemPerBlock);
        printf("  寄存器数量/块: %d\n", dev_prop.regsPerBlock);
        printf("  块的最大维度: (%d, %d, %d)\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
        printf("  网格的最大维度: (%d, %d, %d)\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
    }

    return 0;
}
```

## 编译和运行

```bash
# 编译
nvcc -o query_props query_device_properties.cu
# 运行
./query_props
```

## 总结与关键点回顾
理解 GPU 的硬件参数和内存结构，是高效 GPU 编程的基础。通过这一些我们初步了解了 GPU 的并行计算资源、线程调度方式以及常见的资源限制，后面我们将在这些概念的基础上，了解如何优化 GPU 程序，打开并行计算的大门。

# 参考
1. https://0mean1sigma.com/chapter-3-gpu-compute-and-memory-architecture/ 
2. https://leetgpu.com/playground/7a8c6430-34bc-41d3-9b4d-b758b216a023