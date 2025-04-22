> "What I cannot create, I do not understand." - Richard Feynman

# GPU 里的线程 
从抽象层面，忽略硬件实现，GPU 里的线程和 CPU 里的线程没什么不同，只是在 GPU 里有了不同的线程组织结构。

CUDA 线程组织结构的通俗解释：
1. Thread（线程）：最小的执行单元，就像工厂里的一个工人
2. Block（块）：一组线程的集合，就像一个工作小组
    - 同一个 Block 内的线程可以相互协作，共享内存
    - 一个 Block 最多可以有 1024 个线程
3. Grid（网格）：多个 Block 的集合，就像整个工厂
    - Grid 可以是一维、二维或三维的
    - 每个 Block 在 Grid 中都有一个唯一的 ID
 
举个例子：
假设我们要处理一个 1000x1000 的图片：
- 我们可以创建一个 32x32 的 Grid（总共 1024 个 Block）
- 每个 Block 包含 32x32 个线程（总共 1024 个线程）
- 这样就能并行处理整个图片的每个像素

## hello_world.cu
```c
#include <stdio.h>

// CUDA核函数：在GPU上执行的函数
// __global__ 表示这是一个在GPU上运行的函数
// 每个线程会执行这个函数一次
__global__ void add(float* a, float* b, float* c)
{
    // 计算当前线程的全局索引
    // blockIdx.x: 当前块的索引
    // blockDim.x: 每个块的线程数
    // threadIdx.x: 当前线程在块内的索引
    int i = blockIdx.x blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    // 定义向量大小和初始化数据
    const int N = 4;
    float a[N] = {1, 2, 3, 4};    // 第一个向量
    float b[N] = {1, 1, 1, 1};    // 第二个向量
    float c[N];                    // 结果向量
    
    // 在GPU上分配内存
    floatd_a,d_b,d_c;       // d_前缀表示这是设备(device)内存
    cudaMalloc(&d_a, N sizeof(float));
    cudaMalloc(&d_b, N sizeof(float));
    cudaMalloc(&d_c, N sizeof(float));
    
    // 将数据从主机内存复制到GPU内存 -> 从 RAM 到 VRAM
    cudaMemcpy(d_a, a, N sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动核函数
    // <<<1, N>>> 表示使用1个块，每个块有N个线程
    add<<<1, N>>>(d_a, d_b, d_c);
    
    // 将结果从GPU复制回主机内存 -> 从 VRAM 到 RAM
    cudaMemcpy(c, d_c, N sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印结果
    for(int i = 0; i < N; i++)
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    
    // 释放GPU内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
```

### 编译和运行
要编译和运行这个程序，你需要安装CUDA工具包，然后使用以下命令：

```bash
nvcc -o hw hellow_world.cu
./hw
```

TODO

# 参考
1. https://0mean1sigma.com/what-is-gpgpu-programming/
2. https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
3. https://www.youtube.com/watch?v=86FAWCzIe_4 
4. https://siboehm.com/articles/22/CUDA-MMM
5. https://www.youtube.com/watch?v=GetaI7KhbzM&list=PLU0zjpa44nPXddA_hWV1U8oO7AevFgXnT&index=2
