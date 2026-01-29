#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>

using namespace std;

// ================= CUDA ЯДРО =================
// Умножение каждого элемента на 2
__global__ void gpuProcess(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = data[idx] * 2.0f;
}

// ================= CPU (OpenMP) =================
void cpuProcess(vector<float>& data)
{
    int n = data.size();

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        data[i] = data[i] * 2.0f;
    }
}

// ================= MAIN =================
int main()
{
    const int N = 1'000'000;   // размер массива
    const int BLOCK = 256;

    cout << "Размер массива N = " << N << endl;

    // Исходный массив
    vector<float> data(N, 1.0f);

    // ================= CPU =================
    vector<float> cpuData = data;

    auto c1 = chrono::high_resolution_clock::now();
    cpuProcess(cpuData);
    auto c2 = chrono::high_resolution_clock::now();

    double cpuTime =
        chrono::duration<double, milli>(c2 - c1).count();

    cout << "[CPU] Time (ms): " << cpuTime << endl;

    // ================= GPU =================
    vector<float> gpuData = data;

    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, gpuData.data(),
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    int grid = (N + BLOCK - 1) / BLOCK;

    auto g1 = chrono::high_resolution_clock::now();
    gpuProcess<<<grid, BLOCK>>>(d_data, N);
    cudaDeviceSynchronize();
    auto g2 = chrono::high_resolution_clock::now();

    cudaMemcpy(gpuData.data(),
               d_data,
               N * sizeof(float),
               cudaMemcpyDeviceToHost);

    double gpuTime =
        chrono::duration<double, milli>(g2 - g1).count();

    cout << "[GPU] Time (ms): " << gpuTime << endl;

    // ================= HYBRID =================
    vector<float> hybridData = data;

    int half = N / 2;

    float* d_half;
    cudaMalloc(&d_half, half * sizeof(float));

    // Копируем вторую половину на GPU
    cudaMemcpy(d_half,
               hybridData.data() + half,
               half * sizeof(float),
               cudaMemcpyHostToDevice);

    auto h1 = chrono::high_resolution_clock::now();

    // CPU обрабатывает первую половину
    #pragma omp parallel for
    for (int i = 0; i < half; i++)
    {
        hybridData[i] = hybridData[i] * 2.0f;
    }

    // GPU обрабатывает вторую половину
    int gridHalf = (half + BLOCK - 1) / BLOCK;
    gpuProcess<<<gridHalf, BLOCK>>>(d_half, half);
    cudaDeviceSynchronize();

    // Копируем результат обратно
    cudaMemcpy(hybridData.data() + half,
               d_half,
               half * sizeof(float),
               cudaMemcpyDeviceToHost);

    auto h2 = chrono::high_resolution_clock::now();

    double hybridTime =
        chrono::duration<double, milli>(h2 - h1).count();

    cout << "[HYBRID] Time (ms): " << hybridTime << endl;

    // ================= ПРОВЕРКА =================
    cout << "\nПроверка первых 5 элементов:\n";
    for (int i = 0; i < 5; i++)
    {
        cout << "CPU: " << cpuData[i]
             << " | GPU: " << gpuData[i]
             << " | HYBRID: " << hybridData[i] << endl;
    }

    cudaFree(d_data);
    cudaFree(d_half);

    return 0;
}
