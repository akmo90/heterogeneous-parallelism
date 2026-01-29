#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// ================= CPU РЕАЛИЗАЦИИ =================

// CPU редукция суммы
float cpuReduce(const vector<float>& data)
{
    float sum = 0.0f;
    for (float v : data)
        sum += v;
    return sum;
}

// CPU префиксная сумма (exclusive)
void cpuScan(const vector<float>& in, vector<float>& out)
{
    out[0] = 0.0f;
    for (size_t i = 1; i < in.size(); i++)
        out[i] = out[i - 1] + in[i - 1];
}

// ================= ЗАДАНИЕ 1: CUDA РЕДУКЦИЯ =================

__global__ void reduceSum(const float* input, float* output, int n)
{
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Загружаем данные в shared memory
    s[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Параллельная редукция внутри блока
    for (int step = blockDim.x / 2; step > 0; step >>= 1)
    {
        if (tid < step)
            s[tid] += s[tid + step];
        __syncthreads();
    }

    // Первый поток записывает сумму блока
    if (tid == 0)
        output[blockIdx.x] = s[0];
}

// ================= ЗАДАНИЕ 2: BLELLOCH SCAN =================
// Работает корректно для массива размером 2 * blockDim.x

__global__ void blellochScan(float* data)
{
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int offset = 1;

    // Загружаем данные в shared memory
    temp[2 * tid]     = data[2 * tid];
    temp[2 * tid + 1] = data[2 * tid + 1];

    // Upsweep
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int i1 = offset * (2 * tid + 1) - 1;
            int i2 = offset * (2 * tid + 2) - 1;
            temp[i2] += temp[i1];
        }
        offset <<= 1;
    }

    // Exclusive scan
    if (tid == 0)
        temp[2 * blockDim.x - 1] = 0;

    // Downsweep
    for (int d = 1; d <= blockDim.x; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            int i1 = offset * (2 * tid + 1) - 1;
            int i2 = offset * (2 * tid + 2) - 1;
            float t = temp[i1];
            temp[i1] = temp[i2];
            temp[i2] += t;
        }
    }

    __syncthreads();

    // Запись результата обратно
    data[2 * tid]     = temp[2 * tid];
    data[2 * tid + 1] = temp[2 * tid + 1];
}

// ================= ЗАДАНИЕ 3: ПРОИЗВОДИТЕЛЬНОСТЬ =================

int main()
{
    vector<int> sizes = {1024, 1000000, 10000000};
    vector<int> blocks = {128, 256, 512};

    for (int N : sizes)
    {
        cout << "\n=============================\n";
        cout << "Размер массива: " << N << endl;

        vector<float> h(N);
        for (int i = 0; i < N; i++)
            h[i] = rand() / (float)RAND_MAX;

        // ---------- CPU ----------
        auto c1 = chrono::high_resolution_clock::now();
        float cpu_sum = cpuReduce(h);
        auto c2 = chrono::high_resolution_clock::now();

        double cpu_time =
            chrono::duration<double, milli>(c2 - c1).count();

        cout << "CPU SUM = " << cpu_sum << endl;
        cout << "CPU TIME (ms) = " << cpu_time << endl;

        // ---------- GPU ----------
        float* d_in;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMemcpy(d_in, h.data(), N * sizeof(float),
                   cudaMemcpyHostToDevice);

        for (int block : blocks)
        {
            cout << "\nBlock size: " << block << endl;

            int grid = (N + block - 1) / block;
            float* d_out;
            cudaMalloc(&d_out, grid * sizeof(float));

            auto t1 = chrono::high_resolution_clock::now();
            reduceSum<<<grid, block, block * sizeof(float)>>>(d_in, d_out, N);
            cudaDeviceSynchronize();
            auto t2 = chrono::high_resolution_clock::now();

            double gpu_time =
                chrono::duration<double, milli>(t2 - t1).count();

            // Копируем частичные суммы
            vector<float> h_partial(grid);
            cudaMemcpy(h_partial.data(), d_out,
                       grid * sizeof(float),
                       cudaMemcpyDeviceToHost);

            // Финальная сумма на CPU
            float gpu_sum = 0;
            for (float v : h_partial)
                gpu_sum += v;

            cout << "GPU SUM = " << gpu_sum << endl;
            cout << "GPU TIME (ms) = " << gpu_time << endl;

            // ---------- SCAN (только для одного блока) ----------
            if (N >= 2 * block)
            {
                vector<float> h_scan(2 * block);
                for (int i = 0; i < 2 * block; i++)
                    h_scan[i] = h[i];

                float* d_scan;
                cudaMalloc(&d_scan, 2 * block * sizeof(float));
                cudaMemcpy(d_scan, h_scan.data(),
                           2 * block * sizeof(float),
                           cudaMemcpyHostToDevice);

                auto s1 = chrono::high_resolution_clock::now();
                blellochScan<<<1, block,
                    2 * block * sizeof(float)>>>(d_scan);
                cudaDeviceSynchronize();
                auto s2 = chrono::high_resolution_clock::now();

                cudaMemcpy(h_scan.data(), d_scan,
                           2 * block * sizeof(float),
                           cudaMemcpyDeviceToHost);

                double scan_time =
                    chrono::duration<double, milli>(s2 - s1).count();

                cout << "SCAN GPU TIME (ms) = " << scan_time << endl;
                cout << "Первые 5 элементов scan:\n";
                for (int i = 0; i < 5; i++)
                    cout << h_scan[i] << " ";
                cout << endl;

                cudaFree(d_scan);
            }

            cudaFree(d_out);
        }

        cudaFree(d_in);
    }

    return 0;
}
