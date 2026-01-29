#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000     // размер массива
#define BLOCK 1024   // число потоков в одном CUDA-блоке

// CUDA-ядро для вычисления префиксной суммы (scan)
__global__ void scanKernel(int* data) {

    // Shared memory — быстрая память, доступная всем потокам внутри одного блока
    __shared__ int temp[BLOCK];

    // Номер потока внутри блока
    int t = threadIdx.x;

    // Глобальный индекс элемента массива
    int idx = blockIdx.x * blockDim.x + t;

    // Копируем элементы из глобальной памяти в shared memory
    // Если индекс выходит за пределы массива — кладём 0
    if (idx < N)
        temp[t] = data[idx];
    else
        temp[t] = 0;

    // Ждём, пока все потоки блока загрузят свои данные
    __syncthreads();

    // Параллельный inclusive prefix sum (scan) внутри блока
    // На каждом шаге поток прибавляет значение слева
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int val = 0;

        // Берём элемент, находящийся на offset позиций левее
        if (t >= offset)
            val = temp[t - offset];

        __syncthreads();      // синхронизация перед обновлением
        temp[t] += val;      // прибавляем к текущему элементу
        __syncthreads();      // синхронизация после обновления
    }

    // Записываем результат обратно в глобальную память
    if (idx < N)
        data[idx] = temp[t];
}

int main() {

    // ============================
    // Создание и инициализация массива на CPU
    // ============================
    int* h = new int[N];
    for (int i = 0; i < N; i++)
        h[i] = 1;   // массив из единиц

    // ============================
    // Выделение памяти на GPU
    // ============================
    int* d;
    cudaMalloc(&d, N * sizeof(int));

    // Копируем массив с CPU на GPU
    cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice);

    // ============================
    // Запуск GPU-версии scan
    // ============================
    auto start_gpu = std::chrono::high_resolution_clock::now();

    // Запускаем kernel: N/BLOCK блоков по BLOCK потоков
    scanKernel<<<N / BLOCK, BLOCK>>>(d);

    // Ждём, пока GPU завершит вычисления
    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();

    // Копируем результат обратно на CPU
    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);

    // ============================
    // Последовательная версия scan на CPU
    // ============================
    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < N; i++)
        h[i] += h[i - 1];

    auto end_cpu = std::chrono::high_resolution_clock::now();

    // ============================
    // Вывод результатов
    // ============================
    std::cout << "Last element (should be 1000000): " << h[N - 1] << std::endl;

    std::cout << "CPU time: "
              << std::chrono::duration<double>(end_cpu - start_cpu).count()
              << " seconds" << std::endl;

    std::cout << "GPU time: "
              << std::chrono::duration<double>(end_gpu - start_gpu).count()
              << " seconds" << std::endl;
}
