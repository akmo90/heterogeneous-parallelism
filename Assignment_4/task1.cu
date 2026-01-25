#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000        // размер массива
#define BLOCK 256      // число потоков в одном CUDA-блоке

// CUDA-ядро: считает частичную сумму элементов массива
__global__ void sumKernel(float* input, float* output) {

    // Shared memory — быстрая память внутри одного блока
    __shared__ float cache[BLOCK];

    // Глобальный индекс элемента массива
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Индекс потока внутри блока
    int cacheIndex = threadIdx.x;

    // Локальная сумма для одного потока
    float temp = 0;

    // Каждый поток обрабатывает несколько элементов массива
    // с шагом gridDim.x * blockDim.x
    while (tid < N) {
        temp += input[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Сохраняем частичную сумму в shared memory
    cache[cacheIndex] = temp;

    // Ждём, пока все потоки блока запишут свои данные
    __syncthreads();

    // Параллельная редукция в shared memory
    // Суммируем значения внутри блока
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    // Первый поток блока записывает сумму блока в выходной массив
    if (cacheIndex == 0)
        output[blockIdx.x] = cache[0];
}

int main() {

    // ================================
    // Создание и инициализация массива на CPU
    // ================================
    float* h_data = new float[N];
    for (int i = 0; i < N; i++)
        h_data[i] = 1.0f;   // заполняем массив единицами

    // ================================
    // Последовательное суммирование на CPU
    // ================================
    auto start_cpu = std::chrono::high_resolution_clock::now();

    float cpu_sum = 0;
    for (int i = 0; i < N; i++)
        cpu_sum += h_data[i];

    auto end_cpu = std::chrono::high_resolution_clock::now();

    // ================================
    // Подготовка памяти на GPU
    // ================================
    float *d_data, *d_out;

    // Выделяем память на GPU
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_out, 256 * sizeof(float));

    // Копируем данные с CPU на GPU
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // ================================
    // Запуск CUDA-ядра
    // ================================
    auto start_gpu = std::chrono::high_resolution_clock::now();

    // Запускаем 256 блоков по 256 потоков
    sumKernel<<<256, BLOCK>>>(d_data, d_out);

    // Ждём завершения работы GPU
    cudaDeviceSynchronize();

    // ================================
    // Получаем результат с GPU
    // ================================
    float h_out[256];
    cudaMemcpy(h_out, d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    // Суммируем частичные суммы от каждого блока
    float gpu_sum = 0;
    for (int i = 0; i < 256; i++)
        gpu_sum += h_out[i];

    auto end_gpu = std::chrono::high_resolution_clock::now();

    // ================================
    // Вывод результатов
    // ================================
    std::cout << "CPU sum = " << cpu_sum << std::endl;
    std::cout << "GPU sum = " << gpu_sum << std::endl;

    std::cout << "CPU time = "
              << std::chrono::duration<double>(end_cpu - start_cpu).count()
              << " seconds" << std::endl;

    std::cout << "GPU time = "
              << std::chrono::duration<double>(end_gpu - start_gpu).count()
              << " seconds" << std::endl;
}
