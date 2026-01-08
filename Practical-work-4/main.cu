#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>

#define THREADS 256


// 1. Генерация данных на CPU
// Функция заполняет массив случайными числами
// Выполняется на CPU 
void generateArray(std::vector<float>& arr) {
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = static_cast<float>(rand() % 100);
    }
}

// 2. Редукция суммы (глобальная память)
// Каждый поток читает один элемент из глобальной памяти
// и добавляет его к общему результату 
__global__ void sumGlobal(float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, data[idx]);
    }
}

// 3. Редукция суммы (глобальная + разделяемая память)
// Используется shared memory для уменьшения количества
// обращаемся к глобальной памяти
__global__ void sumShared(float* data, float* result, int n) {

    // Разделяемая память 
    __shared__ float shared[THREADS];

    int tid = threadIdx.x;                         // локальный индекс потока
    int idx = blockIdx.x * blockDim.x + tid;       // глобальный индекс

    // Загрузка данных из глобальной памяти в shared memory
    if (idx < n)
        shared[tid] = data[idx];
    else
        shared[tid] = 0.0f;

    // Синхронизация потоков внутри блока
    __syncthreads();

    // Параллельная редукция внутри блока
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    // Первый поток блока добавляет сумму блока
    // в глобальный результат
    if (tid == 0)
        atomicAdd(result, shared[0]);
}

// 4. Измерение времени выполнения редукции
// Функция запускает один из вариантов редукции
// и измеряет время выполнения
float measureSum(int n, bool useShared) {

    // Выделение и генерация массива на CPU
    std::vector<float> h_data(n);
    generateArray(h_data);

    // Указатели на память GPU
    float* d_data;
    float* d_result;

    // Выделение памяти на GPU
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // Копирование данных с CPU на GPU
    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Обнуление результата на GPU
    cudaMemset(d_result, 0, sizeof(float));

    // Расчёт количества блоков
    int blocks = (n + THREADS - 1) / THREADS;

    // Начало замера времени
    auto start = std::chrono::high_resolution_clock::now();

    // Запуск kernel
    if (useShared)
        sumShared<<<blocks, THREADS>>>(d_data, d_result, n);
    else
        sumGlobal<<<blocks, THREADS>>>(d_data, d_result, n);

    // Ожидание завершения GPU
    cudaDeviceSynchronize();

    // Конец замера времени
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> time = end - start;

    // Освобождение памяти GPU
    cudaFree(d_data);
    cudaFree(d_result);

    return time.count();
}

// 5. Пузырьковая сортировка (локальная память)
// Функция выполняется на GPU
// Использует локальную память потока
__device__ void bubbleSort(float* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}

// 6. Kernel сортировки подмассивов
// Каждый поток сортирует небольшой подмассив
// Используется локальная память
__global__ void sortKernel(float* data, int subSize) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Локальный массив
    float local[16];

    int start = idx * subSize;

    // Копирование данных из глобальной памяти
    // в локальную память потока
    for (int i = 0; i < subSize; i++)
        local[i] = data[start + i];

    // Сортировка локального подмассива
    bubbleSort(local, subSize);

    // Запись результата обратно в глобальную память
    for (int i = 0; i < subSize; i++)
        data[start + i] = local[i];
}

int main() {
    srand(time(nullptr));

    // Размеры массивов по заданию
    int sizes[] = { 10000, 100000, 1000000 };

    for (int n : sizes) {

        // Замер времени для глобальной памяти
        float tGlobal = measureSum(n, false);

        // Замер времени для shared memory
        float tShared = measureSum(n, true);

        std::cout << "Array size: " << n << std::endl;
        std::cout << "Global memory time: " << tGlobal << " ms\n";
        std::cout << "Shared memory time: " << tShared << " ms\n";
        std::cout << "Speedup: " << tGlobal / tShared << "x\n\n";
    }

    return 0;
}
