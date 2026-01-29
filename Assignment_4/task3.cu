#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000   // размер массива

// CUDA-ядро: GPU обрабатывает первую половину массива
__global__ void gpuProcess(float* a) {

    // Глобальный индекс потока
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток обрабатывает один элемент
    if (i < N / 2)
        a[i] = a[i] * a[i];   // возводим элемент в квадрат
}

int main() {

    // ================================
    // Создание массива на CPU
    // ================================
    float* h = new float[N];
    for (int i = 0; i < N; i++)
        h[i] = 2.0f;   // все элементы равны 2

    // ================================
    // Выделение памяти на GPU
    // ================================
    float* d;
    cudaMalloc(&d, (N / 2) * sizeof(float));

    // Копируем первую половину массива с CPU на GPU
    cudaMemcpy(d, h, (N / 2) * sizeof(float), cudaMemcpyHostToDevice);

    // ================================
    // CPU часть (вторая половина массива)
    // ================================
    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (int i = N / 2; i < N; i++)
        h[i] = h[i] * h[i];   // CPU обрабатывает вторую половину

    auto end_cpu = std::chrono::high_resolution_clock::now();

    // ================================
    // GPU часть (первая половина массива)
    // ================================
    auto start_gpu = std::chrono::high_resolution_clock::now();

    // Запускаем CUDA-ядро для первой половины массива
    gpuProcess<<<(N / 2 + 255) / 256, 256>>>(d);

    // Ждём, пока GPU закончит вычисления
    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();

    // ================================
    // Вывод времени выполнения
    // ================================
    std::cout << "CPU time: "
              << std::chrono::duration<double>(end_cpu - start_cpu).count()
              << " seconds" << std::endl;

    std::cout << "GPU time: "
              << std::chrono::duration<double>(end_gpu - start_gpu).count()
              << " seconds" << std::endl;
}
