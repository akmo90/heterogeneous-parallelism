#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
// Количество потоков в блоке
#define THREADS 128
// Максимальная ёмкость стека
#define CAPACITY 256

// Структура данных: ПАРАЛЛЕЛЬНАЯ ОЧЕРЕДЬ
// Реализуем FIFO
// Атомарные операции обеспечивают корректность
struct Queue {
    int* data;        // Массив данных в глобальной памяти GPU
    int head;         // Указатель на начало очереди
    int tail;         // Указатель на конец очереди
    int capacity;     // Ёмкость очереди

    // Инициализация очереди
    __device__ void init(int* buffer, int size) {
        data = buffer;
        head = 0;
        tail = 0;
        capacity = size;
    }

    // Добавление элемента в очередь
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1); 
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    // Извлечение элемента из очереди
    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(&head, 1); 
        if (pos < tail) {
            *value = data[pos];
            return true;
        }
        return false; 
    }
};

// Ядро для тестирования очереди
__global__ void queueKernel(Queue* queue, int* buffer, int* output) {

    // Инициализация очереди одним потоком
    if (threadIdx.x == 0)
        queue->init(buffer, CAPACITY);

    __syncthreads();

    int tid = threadIdx.x;

    // Каждый поток добавляет элемент
    queue->enqueue(tid);

    __syncthreads();

    // Каждый поток пытается извлечь элемент
    int value;
    if (queue->dequeue(&value))
        output[tid] = value;
}

int main() {
    Queue* d_queue;
    int* d_buffer;
    int* d_output;

    cudaMalloc(&d_queue, sizeof(Queue));
    cudaMalloc(&d_buffer, CAPACITY * sizeof(int));
    cudaMalloc(&d_output, THREADS * sizeof(int));

    queueKernel<<<1, THREADS>>>(d_queue, d_buffer, d_output);
    cudaDeviceSynchronize();

    int h_output[THREADS];
    cudaMemcpy(h_output, d_output, THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Queue output: ";
    for (int i = 0; i < 10; i++)
        std::cout << h_output[i] << " ";
    std::cout << "\n";

    cudaFree(d_queue);
    cudaFree(d_buffer);
    cudaFree(d_output);

    return 0;
}
