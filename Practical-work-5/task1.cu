#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Количество потоков в блоке
#define THREADS 128
// Максимальная ёмкость стека
#define CAPACITY 256

// Структура данных: ПАРАЛЛЕЛЬНЫЙ СТЕК
// Реализуем LIFO
// Для синхронизации используется атомарные операции
struct Stack {
    int* data;        // Указатель на массив данных в глобальной памяти GPU
    int top;          // Индекс вершины стека
    int capacity;     // Максимальная ёмкость стека

    // Инициализация стека (выполняется на GPU)
    __device__ void init(int* buffer, int size) {
        data = buffer;
        top = -1;            // Стек пуст
        capacity = size;
    }

    // Добавление элемента в стек
    // atomicAdd гарантирует корректную работу при доступе
    // из нескольких потоков
    __device__ bool push(int value) {
        int pos = atomicAdd(&top, 1); 
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    // Извлечение элемента из стека
    __device__ bool pop(int* value) {
        int pos = atomicSub(&top, 1); 
        if (pos >= 0) {
            *value = data[pos];
            return true;
        }
        return false; 
    }
};

// Ядро для тестирования стека
__global__ void stackKernel(Stack* stack, int* buffer, int* output) {

    // Инициализация стека выполняется одним потоком
    if (threadIdx.x == 0)
        stack->init(buffer, CAPACITY);

    __syncthreads(); // Синхронизация потоков блока

    int tid = threadIdx.x;

    // Каждый поток кладёт своё значение в стек
    stack->push(tid);

    __syncthreads();

    // Каждый поток пытается извлечь элемент
    int value;
    if (stack->pop(&value))
        output[tid] = value;
}

int main() {
    Stack* d_stack;
    int* d_buffer;
    int* d_output;

    // Выделение памяти на GPU
    cudaMalloc(&d_stack, sizeof(Stack));
    cudaMalloc(&d_buffer, CAPACITY * sizeof(int));
    cudaMalloc(&d_output, THREADS * sizeof(int));

    // Запуск CUDA-ядра
    stackKernel<<<1, THREADS>>>(d_stack, d_buffer, d_output);
    cudaDeviceSynchronize();

    // Копирование результата на CPU
    int h_output[THREADS];
    cudaMemcpy(h_output, d_output, THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    // Вывод части результата для проверки
    std::cout << "Stack output: ";
    for (int i = 0; i < 10; i++)
        std::cout << h_output[i] << " ";
    std::cout << "\n";

    // Освобождение памяти GPU
    cudaFree(d_stack);
    cudaFree(d_buffer);
    cudaFree(d_output);

    return 0;
}
