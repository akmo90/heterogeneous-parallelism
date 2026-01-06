#include <iostream>     // Ввод и вывод (cin, cout)
#include <random>       // Генерация случайных чисел
#include <chrono>       // Измерение времени выполнения
#include <omp.h>        // Библиотека OpenMP для параллельных вычислений

using namespace std;

double average_seq(int* arr, int N) {
    long long sum = 0;          // Переменная для хранения суммы элементов

    // Последовательный цикл по всем элементам массива
    for (int i = 0; i < N; i++)
        sum += arr[i];

    // Возвращаем среднее значение
    return double(sum) / N;
}

/*
 * Параллельное вычисление среднего значения массива с использованием OpenMP
 * Используется директива reduction для корректного суммирования
 * из разных потоков
 */
double average_par(int* arr, int N) {
    long long sum = 0;          // Общая сумма

    // Параллельный цикл OpenMP
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++)
        sum += arr[i];

    // Возвращаем среднее значение
    return double(sum) / N;
}

int main() {
    setlocale(0, ""); // Корректное отображение сообщений в консоли

    int N;
    cout << "Enter N (array size): ";
    cin >> N;

    // Проверка корректности размера массива
    if (N <= 0) {
        cout << "Invalid N\n";
        return 1;
    }

    // Динамическое выделение памяти под массив
    int* arr = new int[N];

    // Настройка генератора случайных чисел
    random_device rd;       
    mt19937 gen(rd());     
    uniform_int_distribution<> dist(1, 100); 

    // Заполнение массива случайными числами
    for (int i = 0; i < N; i++)
        arr[i] = dist(gen);

    // Последовательское вычисление
    auto t1 = chrono::high_resolution_clock::now();
    double avg_seq = average_seq(arr, N);
    auto t2 = chrono::high_resolution_clock::now();

    // Время выполнения последовательной версии
    double time_seq = chrono::duration<double, milli>(t2 - t1).count();

    // Параллельное вычисление 
    auto t3 = chrono::high_resolution_clock::now();
    double avg_par = average_par(arr, N);
    auto t4 = chrono::high_resolution_clock::now();

    // Время выполнения параллельной версии
    double time_par = chrono::duration<double, milli>(t4 - t3).count();

    // Вывод результатов
    cout << "Sequential average = " << avg_seq
         << ", time = " << time_seq << " ms\n";

    cout << "Parallel average   = " << avg_par
         << ", time = " << time_par << " ms\n";

    // Освобождение выделенной памяти
    delete[] arr;

    return 0;
}
