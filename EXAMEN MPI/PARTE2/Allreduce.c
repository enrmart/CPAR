#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size, rank;
    int array[4] = {1, 2, 3, 4};
    int sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calcular la suma local de los elementos del arreglo
    int local_sum = 0;
    for (int i = 0; i < 4; i++) {
        local_sum += array[i];
    }

    // Sumar las sumas locales y almacenar el resultado en todos los procesos
    MPI_Allreduce(&local_sum, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Imprimir la suma global en todos los procesos
    printf("En el proceso %d, la suma global es %d\n", rank, sum);

    MPI_Finalize();
    return 0;
}
