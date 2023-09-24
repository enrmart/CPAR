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

    // Sumar las sumas locales en el proceso 0
    MPI_Reduce(&local_sum, &sum, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);

    // Imprimir la suma global en el proceso 0
    if (rank == 0) {
        printf("La suma global es %d\n", sum);
    }

    MPI_Finalize();
    return 0;
}
