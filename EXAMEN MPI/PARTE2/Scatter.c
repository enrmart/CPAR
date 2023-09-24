#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size, rank;
    int all_data[40];
    int local_data[10];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Inicializar el arreglo de datos en el proceso raíz
    if (rank == 0) {
        for (int i = 0; i < 40; i++) {
            all_data[i] = i;
        }
    }

    // Distribuir partes iguales del arreglo de datos a todos los procesos
    MPI_Scatter(&all_data, 10, MPI_INT, &local_data, 10, MPI_INT, 0, MPI_COMM_WORLD);

    // Imprimir el arreglo de datos local en cada proceso
    printf("Datos en el proceso %d: { ", rank);
    for (int i = 0; i < 10; i++) {
        printf("%d ", local_data[i]);
    }
    printf("}\n");

    MPI_Finalize();
    return 0;
}
