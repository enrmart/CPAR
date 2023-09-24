#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size, rank;
    int data[10];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Inicializar el arreglo de datos en el proceso raíz
        for (int i = 0; i < 10; i++) {
            data[i] = i;
        }
    }

    // Transmitir el arreglo de datos desde el proceso raíz a todos los demás procesos
    MPI_Bcast(&data, 10, MPI_INT, 0, MPI_COMM_WORLD);

    // Imprimir el arreglo de datos en cada proceso
    printf("Proceso %d: data = { ", rank);
    for (int i = 0; i < 10; i++) {
        printf("%d ", data[i]);
    }
    printf("}\n");

    MPI_Finalize();
    return 0;
}
