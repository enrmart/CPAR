#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size, rank;
    int send_data[4] = {1, 2, 3, 4};
    int recv_data[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Intercambiar arreglos de datos con todos los demás procesos
    MPI_Alltoall(&send_data, 1, MPI_INT, &recv_data, 1, MPI_INT, MPI_COMM_WORLD);

    // Imprimir los arreglos de datos recibidos en cada proceso
    printf("Datos recibidos en el proceso %d: { ", rank);
    for (int i = 0; i < 4; i++) {
        printf("%d ", recv_data[i]);
    }
    printf("}\n");

    MPI_Finalize();
    return 0;
}
