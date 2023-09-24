#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int count = 4; // Número de elementos del nuevo tipo de datos
    int blocklength = 1; // Longitud de cada elemento del nuevo tipo de datos
    int stride = 5; // Espaciado entre los elementos del nuevo tipo de datos

    // Se crea el nuevo tipo de datos
    MPI_Datatype my_type;
    MPI_Type_contiguous(count, MPI_INT, &my_type);

    // Se ajusta el espaciado entre los elementos del nuevo tipo de datos
    MPI_Type_create_resized(my_type, 0, blocklength * sizeof(int), &my_type);
    MPI_Type_commit(&my_type);

    // Se crea un buffer con los datos a enviar
    int *sendbuf = NULL;
    if (rank == 0) {
        sendbuf = malloc(size * stride * sizeof(int));
        for (int i = 0; i < size * stride; i++) {
            sendbuf[i] = i;
        }
    }

    // Se crea un buffer para recibir los datos
    int *recvbuf = malloc(count * sizeof(int));

    // Se envían los datos utilizando el nuevo tipo de datos
    MPI_Scatter(sendbuf, 1, my_type, recvbuf, count, MPI_INT, 0, MPI_COMM_WORLD);

    // Se imprimen los datos recibidos por cada proceso
    for (int i = 0; i < count; i++) {
        printf("Proceso %d: recvbuf[%d] = %d\n", rank, i, recvbuf[i]);
    }

    MPI_Type_free(&my_type);
    MPI_Finalize();
    return 0;
}
