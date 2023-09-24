#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, flag;
    int msg_size = 100;
    int* sendbuf = NULL;
    int* recvbuf = NULL;
    MPI_Request request; // Declaración de una variable de tipo MPI_Request
    MPI_Status status; // Declaración de una variable de tipo MPI_Status

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sendbuf = (int*)malloc(sizeof(int) * msg_size); // Asignación dinámica de memoria para el buffer de envío
    recvbuf = (int*)malloc(sizeof(int) * msg_size); // Asignación dinámica de memoria para el buffer de recepción

    // Inicialización del buffer de envío con datos
    for (int i = 0; i < msg_size; i++) {
        sendbuf[i] = rank * msg_size + i;
    }

    // Envío asíncrono del mensaje utilizando MPI_Isend
    MPI_Isend(sendbuf, msg_size, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD, &request);

    // Verificación asincrónica del estado de la comunicación utilizando MPI_Test
    do {
        MPI_Test(&request, &flag, &status);
    } while (!flag);

    // Recepción asíncrona del mensaje utilizando MPI_Irecv
    MPI_Irecv(recvbuf, msg_size, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &request);

    // Espera síncrona a que la recepción del mensaje haya sido completada utilizando MPI_Wait
    MPI_Wait(&request, &status);

    // Impresión de los datos recibidos por cada proceso MPI
    printf("Process %d received: ", rank);
    for (int i = 0; i < msg_size; i++) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    // Liberación de la memoria asignada para los buffers de envío y recepción
    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();

    return 0;
}
