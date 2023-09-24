#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, i;
    int msg_size = 100;
    int buf_size = 200;
    int* sendbuf = NULL;
    int* recvbuf = NULL;
    int* buffer = NULL;
    MPI_Request request;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sendbuf = (int*)malloc(sizeof(int) * msg_size);
    recvbuf = (int*)malloc(sizeof(int) * msg_size);
    buffer = (int*)malloc(sizeof(int) * buf_size);

    for (i = 0; i < msg_size; i++) {
        sendbuf[i] = rank * msg_size + i;
    }

    MPI_Buffer_attach(buffer, buf_size);

    MPI_Bsend(sendbuf, msg_size, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

    MPI_Buffer_detach(&buffer, &buf_size);

    MPI_Irecv(recvbuf, msg_size, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &request);

    MPI_Wait(&request, MPI_STATUS_IGNORE);

    printf("Process %d received: ", rank);
    for (i = 0; i < msg_size; i++) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    free(sendbuf);
    free(recvbuf);
    free(buffer);

    MPI_Finalize();

    return 0;
}
