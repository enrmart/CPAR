#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, i;
    int msg_size = 100;
    int* sendbuf = NULL;
    int* recvbuf = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sendbuf = (int*)malloc(sizeof(int) * msg_size);
    recvbuf = (int*)malloc(sizeof(int) * msg_size);

    for (i = 0; i < msg_size; i++) {
        sendbuf[i] = rank * msg_size + i;
    }

    MPI_Ssend(sendbuf, msg_size, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

    MPI_Recv(recvbuf, msg_size, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d received: ", rank);
    for (i = 0; i < msg_size; i++) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();

    return 0;
}
