#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, i;
    int sendcounts[4], displs[4];
    int* sendbuf = NULL;
    int* recvbuf = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sendbuf = (int*)malloc(sizeof(int) * size * size);
    recvbuf = (int*)malloc(sizeof(int) * size * size);

    for (i = 0; i < size * size; i++) {
        sendbuf[i] = i;
    }

    sendcounts[0] = 1;
    displs[0] = 0;
    for (i = 1; i < size; i++) {
        sendcounts[i] = i + 1;
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    MPI_Allgatherv(sendbuf + displs[rank], sendcounts[rank], MPI_INT, recvbuf, sendcounts, displs, MPI_INT, MPI_COMM_WORLD);

    printf("Process %d received: ", rank);
    for (i = 0; i < size * size; i++) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();

    return 0;
}
