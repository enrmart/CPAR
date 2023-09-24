#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Definición de la operación personalizada
void average(void *in, void *inout, int *len, MPI_Datatype *datatype) {
    double *x = (double *) in;
    double *y = (double *) inout;
    int n = *len;
    for (int i = 0; i < n; i++) {
        y[i] = (x[i] + y[i]) / 2;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double sendbuf[2] = {rank + 1.0, rank + 2.0};
    double recvbuf[2];

    MPI_Op average_op;
    MPI_Op_create(average, 1, &average_op);

    MPI_Allreduce(sendbuf, recvbuf, 2, MPI_DOUBLE, average_op, MPI_COMM_WORLD);

    printf("Proceso %d: Media aritmética de los números %.1f y %.1f = %.1f y %.1f, respectivamente.\n",
           rank, sendbuf[0], sendbuf[1], recvbuf[0], recvbuf[1]);

    MPI_Op_free(&average_op);
    MPI_Finalize();
    return 0;
}
