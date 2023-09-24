#include <stdio.h>
#include <mpi.h>

// Función de usuario para la operación MPI_Reduce
void my_reduce(int *invec, int *inoutvec, int *len, MPI_Datatype *datatype) {
    int i;
    for (i = 0; i < *len; i++) {
        inoutvec[i] += invec[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size, data[4] = {1, 2, 3, 4};
    int sum[4] = {0, 0, 0, 0};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Realizar la operación reduce
    MPI_Op my_op;
    MPI_Op_create((MPI_User_function *)my_reduce, 1, &my_op);

    MPI_Reduce(data, sum, 4, MPI_INT, my_op, 0, MPI_COMM_WORLD);

    MPI_Op_free(&my_op);

    if (rank == 0) {
        printf("Resultado: %d %d %d %d\n", sum[0], sum[1], sum[2], sum[3]);
    }

    MPI_Finalize();
    return 0;
}
/*Para utilizar la función my_reduce en la operación MPI_Reduce, primero se debe crear un operador de usuario utilizando la función MPI_Op_create.
 En este caso, el operador de usuario se llama my_op. Luego se utiliza MPI_Reduce con my_op como el operador de reducción. 
Finalmente, se libera el operador de usuario utilizando MPI_Op_free. En el proceso con rango 0, se imprime el resultado de la operación reduce.*/