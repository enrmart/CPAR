#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, i;
    int msg_size = 100;
    int* sendbuf = NULL;
    int* recvbuf = NULL;
    MPI_Request request; // Declaración de una variable de tipo MPI_Request

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sendbuf = (int*)malloc(sizeof(int) * msg_size); // Asignación dinámica de memoria para el buffer de envío
    recvbuf = (int*)malloc(sizeof(int) * msg_size); // Asignación dinámica de memoria para el buffer de recepción

    // Inicialización del buffer de envío con datos
    for (i = 0; i < msg_size; i++) {
        sendbuf[i] = rank * msg_size + i;
    }

    // Envío síncrono del mensaje utilizando MPI_Rsend
    MPI_Rsend(sendbuf, msg_size, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

    // Recepción no bloqueante del mensaje utilizando MPI_Irecv
    MPI_Irecv(recvbuf, msg_size, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &request);

    // Espera a que la recepción del mensaje haya sido completada utilizando MPI_Wait
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // Impresión de los datos recibidos por cada proceso MPI
    printf("Process %d received: ", rank);
    for (i = 0; i < msg_size; i++) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    // Liberación de la memoria asignada para los buffers de envío y recepción
    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();

    return 0;
}

/*MPI_Rsend de manera síncrona, lo que significa que el proceso no puede continuar hasta que el mensaje haya sido entregado al proceso destino. 
Sin embargo, a diferencia de la función MPI_Ssend, la función MPI_Rsend no bloquea la ejecución posterior del código, 
lo que permite al programa continuar con otras tareas mientras se espera a que se complete el envío del mensaje.*/



/*En resumen, estas expresiones son una forma conveniente de calcular los rangos de los procesos vecinos en un anillo lógico 
sin tener que escribir casos especiales para los procesos que están en los extremos del anillo.*/
