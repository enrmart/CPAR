#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int my_rank, num_procs, source, dest, tag = 0;
    char message[100];
    MPI_Status status;
    int error_code;
    int ierr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    if (my_rank != 0) {
        sprintf(message, "Hola, soy el proceso %d", my_rank);
        dest = 0;
        ierr=MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
        if ( ierr!= MPI_SUCCESS) {
            MPI_Error_class(ierr, &error_code);
            fprintf(stderr, "Error al enviar el mensaje desde el proceso %d: %d\n", my_rank, error_code);//Imprimimos el codigo de error correspondiente a la clase
        }
    } else {
        for (source = 1; source < num_procs; source++) {
            ierr=MPI_Recv(message, 100, MPI_CHAR, -4, tag, MPI_COMM_WORLD, &status);
            if (ierr!= MPI_SUCCESS) {
                MPI_Error_class(ierr, &error_code);
                fprintf(stderr, "Error al recibir el mensaje desde el proceso %d: %d\n", source, error_code);//Imprimimos el codigo de error correspondiente a cada clase
            }else{
                printf("%s\n", message);
            }
        }
    }

    MPI_Finalize();
    return 0;
}