#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int my_rank, num_procs, source, dest, tag = 0;
    char message[100];
    MPI_Status status;
    int resultlen;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    if (my_rank != 0) {
        sprintf(message, "Hola, soy el proceso %d", my_rank);
        dest = 0;
        int ierr=MPI_Send(message, strlen(message)+1, MPI_CHAR, -5, tag, MPI_COMM_WORLD);
        if(ierr!=MPI_SUCCESS){
            char error[MPI_MAX_ERROR_STRING];
            MPI_Error_string(ierr,error,&resultlen);
            printf("Error en el send %s\n",error);
            MPI_Abort(MPI_COMM_WORLD, -1);

        }
    } else {
        for (source = 1; source < num_procs; source++) {
            int ierr=MPI_Recv(message, 100, MPI_CHAR, -4, tag, MPI_COMM_WORLD, &status);
            if(ierr!=MPI_SUCCESS){
                char error[MPI_MAX_ERROR_STRING];
                MPI_Error_string(ierr,error,&resultlen);
                printf("Error en el receive %s\n",error);
                MPI_Abort(MPI_COMM_WORLD, -1);

            }
            printf("%s\n", message);
        }
    }

    MPI_Finalize();
    return 0;
}


