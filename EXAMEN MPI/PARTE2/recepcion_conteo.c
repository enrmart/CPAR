#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    int rank = 0; 
    int data = 0; 
    int tag = 0;
    
    MPI_Status stat; int count;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0){
        int data[5] = {1,2,3,4,5};
        MPI_Send( data, 5, MPI_INT, 1, tag, MPI_COMM_WORLD );
    } else if (rank==1) {
        int data[10]; 
        MPI_Recv( data, 10, MPI_INT, 0, tag, MPI_COMM_WORLD, &stat );
        MPI_Get_count( &stat, MPI_INT, &count );
        printf("El conteo de %d es %d\n",rank,count);//Devuelve el numero de elementos recibidos en la operacion anterior
}
    
    
    MPI_Finalize();
    return 0;
}