#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    int rank = 0; 
    int data = 0; 
    int tag = 0;
    MPI_Status stat;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0){
        int array[5]={1,2,3,4,5};
        srand(0);
        int random_size = (int)(rand()%10) + 1;
        MPI_Send(array, random_size, MPI_INT, 1, tag, MPI_COMM_WORLD );
    }
    if(rank == 1){
        int size;
        MPI_Probe(0, tag, MPI_COMM_WORLD, &stat); //Se sondea al emisor para saber de cuanto va a ser el tamaño que va a enviar 
        MPI_Get_count(&stat, MPI_INT, &size); //Devuelve el numero de elementos que se van a obtener en el receive
        int data[ size ]; //Se reserva la memoria en funcion del numero de elementos anterior
        MPI_Recv(data, size, MPI_INT, 0, tag, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("Received %2d elems from %d\n", size, 0 );
        for(int i=0;i<size;i++){
            printf("El numero de la posicion %d es %d\n",i,data[i]);
        }
    }

    MPI_Finalize();
    return 0;
}