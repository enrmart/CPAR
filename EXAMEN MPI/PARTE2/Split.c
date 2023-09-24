#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Comm global_comm, even_comm, odd_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    global_comm = MPI_COMM_WORLD;

    // Dividir el comunicador global en subcomunicadores para procesos pares e impares
    int color = (rank % 2 == 0) ? 0 : 1;  // Los procesos pares tienen color 0, los impares color 1
    int key = rank / 2;  // La clave se calcula como el índice dividido entre 2
    MPI_Comm_split(global_comm, color, key, &even_comm);  // Comunicador para procesos pares
    MPI_Comm_split(global_comm, 1-color, key, &odd_comm);  // Comunicador para procesos impares

    // Imprimir información en cada subcomunicador
    if (color == 0) {
        int even_rank, even_size;
        MPI_Comm_rank(even_comm, &even_rank);
        MPI_Comm_size(even_comm, &even_size);
        printf("En el comunicador de procesos pares y su color es el %d, el proceso %d de %d\n",color,even_rank, even_size);
    } else {
        int odd_rank, odd_size;
        MPI_Comm_rank(odd_comm, &odd_rank);
        MPI_Comm_size(odd_comm, &odd_size);
        printf("En el comunicador de procesos impares y su color es el %d, el proceso %d de %d\n", color,odd_rank, odd_size);
    }

    MPI_Finalize();
    return 0;
}
/*En este código, cada proceso calcula su "color" y "clave" en función de su rango. 
Los procesos con índice par tienen un color de 0, mientras que los procesos con índice impar tienen un color de 1. 
La clave se calcula como el índice del proceso dividido entre 2.

Luego, se utilizan las funciones MPI_Comm_split para dividir el comunicador global en dos subcomunicadores: uno para los procesos con color 0 
y otro para los procesos con color 1. La clave se utiliza para determinar el orden en que se establecen las nuevas clasificaciones de rango 
de los procesos en el nuevo comunicador.

Finalmente, se utiliza cada subcomunicador para imprimir información sobre los procesos que pertenecen a él. 
En este ejemplo, los procesos de índice par imprimirán información en el subcomunicador de procesos pares 
y los procesos de índice impar imprimirán información en el subcomunicador de procesos impares.*/