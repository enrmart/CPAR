#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size, rank;
    int local_data[10];
    int all_data[40];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Inicializar el arreglo de datos local en cada proceso
    for (int i = 0; i < 10; i++) {
        local_data[i] = rank * 10 + i;
    }

    // Recopilar todos los arreglos de datos en el proceso raíz
    MPI_Gather(&local_data, 10, MPI_INT, &all_data, 10, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Imprimir el arreglo de datos completo en el proceso raíz
        printf("Datos recopilados: { ");
        for (int i = 0; i < 40; i++) {
            printf("%d ", all_data[i]);
        }
        printf("}\n");
    }

    MPI_Finalize();
    return 0;
}
