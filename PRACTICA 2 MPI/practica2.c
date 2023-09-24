//============================================================================
// Name:			KMEANS.c
// Compilacion:	gcc KMEANS.c -o KMEANS -lm
//============================================================================

//Los integrantes de este grupo son:
//Carlos Martin Sanz
//Enrique Martin Calvo


#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <mpi.h>

//Constantes
#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
Muestra el correspondiente errro en la lectura de fichero de datos
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tEl fichero %s contiene demasiadas columnas.\n", filename);
			fprintf(stderr,"\tSe supero el tamano maximo de columna MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error leyendo el fichero %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error escibiendo en el fichero %s.\n", filename);
			break;
	}
	fflush(stderr);	
}

/* 
Lectura del fichero para determinar el numero de filas y muestras (samples)
*/
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}	    
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;  
        return 0;
    }
    else
	{
    	return -2;
	}
}

/* 
Carga los datos del fichero en la estructra data
*/
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {         
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/* 
Escribe en el fichero de salida la clase a la que perteneces cada muestra (sample)
*/
int writeResult(int *classMap, int lines, const char* filename)
{	
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}

/*
Copia el valor de los centroides de data a centroids usando centroidPos como
mapa de la posicion que ocupa cada centroide en data
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
Calculo de la distancia euclidea
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Funcion de clasificacion, asigna una clase a cada elemento de data
*/
int classifyPoints(float *myData, float *centroids, int *classMap, int mySize, int samples, int K){
	int i,j;
	int class;
	float dist, minDist;
	int changes=0;
	//Al realizar todos los procesos la clasificacion de sus puntos, solo deben de recorrer su tamaño
	for(i=0; i<mySize; i++)
	{
		class=1;
		minDist=FLT_MAX;
		for(j=0; j<K; j++)
		{
			dist=euclideanDistance(&myData[i*samples], &centroids[j*samples], samples);

			if(dist < minDist)
			{
				minDist=dist;
				class=j+1;
			}
		}
		
		if(classMap[i]!=class)
		{
			changes++;
		}
		classMap[i]=class;
	}
	return(changes);
}

/*
Recalcula los centroides a partir de una nueva clasificacion
*/
float recalculateCentroids(float *myData, float *centroids, int *classMap, int mySize, int samples, int K,int rank){
	int class, i, j;
	float maxDist=0;
		
	int *pointsPerClass=(int*)calloc(K,sizeof(int));
	int *pointsPerClass_aux=NULL;

	float *auxCentroids=(float*)calloc(K*samples, sizeof(float));
	float *globalCentroids=NULL;

	//Variables que representaran los datos globales para calcular los centroides totales
	if(rank == 0) {
		pointsPerClass_aux=(int*)calloc(K,sizeof(int));
		globalCentroids=(float*)calloc(K*samples, sizeof(float));
	}
		
	if (pointsPerClass == NULL || auxCentroids == NULL)
	{
		fprintf(stderr,"Error alojando memoria\n");
		MPI_Abort(MPI_COMM_WORLD,-4);
	}

	//pointPerClass: numero de puntos clasificados en cada clase
	//auxCentroids: media de los puntos de cada clase 
	
	//Cada porceso calcula el numero de puntos de cada clase que tiene y sus posibles centroides
	for(i=0; i<mySize; i++) 
	{
		class=classMap[i];
		pointsPerClass[class-1] = pointsPerClass[class-1] +1;
		for(j=0; j<samples; j++){
			auxCentroids[(class-1)*samples+j] += myData[i*samples+j];
		}
	}

	//El 0 recibe los datos totales para calcular los nuevos centroides 	
	MPI_Reduce(auxCentroids, globalCentroids,K*samples, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(pointsPerClass,pointsPerClass_aux, K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank==0){
		maxDist=FLT_MIN;
		//Calcula los centroides
		for(i=0; i<K; i++) 
		{
			for(j=0; j<samples; j++){
				globalCentroids[i*samples+j] /= pointsPerClass_aux[i];
			}
		}

		float *distCentroids=(float*)malloc(K*sizeof(float));
		if(distCentroids == NULL){
				fprintf(stderr,"Error alojando memoria\n");
				MPI_Abort(MPI_COMM_WORLD,-4);
		}
		//Calcula la distancia entre los centroides nuevos y antiguos
		for(i=0; i<K; i++){
			distCentroids[i]=euclideanDistance(&centroids[i*samples], &globalCentroids[i*samples], samples);
			if(distCentroids[i]>maxDist) {
				maxDist=distCentroids[i];
			}
		}

		memcpy(centroids, globalCentroids, (K*samples*sizeof(float)));
		free(globalCentroids);
		free(pointsPerClass_aux);
		free(distCentroids);
	}
	//Todos reciben los nuevos centroides
	MPI_Bcast(centroids, K*samples, MPI_FLOAT, 0, MPI_COMM_WORLD);
	free(pointsPerClass);
	free(auxCentroids);
	return(maxDist);
}




int main(int argc, char* argv[])
{

	int rank = 0; int tag = 0; int nprocs=0;
	MPI_Status stat;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

	//START CLOCK***************************************
	double t_begin, t_end,tiempoTotal=0,tiempoTotalp;
	t_begin = MPI_Wtime();
	//**************************************************
	
	//VARIABLES GLOBALES
	int lines = 0, samples= 0,error;
	float *data=NULL;
	int *displs=NULL;
	int *sendcounts=NULL;
 	
		/*
		* PARAMETROS
		*
		* argv[1]: Fichero de datos de entrada
		* argv[2]: Numero de clusters
		* argv[3]: Numero maximo de iteraciones del metodo. Condicion de fin del algoritmo
		* argv[4]: Porcentaje minimo de cambios de clase. Condicion de fin del algoritmo.
		* 			Si entre una iteracion y la siguiente el porcentaje cambios de clase es menor que
		* 			este procentaje, el algoritmo para.
		* argv[5]: Precision en la distancia de centroides depuesde la actualizacion
		* 			Es una condicion de fin de algoritmo. Si entre una iteracion del algoritmo y la 
		* 			siguiente la distancia maxima entre centroides es menor que esta precsion el
		* 			algoritmo para.
		* argv[6]: Fichero de salida. Clase asignada a cada linea del fichero.
		* */
		if(argc !=  7)
		{
			fprintf(stderr,"EXECUTION ERROR KMEANS Iterative: Parameters are not correct.\n");
			fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
			fflush(stderr);
			MPI_Abort(MPI_COMM_WORLD,-1);
		}

	

	//Lectura de los datos de entrada
	// lines = numero de puntos;  samples = numero de dimensiones por punto
	
	//La lectura de datos realizada en secuencial 
	if(rank==0){
		error = readInput(argv[1], &lines, &samples);
		if(error != 0){
			showFileError(error,argv[1]);
			MPI_Abort(MPI_COMM_WORLD,error);
		}

		data = (float*)calloc(lines*samples,sizeof(float));
		if (data == NULL){
			fprintf(stderr,"Error alojando memoria\n");
			MPI_Abort(MPI_COMM_WORLD,-4);
		}

		error = readInput2(argv[1], data);
		if(error != 0){
			showFileError(error,argv[1]);
			MPI_Abort(MPI_COMM_WORLD,error);
		}
		//Calculos para la distribuccion de los datos 
		sendcounts = (int *)malloc(nprocs*sizeof(int));
        displs = (int *)malloc(nprocs*sizeof(int));
		int pos=0;
        for(int ranks=0;ranks<nprocs;ranks++){
            displs[ranks]=pos*samples;
            int size=lines/nprocs;
            int restos=lines%nprocs;
            size=(ranks<restos)?size+1:size;
            sendcounts[ranks]=size*samples;
            pos=pos+size;
        }
	}

	//Notificamos los parametros leidos del archivo a los demas 
	MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&samples,1,MPI_INT,0,MPI_COMM_WORLD);
	
	//Calcular el tamaño de cada proceso para reservar su memoria necesaria   
	int mySize=lines/nprocs;
    int resto=lines%nprocs;
    mySize=(rank<resto)?mySize+1:mySize;
	float * myData=(float *) malloc(mySize*samples*sizeof(float));//Vector propio que recibira todo
	

	//Enviar y repatir los datos a los distintos procesos 
	MPI_Scatterv(data, sendcounts, displs, MPI_FLOAT,myData,mySize*samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	

	// parametros del algoritmo. La entrada no esta valdidada
	//Deben ser valores iguales para todos
	int K=atoi(argv[2]); 
	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(lines*atof(argv[4])/100.0);
	float maxThreshold=atof(argv[5]);


	//poscion de los centroides en data
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(mySize,sizeof(int)); 
	
	if (centroids == NULL || classMap == NULL){
		fprintf(stderr,"Error alojando memoria\n");
		MPI_Abort(MPI_COMM_WORLD,-4);
	}

	//Inicializacion de los centroides realizada por el proceso maestro 
	if(rank==0){

		int *centroidPos = (int*)calloc(K,sizeof(int));
		if(centroidPos == NULL){
			fprintf(stderr,"Error alojando memoria\n");
			MPI_Abort(MPI_COMM_WORLD,-4);
		}

		srand(0);
		int i;
		for(i=0; i<K; i++) 
			centroidPos[i]=rand()%lines;
		//Carga del array centroids con los datos del array data
		//los centroides son puntos almacenados en data
		initCentroids(data, centroids, centroidPos, samples, K);
		free(centroidPos);
	}	

	//Notificacion de los centroides a los demas procesos 
	MPI_Bcast(centroids,K*samples,MPI_INT,0,MPI_COMM_WORLD);

	if(rank==0){//Esta es la forma para que se imprima una unica vez
	// Resumen de datos cargados
		printf("\n\tFichero de datos: %s \n\tPuntos: %d\n\tDimensiones: %d\n", argv[1], lines, samples);
		printf("\tNumero de clusters: %d\n", K);
		printf("\tNumero maximo de iteraciones: %d\n", maxIterations);
		printf("\tNumero minimo de cambios: %d [%g%% de %d puntos]\n", minChanges, atof(argv[4]), lines);
		printf("\tPrecision maxima de los centroides: %f\n", maxThreshold);
	}
		
	//Otras variables para bucle
	float distCentp, distCent;
	int changes,changesp;
	int flag_stop=1;
	int it=0;
	
	//END CLOCK*****************************************
	t_end = MPI_Wtime();
	if(rank==0){
		printf("\nAlojado de memoria del proceso %d : %f segundos\n",rank,t_end-t_begin);
		fflush(stdout);
	}
	//**************************************************
	//START CLOCK***************************************
	t_begin = MPI_Wtime();

	//***************************************************/
	do{
		//Calcula la distancia desde cada punto al centroide
		//Asigna cada punto al centroide mas cercano
		changesp=classifyPoints(myData, centroids, classMap, mySize, samples, K);
		
		//Los cambios totales realizados entre todos los procesos 
		MPI_Reduce(&changesp,&changes,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
		
		//Recalcula los centroides: calcula la media dentro de cada centoide
		distCentp=recalculateCentroids(myData, centroids, classMap, mySize, samples, K,rank);
		
		//La maxima distancia que existe entre los centroides 
		MPI_Reduce(&distCentp,&distCent,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);

		//Control del final del bucle 	
		if(rank==0){
			it++;
			printf("\n[%d] Cambios de cluster: %d\tMax. dist. centroides: %f", it, changes, distCent);
			if((changes<=minChanges) || (it>=maxIterations) || (distCent<=maxThreshold)) flag_stop=0;
		}

		//Notificacion a los demas de la finalizacion del bucle 
		MPI_Bcast(&flag_stop,1,MPI_INT,0,MPI_COMM_WORLD);
	} while(flag_stop);

	int *allClassMap = NULL;

	if (rank == 0) {
    	allClassMap = (int*)calloc(lines, sizeof(int));
		int pos=0;
        for(int ranks=0;ranks<nprocs;ranks++){
            displs[ranks]=pos;
            int size=lines/nprocs;
            int restos=lines%nprocs;
            size=(ranks<restos)?size+1:size;
            sendcounts[ranks]=size;
            pos=pos+size;
        }
	}
	//Todos los datos a imprimir para la escritura secuencial 
	MPI_Gatherv(classMap, mySize, MPI_INT, allClassMap, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank==0){
		
		//Condiciones de fin de la ejecucion
		if (changes<=minChanges) {
			printf("\n\nCondicion de parada: Numero minimo de cambios alcanzado: %d [%d]",changes, minChanges);
		}
		else if (it>=maxIterations) { 
			printf("\n\nCondicion de parada: Numero maximo de iteraciones alcanzado: %d [%d]",it, maxIterations);
		}
		else{
			printf("\n\nCondicion de parada: Precision en la actualizacion de centroides alcanzada: %g [%g]",distCent, maxThreshold);
		}	

		//Escritura en fichero de la clasificacion de cada punto de manera secuencial 
		error = writeResult(allClassMap, lines, argv[6]);
		if(error != 0){
			showFileError(error, argv[6]);	
			MPI_Abort(MPI_COMM_WORLD,error);
		}
		free(allClassMap);
	}

	//END CLOCK*****************************************
	t_end = MPI_Wtime();
	if(rank==0){
		printf("\nComputacion: %f segundos", t_end-t_begin);
		fflush(stdout);
	}
	//**************************************************
	//START CLOCK***************************************
	t_begin= MPI_Wtime();
	//**************************************************


	//Liberacion de la memoria dinamica
	free(classMap);
	free(centroids);

	//END CLOCK*****************************************
	t_end = MPI_Wtime();
	if(rank==0){
		printf("\n\nLiberacion: %f segundos\n",t_end-t_begin);
		fflush(stdout);
	}
	//***************************************************/
	MPI_Finalize();
	return 0;
}
