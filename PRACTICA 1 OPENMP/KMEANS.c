//============================================================================
// Name:			KMEANS.c
// Compilacion:	gcc KMEANS.c -o KMEANS -lm
//============================================================================

// PRACTICA OPEN MP 2022-2023
// Practica realizada por:
// Carlos Martín Sanz y Enrique Martín Calvo

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <omp.h>

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
    	return -3; //No file found
}

/*
Copia el valor de los centroides de data a centroids usando centroidPos como
mapa de la posicion que ocupa cada centroide en data
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	//No se paraleliza esto porque puede empeorar el rendimiento 
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
	int i;

	#pragma omp parallel for reduction(+:dist)
	for(i=0; i<samples; i++) 
	{
		// Modificamos la operacion para que unicamente tenga que acceder una vez a esas posiciones dentro de la memoria
		float diff = point[i] - center[i];
		dist += diff*diff;
	}


	dist = sqrt(dist);
	return(dist);
}

/*
Funcion de clasificacion, asigna una clase a cada elemento de data
*/
int classifyPoints(float *data, float *centroids, int *classMap, int lines, int samples, int K){
	int i,j;
	int class;
	float dist, minDist;
	int changes=0;

	// minDist y dist privado para que cada hilo tenga su copia y se calcule bien el menor
	// reduccion en la suma de changes
	// classMap shared por defecto
	#pragma omp parallel for private(j,class, minDist, dist) reduction(+:changes) schedule (dynamic)
	for(i=0; i<lines; i++)
	{
		class=1;
		minDist=FLT_MAX;

		for(j=0; j<K; j++)
		{
		
			dist=euclideanDistance(&data[i*samples], &centroids[j*samples], samples);
			

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
float recalculateCentroids(float *data, float *centroids, int *classMap, int lines, int samples, int K){
    int class, i, j;
    int *pointsPerClass;
	float *auxCentroids;
	float *distCentroids;
	float maxDist=FLT_MIN;

	
    pointsPerClass=(int*)calloc(K,sizeof(int));
    auxCentroids=(float*)calloc(K*samples, sizeof(float));
	distCentroids=(float*)malloc(K*sizeof(float));
	
    if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
    {
        fprintf(stderr,"Error alojando memoria\n");
        exit(-4);
    }

    //pointPerClass: numero de puntos clasificados en cada clase
    //auxCentroids: media de los puntos de cada clase 

	// Creamos una region paralela
	#pragma omp parallel
	{


	//shared  por defecto --> pointsPerClass, auxCentroids, classMap, data, centroids)
	//#pragma omp parallel for private(class, j)
	#pragma omp for private(class, j)
    for(i=0; i<lines; i++) 
    {
        class=classMap[i];

		// atomic lectura y escritura (update)
		#pragma omp atomic update
        pointsPerClass[class-1] = pointsPerClass[class-1] +1;

        for(j=0; j<samples; j++){
			#pragma omp atomic update
            auxCentroids[(class-1)*samples+j] += data[i*samples+j];
        }
    }

	//shared por defecto auxCentroids, pointsPerClass, K, samples
	#pragma omp for private(j)
    for(i=0; i<K; i++) 
    {
        for(j=0; j<samples; j++){
			#pragma omp atomic
            auxCentroids[i*samples+j] /= pointsPerClass[i];
        } 
    }

	// Hacemos la reduccion de maxDist, entonces ya es privada
	// centroids, auxCentroids y samples son shared por defecto
    #pragma omp for reduction(max: maxDist) 
	for(i=0; i<K; i++){
		#pragma omp atomic write
        distCentroids[i]=euclideanDistance(&centroids[i*samples], &auxCentroids[i*samples], samples);
		if(distCentroids[i]>maxDist) {
		    maxDist=distCentroids[i];
        }
    }
	
	}
	
    memcpy(centroids, auxCentroids, (K*samples*sizeof(float)));
    free(distCentroids);
    free(pointsPerClass);
    free(auxCentroids);
    return(maxDist);

}




int main(int argc, char* argv[])
{

	//START CLOCK***************************************
	//clock_t start, end;
	//start = clock();
	double tiempoPar,tiempoMem;
	tiempoMem = omp_get_wtime();

	//**************************************************
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
		exit(-1);
	}

	//Lectura de los datos de entrada
	// lines = numero de puntos;  samples = numero de dimensiones por punto
	int lines = 0, samples= 0;  
	
	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}
	
	float *data = (float*)calloc(lines*samples,sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Error alojando memoria\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

	// prametros del algoritmo. La entrada no esta valdidada
	int K=atoi(argv[2]); 
	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(lines*atof(argv[4])/100.0);
	float maxThreshold=atof(argv[5]);

	
	//poscion de los centroides en data
	int *centroidPos = (int*)calloc(K,sizeof(int));
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(lines,sizeof(int));
	//Otras variables
	float distCent;
    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Error alojando memoria\n");
		exit(-4);
	}
	int it=0;
	int changes = 0;

	// Centroides iniciales
	srand(0);
	int i;
	for(i=0; i<K; i++) 
		centroidPos[i]=rand()%lines;
	
	//Carga del array centroids con los datos del array data
	//los centroides son puntos almacenados en data
	initCentroids(data, centroids, centroidPos, samples, K);

	// Resumen de datos caragos
	printf("\n\tFichero de datos: %s \n\tPuntos: %d\n\tDimensiones: %d\n", argv[1], lines, samples);
	printf("\tNumero de clusters: %d\n", K);
	printf("\tNumero maximo de iteraciones: %d\n", maxIterations);
	printf("\tNumero minimo de cambios: %d [%g%% de %d puntos]\n", minChanges, atof(argv[4]), lines);
	printf("\tPrecision maxima de los centroides: %f\n", maxThreshold);
	
	//END CLOCK*****************************************
	tiempoMem = omp_get_wtime() - tiempoMem;

	printf("\nAlojado de memoria: %f segundos\n", tiempoMem);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	//start = clock();
	tiempoPar = omp_get_wtime();
	//**************************************************

	do{
		it++;
		//Calcula la distancia desde cada punto al centroide
		//Asigna cada punto al centroide mas cercano
		changes=classifyPoints(data, centroids, classMap, lines, samples, K);
		//Recalcula los centroides: calcula la media dentro de cada centoide
		distCent=recalculateCentroids(data, centroids, classMap, lines, samples, K);
		printf("\n[%d] Cambios de cluster: %d\tMax. dist. centroides: %f", it, changes, distCent);
	} while((changes>minChanges) && (it<maxIterations) && (distCent>maxThreshold));

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

	//Escritura en fichero de la clasificacion de cada punto
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	//END CLOCK*****************************************
	tiempoPar = omp_get_wtime() - tiempoPar;
	printf("\nComputacion: %f segundos",  tiempoPar);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	//start = clock();
	double liberacionTime;
	liberacionTime = omp_get_wtime();
	//**************************************************


	//Liberacion de la memoria dinamica
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);

	//END CLOCK*****************************************
	liberacionTime = omp_get_wtime() -liberacionTime;
	printf("\n\nLiberacion: %f segundos\n", liberacionTime);
	fflush(stdout);
	//***************************************************/
	return 0;
}
