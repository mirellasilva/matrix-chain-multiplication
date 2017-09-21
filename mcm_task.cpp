#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#define LOG 1
#define DEBUG 0
#define BLOCK_SIZE 16
#define MAX 4294967294

// Global variables
int nm, nt, contador;
unsigned long int mult;

// Create global file [for better function calling]
FILE *output_file;

typedef struct matrices{
	float *p;
	int n, m, gpu;
} mat;

double rtclock() {
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday (&Tp, &Tzp);
	if (stat != 0) printf("Error return from gettimeofday: %d",stat);
	return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void printParenthesis(int i, int j, unsigned short int *bracket, int *matrix) {
	if(i == j) {
		fprintf(output_file, "[%d]", *matrix);
		*matrix = *matrix + 1;
		return;
	}

	fprintf(output_file, "(");

	printParenthesis(i, bracket[i+j*(j-1)/2], bracket, matrix);

	printParenthesis(bracket[i+j*(j-1)/2] + 1, j, bracket, matrix);

	fprintf(output_file, ")");
}

float* createMatrix(int n, int m, int id) {
	float *a = (float*) malloc(sizeof(float)*n*m);
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			a[(i*m)+j] = i==j ? 1 : 0;

	return a;
}

float* matrixMulti(float *a, float *b, int n, int r, int m) {
	float *c = (float*) calloc(n*m, sizeof(float));

	float sum = 0;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			for(int k = 0; k < r; k++) {
				sum += a[(i*r)+k] * b[(k*m)+j];

				mult++;
			}

			c[(i*m)+j] = sum;
			sum = 0;
		}
	}

	free(a);
	free(b);

	return c;
}

/*
 * @brief Multiplies matrices following best parenthesization, serial
 */
mat* matrixOrderMultiplication(
		int i,
		int j,
		unsigned short int *bracket,
		unsigned short int *dim) {
	mat *a, *b, *c;
	c = (mat*) malloc(sizeof(mat));

	if(i == j) {
		c->p = createMatrix(dim[i-1], dim[i], i-1);
		c->n = dim[i-1];
		c->m = dim[i];

		return c;
	}

	a = matrixOrderMultiplication(i, bracket[i+j*(j-1)/2], bracket, dim);

	b = matrixOrderMultiplication(bracket[i+j*(j-1)/2] + 1, j, bracket, dim);

	c->p = matrixMulti(a->p, b->p, a->n, a->m, b->m);
	c->n = a->n;
	c->m = b->m;

	free(a);
	free(b);

	return c;
}

/*
 * @brief Finds best parenthesization, serial
 */
unsigned short int* matrixChainOrder(unsigned short int dim[], int n) {
	unsigned long int *m =
		(unsigned long int*) malloc(sizeof(unsigned long int)*(n+(n*(n-1))/2));

	unsigned short int *bracket =
		(unsigned short int*) malloc(sizeof(unsigned short int)*(n+(n*(n-1))/2));

	unsigned int q;

	for(int i = 1; i < n; i++)
		m[i+i*(i-1)/2] = 0;

	for(int L = 2; L < n; L++) {
		for(int i = 1; i < n-L+1; i++) {
			int j = i+L-1;

			m[i+j*(j-1)/2] = MAX;

			for(int k = i; k <= j-1; k++) {
				q = m[i+k*(k-1)/2] + m[(k+1)+j*(j-1)/2] + dim[i-1]*dim[k]*dim[j];

				if(q <= m[i+j*(j-1)/2]) {
					m[i+j*(j-1)/2] = q;
					bracket[i+j*(j-1)/2] = k;
				}
			}
		}
	}

	int matrix = 0;

	if(DEBUG) {
		fprintf(output_file, "Optimal Parenthesization is: ");
		printParenthesis(1, n-1, bracket, &matrix);
	}

	if(LOG)
		fprintf(output_file, "\nOptimal Cost is : %lu\n", m[1+(n-1)*((n-1)-1)/2]);

	free(m);

	return bracket;
}

/*
 * @brief Multiplies matrices parallel
 */
void matrixMultiplicationPar(
		int i,
		int j,
		unsigned short int *dim,
		mat** results) {


	if(i == j) {
		mat *c = (mat*) malloc(sizeof(mat));
		c->p = createMatrix(dim[i], dim[i+1], i);
		c->n = dim[i];
		c->m = dim[i+1];
		results[i] = c;
		return;
	}

	int k = (j-i)/2;
	if (DEBUG) {
		printf("\nspliting: (%d - %d) (%d, %d)", i, i+k, i+k+1, j);
	}

	matrixMultiplicationPar(i, i+k, dim, results);
	matrixMultiplicationPar(i+k+1, j, dim, results);

	#pragma omp task depend(in: results[i+k+1]) depend(inout: results[i]) 
	{

		mat *c = (mat*) malloc(sizeof(mat));
		if (DEBUG) {
			printf("\nmultiplying: %d * %d = %d", i, i+k+1, j);
		}	
		mat *a, *b;
		a = results[i];
		b = results[i+k+1];
		#pragma omp target map(to: a,b) map(from: c)
		c->p = matrixMulti(a->p, b->p, a->n, a->m, b->m);
		c->n = a->n;
		c->m = b->m;
		free(a);
		free(b);

		results[i] = c;
	}
}

int main(int argc, char **argv) {
	double start, end, order_s, order_p, mult_s, mult_p, serial, parallel;

	// Create output_fileput file
	output_file = fopen("output_fileput_target.dat", "w+");

	int arq = 1;
	while(arq < argc) {
		char filename[50] = "./input/";
		strcat(filename, argv[arq]);

		FILE *in = fopen(filename, "r+");
		// Create input file

		fprintf(output_file, "\n%s -----------------------------------", argv[arq]);
		printf("%s -----------------------------------", argv[arq]);

		// Read input files
		fscanf(in, "%d", &nt);
		fscanf(in, "%d", &nm);
		nt = 32; // explicitly overwriting for workstation

		// nm = # of matrices
		// n = # of matrix dimensions
		int n = nm + 1;

		// Creating array of matrix dimensions
		unsigned short int *dimensions =
			(unsigned short int*) malloc(sizeof(unsigned short int) * n);

		// Read input data
		for(int i = 0; i < n; i++) {
			fscanf(in, "%hu", &dimensions[i]);
		}

		// Create table for optimal parenthesization
		unsigned short int *bracket;

		/*
		 *  Matrix Chain Multiplication Parentheses
		 */
		start = rtclock();
		bracket = matrixChainOrder(dimensions, n);
		end = rtclock();
		order_s = end-start;
		fprintf(output_file, "Matrix Chain Order -> Serial: %.10f\n\n", end-start);

		/*
		 *  Multiplication with order
		 */
		mat *c;
		mult = 0;
		start = rtclock();
		c = matrixOrderMultiplication(1, nm, bracket, dimensions);
		end = rtclock();
		if(DEBUG) {
			fprintf(output_file, "Result:\n");

			for(int i = 0; i < c->n; i++) {
				for(int j = 0; j < c->m; j++)
					fprintf(output_file, "%.1f ", c->p[(i*c->m)+j]);

				fprintf(output_file, "\n");
			}
		}
		serial = order_s + (end-start);
		fprintf(output_file, "Number of multiplications: %lu\n", mult);
		fprintf(
				output_file,
				"Matrix Chain Multiplication with optimal parenthesization -> " \
				"Serial: %.10f\n\n",
				end-start
		       );
		free(c->p);
		free(c);

		mult = 0;
		contador = 0;
		mat** results = (mat**) malloc(sizeof(mat*)*n);
		start = omp_get_wtime();
		#pragma omp parallel num_threads(nt)
		#pragma omp single nowait
		matrixMultiplicationPar(0, nm, dimensions, results);
		#pragma omp taskwait
		end = omp_get_wtime();
		c = results[0];
		if(DEBUG) {
			fprintf(output_file, "Result:\n");

			for(int i = 0; i < c->n; i++) {
				for(int j = 0; j < c->m; j++)
					fprintf(output_file, "%.1f ", c->p[(i*c->m)+j]);

				fprintf(output_file, "\n");
			}
		}
		mult_p = end-start;
		fprintf(output_file, "Number of multiplications: %lu\n", mult);
		fprintf(output_file, "Matrix Chain Multiplication " \
				" -> Parallel: %.10f\n", end-start);

		serial = order_s + mult_s;
		parallel = mult_p;

		fprintf(output_file, "\nSpeedup Matrix Multi: %.2f\n", mult_s/mult_p);
		fprintf(output_file, "\nSpeedup Total: %.2f\n", serial/parallel);
		printf("\nSpeedup: %g\n\n", serial/parallel);
		printf("\nSerial: %.5f, Par: %.5f\n\n", serial, parallel);

		free(bracket);
		fclose(in);
		arq++;
	}

	fclose(output_file);

	return 0;
}
