#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#define LOG 1
#define DEBUG 0
#define BLOCK_SIZE 16
#define MAX 4294967294

// Global variables
int nm, nt;
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
			a[(i*m)+j] = (float) id+0.00001;

	return a;
}

float* matrix_mult(float *a, float *b, int n, int r, int m) {
	float *c = (float*) calloc(n*m, sizeof(float));

	float sum = 0;
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++) {
			for(int k = 0; k < r; k++) {
				sum += a[(i*r)+k] * b[(k*m)+j];

				mult++;
			}

			c[(i*m)+j] = sum;
			sum = 0;
		}

	free(a);
	free(b);

	return c;
}

mat* matrixOrderMultiplication(
	int i,
	int j,
	unsigned short int *bracket,
	unsigned short int *p
) {
	mat *a, *b, *c;
	c = (mat*) malloc(sizeof(mat));

	if(i == j) {
		c->p = createMatrix(p[i-1], p[i], i-1);
		c->n = p[i-1];
		c->m = p[i];

		return c;
	}

	a = matrixOrderMultiplication(i, bracket[i+j*(j-1)/2], bracket, p);

	b = matrixOrderMultiplication(bracket[i+j*(j-1)/2] + 1, j, bracket, p);

	c->p = matrix_mult(a->p, b->p, a->n, a->m, b->m);
	c->n = a->n;
	c->m = b->m;

	free(a);
	free(b);

	return c;
}

void matrixChainMultiplication(unsigned short int *p, int n) {
	float *a, *b;

	a = createMatrix(p[0], p[1], 0);

	b = createMatrix(p[1], p[2], 1);

	a = matrix_mult(a, b, p[0], p[1], p[2]);

	for(int i = 2; i < n - 1; i++) {
		b = createMatrix(p[i], p[i+1], i);

		a = matrix_mult(a, b, p[0], p[i], p[i+1]);
	}

	if(DEBUG) {
		fprintf(output_file, "\nResult:\n");

		for(int i = 0; i < p[0]; i++) {
			for(int j = 0; j < p[n-1]; j++)
				fprintf(output_file, "%f ", a[(i*p[n-1])+j]);

			fprintf(output_file, "\n");
		}
	}
}

unsigned short int* matrixChainOrder(unsigned short int p[], int n) {
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
				q = m[i+k*(k-1)/2] + m[(k+1)+j*(j-1)/2] + p[i-1]*p[k]*p[j];

				if(q <= m[i+j*(j-1)/2]) {
					m[i+j*(j-1)/2] = q;
					bracket[i+j*(j-1)/2] = k;
				}
			}
		}
	}

	int matrix = 0;

	if(DEBUG) {
		fprintf(output_file, "Optimal Parenthesization is : ");
		printParenthesis(1, n-1, bracket, &matrix);
	}

	if(LOG)
		fprintf(output_file, "\nOptimal Cost is : %lu\n", m[1+(n-1)*((n-1)-1)/2]);

	free(m);

	return bracket;
}

int main(int argc, char **argv) {
	double start, end, order_s, order_p, serial, parallel;

	// Create output_fileput file
	output_file = fopen("output_fileput.dat", "w+");

	int arq = 1;
	while(arq < argc) {
		char filename[50] = "./input/";
		strcat(filename, argv[arq]);

		FILE *in = fopen(filename, "r+");
		// Create input file

		fprintf(output_file, "\n%s -----------------------------------", argv[arq]);
		printf("\n%s -----------------------------------", argv[arq]);

		// Read input files
		fscanf(in, "%d", &nt);
		fscanf(in, "%d", &nm);

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
		unsigned short int *bracket =
			(unsigned short int*) malloc(sizeof(unsigned short int)*(n+(n*(n-1))/2));

		/*
		 *  Matrix Chain Multiplication Parentheses
		 */

		start = rtclock();
		bracket = matrixChainOrder(dimensions, n);
		end = rtclock();
		order_s = end-start;
		fprintf(output_file, "Matrix Chain Order -> Serial: %g\n\n", end-start);

		/*
		 *  Multiplication with order
		 */

		// Create struct used to store c matrices
		mat *c = (mat*) malloc(sizeof(mat));

		mult = 0;
		start = rtclock();
		c = matrixOrderMultiplication(1, n - 1, bracket, dimensions);
		end = rtclock();
		if(DEBUG) {
			fprintf(output_file, "Result:\n");

			for(int i = 0; i < c->n; i++) {
				for(int j = 0; j < c->m; j++)
					fprintf(output_file, "%f ", c->p[(i*c->m)+j]);

				fprintf(output_file, "\n");
			}
		}
		serial = order_s + (end-start);
		fprintf(output_file, "Number of multiplications: %lu\n", mult);
		fprintf(
			output_file,
			"Matrix Chain Multiplication with optimal parenthesization -> " \
				"Serial: %g\n\n",
			end-start
		);
		free(c->p);
		free(c);

		free(bracket);
		fclose(in);
		arq++;
	}

	fclose(output_file);

	return 0;
}
