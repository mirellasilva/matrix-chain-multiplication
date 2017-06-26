#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

using namespace std;

#define LOG 1
#define DEBUG 0
#define BLOCK_SIZE 16
#define MAX 4294967294

// Global variables
int nm, nt;
unsigned long int mult;

// Create global file [for better function calling]
FILE *out;

// Struct to check where is matrix, and to store matrix data
typedef struct matrices{
  float *p;
  int n, m, gpu;
} mat;

// Return time to benchmark
double rtclock()
{
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday (&Tp, &Tzp);
	if (stat != 0) printf("Error return from gettimeofday: %d",stat);
	return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

// CUDA multiplying A x B kernel
__global__ void matrix_mult_gpu(float *a, float *b, float *c, int n, int r, int m)
{ 
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  
  if(col < m && row < n){
    for(int i = 0; i < r; i++)
      sum += a[row*r + i] * b[i*m + col];
       
    c[row*m + col] = sum;
  }
}

// Print parenthesis using recursion
void printParenthesis(int i, int j, unsigned short int *bracket, int *matrix){  
  if(i == j){
    fprintf(out, "[%d]", *matrix);
    *matrix = *matrix + 1;
    return;
  }
  
  fprintf(out, "(");
  
  printParenthesis(i, bracket[i+j*(j-1)/2], bracket, matrix);
    
  printParenthesis(bracket[i+j*(j-1)/2] + 1, j, bracket, matrix);  
  
  fprintf(out, ")"); 
}

// Create matrices, used intead keeping matrices on memory
float* createMatrix(int n, int m, int id){ 
  float *a = (float*) malloc(sizeof(float)*n*m);
  
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
      a[(i*m)+j] = (float) id+0.00001;
  
  return a;
}

// Multiplying A x B serial
float* matrix_mult(float *a, float *b, int n, int r, int m){
  float *c = (float*) calloc(n*m, sizeof(float));

  float sum = 0;
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++){
      for(int k = 0; k < r; k++){
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

// Call kernel to multiply chain of matrices, uses recursion
mat* matrixOrderMultiplication_gpu(int i, int j, unsigned short int *bracket, unsigned short int *p, int chamada){
  float *gA, *gB, *gC;
  mat *a, *b, *c;
  
  c = (mat*) malloc(sizeof(mat));

  if(i == j){    
    c->p = createMatrix(p[i-1], p[i], i-1);
    c->n = p[i-1];
    c->m = p[i];
    c->gpu = 0;
    
    return c;
  }
  
  a = matrixOrderMultiplication_gpu(i, bracket[i+j*(j-1)/2], bracket, p, chamada+1);
    
  b = matrixOrderMultiplication_gpu(bracket[i+j*(j-1)/2] + 1, j, bracket, p, chamada+1);

  // Cria GRID e Blocks
  dim3 dim_grid((a->n - 1) / BLOCK_SIZE + 1, (b->m - 1) / BLOCK_SIZE + 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  // Aloca memória na GPU e Copia memoria    
  if(!a->gpu){
    cudaMalloc((void **)&gA, sizeof(float)*a->n*a->m);
    cudaMemcpy(gA, a->p, sizeof(float)*a->n*a->m, cudaMemcpyHostToDevice);
  } else
    gA = a->p;
  
  if(!b->gpu){
    cudaMalloc((void **)&gB, sizeof(float)*b->n*b->m);
    cudaMemcpy(gB, b->p, sizeof(float)*b->n*b->m, cudaMemcpyHostToDevice);
  } else
    gB = b->p;

  cudaMalloc((void **)&gC, sizeof(float)*a->n*b->m); 
  
  // Executa kernel na GPU
  matrix_mult_gpu<<<dim_grid, dim_block>>>(gA, gB, gC, a->n, a->m, b->m);
  
  // Aguarda processamento da GPU
  cudaDeviceSynchronize();
  
  // Atualiza estruturas
  c->p = gC;
  c->n = a->n;
  c->m = b->m;
  c->gpu = 1;

  // Free nas memorias da GPU
  cudaFree(gA);
  cudaFree(gB);
  
  // Dealoca memoria das estruturas
  free(a);
  free(b);

  if(!chamada){
    c->p = (float*) malloc(sizeof(float)*c->n*c->m);
    cudaMemcpy(c->p, gC, sizeof(float)*c->n*c->m, cudaMemcpyDeviceToHost);
  }
  
  return c;
}

// Serial chain multiplication using recursion
void matrixChainMultiplication(unsigned short int *p, int n){
  float *a, *b;

  a = createMatrix(p[0], p[1], 0);
  
  b = createMatrix(p[1], p[2], 1);

  a = matrix_mult(a, b, p[0], p[1], p[2]);
  
  for(int i = 2; i < n - 1; i++){
    b = createMatrix(p[i], p[i+1], i);
    
    a = matrix_mult(a, b, p[0], p[i], p[i+1]);
  }
  
  if(DEBUG){
    fprintf(out, "\nResult:\n");
  
    for(int i = 0; i < p[0]; i++){
      for(int j = 0; j < p[n-1]; j++)
        fprintf(out, "%f ", a[(i*p[n-1])+j]);

      fprintf(out, "\n");
    }
  }  
}


// Multiply chain of matrices serial
mat* matrixOrderMultiplication(int i, int j, unsigned short int *bracket, unsigned short int *p){  
  mat *a, *b, *c;
  c = (mat*) malloc(sizeof(mat));

  if(i == j){    
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

// Find best parenthesis parallel
unsigned short int* matrixChainOrder_Parallel(unsigned short int p[], int n)
{
  unsigned long int *m = (unsigned long int*) malloc(sizeof(unsigned long  int)*(n+(n*(n-1))/2));

  unsigned short int *bracket = (unsigned short int*) malloc(sizeof(unsigned short int)*(n+(n*(n-1))/2)); 

  // Create pool of threads
  #pragma omp parallel num_threads(nt)
  {
    // Put zero on main diagonal
    #pragma omp for
    for (int i = 1; i < n; i++)
      m[i+i*(i-1)/2] = 0;
  
    for (int L = 2; L < n; L++){
      // Fill next diagonals
      #pragma omp for
      for (int i = 1; i < n-L+1; i++){
        int j = i+L-1;
        
        m[i+j*(j-1)/2] = 4294967294;
        
        for (int k = i; k <= j-1; k++){         
          int q = m[i+k*(k-1)/2] + m[(k+1)+j*(j-1)/2] + p[i-1]*p[k]*p[j];
          
          if (q < m[i+j*(j-1)/2]){
            m[i+j*(j-1)/2] = q;        
            bracket[i+j*(j-1)/2] = k;
          }
        }
      }
    }
  }  

  int matrix = 0;
  
  if(DEBUG){
    fprintf(out, "Optimal Parenthesization is : ");   
    printParenthesis(1, n-1, bracket, &matrix);
  }
  
  if(LOG)
    fprintf(out, "\nOptimal Cost is : %lu\n", m[1+(n-1)*((n-1)-1)/2]);

  free(m);
  
  return bracket;
}

// Find best parenthesis serial
void matrixChainOrder(unsigned short int p[], int n){
  unsigned long int *m = (unsigned long int*) malloc(sizeof(unsigned long int)*(n+(n*(n-1))/2));

  unsigned short int *bracket = (unsigned short int*) malloc(sizeof(unsigned short int)*(n+(n*(n-1))/2));
  
  unsigned long int q;
  
  for(int i = 1; i < n; i++)
    m[i+i*(i-1)/2] = 0;
  
  for(int L = 2; L < n; L++)
  {    
    for(int i = 1; i < n-L+1; i++)
    {
      int j = i+L-1;
      
      m[i+j*(j-1)/2] = MAX;

      for(int k = i; k <= j-1; k++)
      {         
        q = m[i+k*(k-1)/2] + m[(k+1)+j*(j-1)/2] + p[i-1]*p[k]*p[j];
          
        if(q < m[i+j*(j-1)/2])
        {
          m[i+j*(j-1)/2] = q;        
          bracket[i+j*(j-1)/2] = k;
        }
      }
    }
  }
  
  int matrix = 0; 
  
  if(DEBUG){
    fprintf(out, "Optimal Parenthesization is : ");
    printParenthesis(1, n-1, bracket, &matrix);
  }
  
  if(LOG)
    fprintf(out, "\nOptimal Cost is : %lu\n", m[1+(n-1)*((n-1)-1)/2]);

  free(m);
  free(bracket);  
}

int main(int argc, char **argv){
  double start, end, order_s, order_p, mult_s, mult_p, serial, parallel;

  // Create output file 
  out = fopen("output.dat", "w+");

  int arq = 1;
  while(arq < argc){
    order_s = order_p = serial = parallel = mult_s = mult_p = 0;
    
    char filename[50] = "./input/";
    strcat(filename, argv[arq]);

    // Create input file
    FILE *in = fopen(filename, "r+");
    
  
    fprintf(out, "\n%s ------------------------------------------\n", argv[arq]);
    printf("\n%s ------------------------------------------\n", argv[arq]);
  
    // Read input files
    fscanf(in, "%d", &nt);
    fscanf(in, "%d", &nm);
    
    // Creating matrices
    unsigned short int arr[nm+1];

    // Read input data
    for(int i = 0; i < nm+1; i++){
      fscanf(in, "%hu", &arr[i]);
    }
  
    // Get size of input
    int n = sizeof(arr)/sizeof(arr[0]);      

    // Create table for optimal parenthesization
    unsigned short int *bracket = (unsigned short int*) malloc(sizeof(unsigned short int)*(n+(n*(n-1))/2));
  
    /*
     *  Chain Matrix Multiplication
     */

    if(DEBUG){
      mult = 0;
      start = rtclock();
      matrixChainMultiplication(arr, n);
      end = rtclock();
      fprintf(out, "Number of multiplications: %lu\n", mult);
      fprintf(out, "Matrix Chain Multiplication -> Serial: %g\n\n", end-start);
    }
  
    /*
     *  Matrix Chain Multiplication Parentheses
     */
  
    start = rtclock();
    matrixChainOrder(arr, n);
    end = rtclock();
    order_s = end-start;
    fprintf(out, "Matrix Chain Order -> Serial: %g\n\n", end-start);
  
    start = omp_get_wtime();
    bracket = matrixChainOrder_Parallel(arr, n);
    end = omp_get_wtime();
    order_p = end-start;
    fprintf(out, "Matrix Chain Order -> Parallel: %g\n\n", end-start);

    /*
     *  Multiplication with order
     */

    // Create struct used to store c matrices
    mat *c = (mat*) malloc(sizeof(mat));
  
    mult = 0;  
    start = rtclock();
    c = matrixOrderMultiplication(1, n - 1, bracket, arr);
    end = rtclock();
    if(DEBUG){
      fprintf(out, "Result:\n");

      for(int i = 0; i < c->n; i++){
        for(int j = 0; j < c->m; j++)
          fprintf(out, "%f ", c->p[(i*c->m)+j]);
      
        fprintf(out, "\n");
      }
    }
    mult_s = end-start;
    fprintf(out, "Number of multiplications: %lu\n", mult);
    fprintf(out, "Matrix Chain Multiplication with optimal parenthesization -> Serial: %g\n\n", end-start);
    free(c->p);
    free(c);


    start = rtclock();
    c = matrixOrderMultiplication_gpu(1, n - 1, bracket, arr, 0);
    end = rtclock();   
    if(DEBUG){
      fprintf(out, "Result:\n");

      for(int i = 0; i < c->n; i++){
        for(int j = 0; j < c->m; j++)
          fprintf(out, "%f ", c->p[(i*c->m)+j]);
      
        fprintf(out, "\n");
      }
    }
    mult_p = end-start;
    fprintf(out, "Number of multiplications: %lu\n", mult);
    fprintf(out, "Matrix Chain Multiplication with optimal parenthesization -> Parallel: %g\n", end-start);

    serial = order_s + mult_s;
    parallel = order_p + mult_p;

    fprintf(out, "\nSpeedup Order: %g\n", order_s/order_p);
    fprintf(out, "\nSpeedup Matrix: %g\n", mult_s/mult_p);
    fprintf(out, "\nSpeedup Total: %g\n", serial/parallel);
    printf("\nSpeedup: %g\n", serial/parallel);

    free(bracket);
    free(c);
    fclose(in);
    arq++;
  }
  
  
  fclose(out);  
  
  return 0;
}
