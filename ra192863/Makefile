CC=nvcc
CFLAGS=-Xcompiler "-fopenmp"

all:projeto

projeto:
	$(CC) mcm_parallel.cu $(CFLAGS) -o mcm_parallel
	g++ generator.cpp -o generator

run:projeto
	./mcm_parallel arq1.in arq2.in arq3.in arq4.in arq5.in arq6.in

clean:
	rm mcm_parallel generator
