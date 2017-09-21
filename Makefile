CC=clang++
CFLAGS="-fopenmp"

all:projeto

projeto:
	$(CC) mcm_task.cpp $(CFLAGS) -o mcm_task
	g++ generator.cpp -o generator

run:projeto
	./mcm_task arq1.in arq2.in arq3.in arq4.in arq5.in arq6.in arq10.in

clean:
	rm mcm_task generator
