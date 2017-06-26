#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv){
  int n = atoi(argv[1]);
  int min = atoi(argv[2]);
  int max = atoi(argv[3]);

  srand(time(NULL));
  
  printf("2 %d\n", n);

  for(int i = 0; i < n+1; i++){
    int r = min + rand()%(max-min);
    printf("%d ", ((r > 2)? r:2));
  }

  printf("\n");
}
