#include <vector>

#ifndef CHROMO_LEN
#define CHROMO_LEN 10
#endif

typedef struct
{
    int id;
    float x, y;
} City;

typedef struct
{
    std::vector<int> genes;
    double fitness;
} Chromosome;

typedef struct
{
    std::vector<Chromosome> population;
    int fittest_index;
} Generation;

typedef struct
{
    int genes[CHROMO_LEN];
    double fitness;
} MPI_Chromosome;