#include "datatypes.h"

void parse_arguments(int, char **);
void help();
void init_distances_matrix();

void fitness_evaluation(Chromosome &);
void fill_sample_chromosome(Chromosome &);
void print_chromosome(const Chromosome &);

void get_fittest_index(Generation &);
void fill_sample_generation(Generation &);

void selection_criteria(Generation &, int, float);
void mutate(Chromosome &);
void two_point_crossover(const Chromosome &, const Chromosome &, Chromosome &, Chromosome &);
void crossover(Generation &);
void next_generation(Generation &);
void fitness_evalutaion_generation(Generation &);