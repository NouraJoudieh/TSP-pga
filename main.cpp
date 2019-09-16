
/* 
 
 +------------------------------+
 | Course:                      |
 | I3340 - Parallel Computing   |
 +----------------+-------------+
 |      Name      | File Number |
 +----------------+-------------+
 | Noura Joudieh  |    87259    |
 | Dana  Nada     |    85820    |
 | Hamza Jadid    |    86611    |
 +----------------+-------------+

 
 Created on July 7, 2019, 15:55 PM
 */

#include <cstdlib>  //stdlib.h
#include <iostream> //reading writing like stdio.h
#include <vector>   //like arraylist in java
#include <cstring>  //like string.h
#include <cmath>    //like math.h
#include <algorithm> //has algorithms including sorting shuffling ...
#include <chrono>    //like time.h
#include <random>    //for random generator
#include <numeric>   //has functions like distance,sum,...
#include <ctime>
#include <getopt.h>
#include <mpi.h>
#include <omp.h>
#include "functions.h"

//#define DEBUG 1
#define BUFFER_SIZE 1024
#define NUMBER_OF_CHUNKS 4
#define SELECTION_FACTOR 0.5f
#define SEND_CRITERIA 100

using namespace std; //has the data types of the above libraries

int CHROMOSOME_LENGTH;
int POPULATION_SIZE;
int MAX_NB_GENERATIONS;
int MUTATION_FACTOR;
int PARTIAL_SELECTION;
char cities_file[255];

City *cities;

float **distances_matrix; //two dimensional arrays

Generation generation;
Chromosome maxChromo;



int main(int argc, char **argv)
{

    parse_arguments(argc, argv);

    cout << "Parameters:"
         << "\tPopulation: " << POPULATION_SIZE
         << "\tNumber of Generations: " << MAX_NB_GENERATIONS
         << "\tMutation Rate " << MUTATION_FACTOR << "%"
         << endl;

    cout << "Generation"
         << ","
         << "Fitness" << endl;
    
    MPI_Init(&argc, &argv);
    int numtasks, rank, source = 0, dest, tag = 1;

    

    MPI_Status stat;
    MPI_Request request;
    int flag = 0, recv_request = 0;

    MPI_Datatype mpi_chromosome_type, oldtypes[2];
    int blockcounts[2];
    MPI_Aint offsets[2], extent;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    offsets[0] = 0;
    oldtypes[0] = MPI_INT;
    blockcounts[0] = CHROMO_LEN;
    MPI_Type_extent(MPI_INT, &extent);
    offsets[1] = CHROMO_LEN * extent;
    oldtypes[1] = MPI_DOUBLE;
    blockcounts[1] = 1;
    MPI_Type_struct(2, blockcounts, offsets, oldtypes, &mpi_chromosome_type);
    MPI_Type_commit(&mpi_chromosome_type);

    int local_generation_start = rank * (MAX_NB_GENERATIONS / numtasks);
    int local_generation_end = local_generation_start + (MAX_NB_GENERATIONS / numtasks);
    init_distances_matrix();

    fill_sample_generation(generation);
    maxChromo = generation.population[generation.population.size() - 1];
    MPI_Chromosome sentVector[PARTIAL_SELECTION] = {0};
    MPI_Chromosome recvVector[PARTIAL_SELECTION] = {0};

    for (int i = local_generation_start, counter = 1; i < local_generation_end; i++, counter++)
    {
        vector<Chromosome> recieve;
        if (!recv_request)
        {
            MPI_Irecv(recvVector, PARTIAL_SELECTION, mpi_chromosome_type, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
            recv_request = 1;
        }

        MPI_Test(&request, &flag, &stat);
        if (flag)
        {
            recv_request = 0;
            for (int u = 0; u < PARTIAL_SELECTION; u++)
            {
                Chromosome chromo;
                chromo.fitness = recvVector[u].fitness;

                for (int p = 0; p < CHROMO_LEN; p++)
                {
                    chromo.genes.push_back(recvVector[u].genes[p]);
                }
                if (chromo.fitness == 0)
                {
                    fill_sample_chromosome(chromo);
                    fitness_evaluation(chromo);
                }
                recieve.push_back(chromo);
            }
            generation.population.insert(generation.population.begin(), recieve.begin(), recieve.end());
        }

        if (counter == SEND_CRITERIA)
        {
            for (int j = 0; j < PARTIAL_SELECTION; j++)
            {
                copy(generation.population[POPULATION_SIZE - j - 1].genes.begin(), generation.population[POPULATION_SIZE - j - 1].genes.end(), sentVector[j].genes);
                sentVector[j].fitness = generation.population[POPULATION_SIZE - j - 1].fitness;
            }
            MPI_Isend(sentVector, PARTIAL_SELECTION, mpi_chromosome_type, (rank + 1) % numtasks, 0, MPI_COMM_WORLD, &request);
            counter = 1;
        }
        //add lock file so no two processes enter together
        cout << i + 1 << "," << generation.population[generation.population.size() - 1].fitness << endl;
        next_generation(generation);
        if (generation.population[generation.population.size() - 1].fitness > maxChromo.fitness)
        {
            #pragma omp critical
            maxChromo = generation.population[generation.population.size() - 1];
        }
    }
    if (rank == 0)
    {
        cout << "Best Fitness = " << maxChromo.fitness << endl;
    }
    MPI_Type_free(&mpi_chromosome_type);
    MPI_Finalize();
    return EXIT_SUCCESS;
}

void parse_arguments(int argc, char **argv)
{
    int c;
    while (1)
    {
        static struct option long_options[] = {
            {"file", required_argument, 0, 'f'},
            {"population", required_argument, 0, 'p'},
            {"generation", required_argument, 0, 'g'},
            {"send", required_argument, 0, 's'},
            {"mutation", required_argument, 0, 'm'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}};
        int option_index = 0;
        c = getopt_long(argc, argv, "f:p:g:s:m:h",
                        long_options, &option_index);
        if (c == -1)
            break;
        switch (c)
        {
        case 'f':
            strcpy(cities_file, optarg);
            break;
        case 'p':
            POPULATION_SIZE = atoi(optarg);
            break;
        case 'g':
            MAX_NB_GENERATIONS = atoi(optarg);
            break;
        case 's':
            PARTIAL_SELECTION = atoi(optarg);
            break;
        case 'm':
            MUTATION_FACTOR = atoi(optarg);
            break;
        case 'h':
            help();
            exit(0);
        default:
            abort();
        }
    }
    if (argc != 11)
    {
        help();
        exit(0);
    }
}

void help()
{
    cout << "Usage: ./tsp-pga [OPTIONS]\n"
            "ALL OPTIONS ARE REQUIRED\n"
            "where options are :\n"
            "-p <value>, --population <value>\n"
            "population is a positive integer specifying the population size.\n\n"

            "-g <value>, --generation <value>\n"
            "generations is a positive integer specifying how many generations to run the GA for.\n\n"

            "-s <value>, --send <value>\n"
            "the number of the population that should be preserved from generation to generation.\n\n"

            "-m <value>, --mutation <value>\n"
            "specifies how many mutations to apply to each member of the population."
            
            "-f <value>, --file <value>\n"
            "specifies the file path to apply to the tsp-pga on."
         << endl;
}

void read_cities_from_file()
{
    FILE *fp;
    int nb_cities, line;
    char buffer[BUFFER_SIZE];
    float x, y;

    fp = fopen(cities_file, "r");

    for (int i = 0; i < 4; i++)
        fgets(buffer, BUFFER_SIZE, fp);
    if (!fscanf(fp, "DIMENSION : %d", &nb_cities))
    {
        cout << "Illegal TSP locations file format. Expecting the DIMENSION at line 5." << endl;
        exit(0);
    }
    CHROMOSOME_LENGTH = nb_cities;
    for (int i = 0; i < 2; i++)
        fgets(buffer, BUFFER_SIZE, fp);

    cities = (City *)malloc(sizeof(City) * nb_cities);
    rewind(fp);

    for (int i = 0; i < 7; i++)
        fgets(buffer, BUFFER_SIZE, fp);

    while (fscanf(fp, "%d %f %f", &line, &x, &y) > 0 && line <= nb_cities)
    {
        cities[line - 1].id = line;
        cities[line - 1].x = x;
        cities[line - 1].y = y;
#ifdef DEBUG
        cout << "City " << line << " " << x << " " << y << endl;
#endif
    }
    fclose(fp);
}

inline double get_distance(City city1, City city2) { return sqrt(pow(city1.x - city2.x, 2) + pow(city1.y - city2.y, 2)); }

void print_distances_matrix()
{
    cout << "\n\t";
    for (int i = 0; i < CHROMOSOME_LENGTH; ++i)
        cout << "|" << i << "|\t\t";
    for (int i = 0; i < CHROMOSOME_LENGTH; ++i)
    {
        cout << endl;
        cout << "|" << i << "|";
        for (int j = 0; j < CHROMOSOME_LENGTH; j++)
            cout << "d(" << i << "," << j << ")=" << distances_matrix[i][j] << "\t";
    }
    cout << endl;
}

void init_distances_matrix()
{
    read_cities_from_file();
    //a 2D array needs two allocations one for the main
    distances_matrix = (float **)malloc(sizeof(float *) * CHROMOSOME_LENGTH);
    //and one for each entry of this array knowing it is an array too
    for (int i = 0; i < CHROMOSOME_LENGTH; i++)
        distances_matrix[i] = (float *)calloc(CHROMOSOME_LENGTH, sizeof(float));
#ifdef DEBUG
    print_distances_matrix();
#endif
    for (int i = 0; i < CHROMOSOME_LENGTH - 1; i++)
    {
        for (int j = i + 1; j < CHROMOSOME_LENGTH; j++)
        {
            float distance = get_distance(cities[i], cities[j]);
            distances_matrix[i][j] = distances_matrix[j][i] = distance;
        }
    }
    free(cities); //we care about the distances so we can now free the array
#ifdef DEBUG
    print_distances_matrix();
#endif
}

void fitness_evaluation(Chromosome &chromosome)
{
    float sum = distances_matrix[chromosome.genes[0]][chromosome.genes[CHROMOSOME_LENGTH - 1]];
#pragma omp parallel for reduction(+ \
                                   : sum)

    for (int k = 0; k < CHROMOSOME_LENGTH - 1; k++)
    {
        sum += distances_matrix[chromosome.genes[k]][chromosome.genes[k + 1]];
    }
    chromosome.fitness = 1.0 / sum;
}

void fill_sample_chromosome(Chromosome &chromosome)
{
    for (int i = 0; i < CHROMOSOME_LENGTH; i++)
        chromosome.genes.push_back(i);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine dre(seed);
    shuffle(chromosome.genes.begin(), chromosome.genes.end(), dre);
    fitness_evaluation(chromosome);
}

void print_chromosome(const Chromosome &chromosome)
{
    cout << "Fitness = " << chromosome.fitness << "\t, Genes = ";
    for (int i = 0; i < CHROMOSOME_LENGTH; i++)
        cout << chromosome.genes[i] << "_";
    cout << endl;
}

void get_fittest_index(Generation &generation)
{
    double min = generation.population[0].fitness;
    for (int i = 0; i < POPULATION_SIZE; i++)
    {
        if (generation.population[i].fitness < min)
        {
            min = generation.population[i].fitness;
            generation.fittest_index = i;
        }
    }
#ifdef DEBUG
    cout << "The fittest chromosome is " << generation.fittest_index << " with fitness = " << min << endl;
#endif
}

bool fitter(Chromosome a, Chromosome b)
{
    return (a.fitness < b.fitness);
}

void fill_sample_generation(Generation &generation)
{
    for (int i = 0; i < POPULATION_SIZE; i++)
    {
        Chromosome chromosome;
        fill_sample_chromosome(chromosome);
        generation.population.push_back(chromosome);
    }

    sort(generation.population.begin(), generation.population.end(), fitter);

    get_fittest_index(generation);
#ifdef DEBUG
    cout << "The fittest chromosome is " << generation.fittest_index << endl;
#endif
}

void selection_criteria(Generation &generation)
{
    int population_length = generation.population.size();
    int from = 0;
    int to = 0;
    vector<Chromosome> selected_chromosomes(0), local_selected;
#pragma omp parallel private(local_selected)
    {
        for (int i = 0; i < NUMBER_OF_CHUNKS; i++)
        {
            from = to;
            to = ((i + 1) * 10) + to;
            int max_selection = (to - from) * 0.5;
#pragma omp for
            for (int selection_counter = 0; selection_counter <= max_selection; selection_counter++)
            {
                unsigned seed = chrono::system_clock::now().time_since_epoch().count() + rand();
                srand(seed);
                int random_index = (rand() % (to - from)) + from;
                local_selected.push_back(generation.population[random_index]);
            }
        }
#pragma omp critical
        selected_chromosomes.insert(selected_chromosomes.begin(), local_selected.begin(), local_selected.end());
    }
    generation.population.swap(selected_chromosomes);
}

void mutate(Generation &generation, float mutation_factor)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count() + rand();
    srand(seed);
    int random_index = rand() % generation.population.size();
    Chromosome Chromosome = generation.population[random_index];
    seed = chrono::system_clock::now().time_since_epoch().count() + rand();
    srand(seed);
    int random_probability = rand() % 100;
    int first_index, second_index;
    if (random_probability <= mutation_factor)
    {
        seed = chrono::system_clock::now().time_since_epoch().count() + rand();
        srand(seed);
        first_index = (rand() % (Chromosome.genes.size() + 1)) - 1;
        do
        {
            seed = chrono::system_clock::now().time_since_epoch().count() + rand();
            srand(seed);
            second_index = (rand() % (Chromosome.genes.size() + 1)) - 1;
        } while (first_index == second_index);
        //swap(Chromosome.genes[first_index], Chromosome.genes[second_index]);
    }
}

bool contains(const Chromosome &Chromosome, int gene)
{
    for (int i = 0; i < Chromosome.genes.size(); i++)
    {
        if (Chromosome.genes[i] == gene)
            return true;
    }
    return false;
}
void two_point_crossover(const Chromosome &mother_chromosome, const Chromosome &father_chromosome, Chromosome &brother, Chromosome &sister)
{
    int first_index, second_index;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count() + rand();
    srand(seed);
    first_index = rand() % (mother_chromosome.genes.size() / 2);
    second_index = (rand() % (mother_chromosome.genes.size() / 2)) + mother_chromosome.genes.size() / 2;
    //brother.genes.clear(); //! we are taking a reference so it might be filled
    //sister.genes.clear();
    for (int i = first_index; i < second_index; i++)
    {
        brother.genes.push_back(mother_chromosome.genes[i]);
        sister.genes.push_back(father_chromosome.genes[i]);
        //!use push_back to add to the end and push to add to the front
    }
    //! Now start the crossover
    //now fill the brother from the father and the sister from the mother
    for (int i = 0, counter = 0; i < father_chromosome.genes.size(); i++)
    {
        if (!contains(brother, father_chromosome.genes[i]))
        {
            if (i < brother.genes.size())
            {
                brother.genes.insert(brother.genes.begin() + counter++, father_chromosome.genes[i]);
            }
            else
            {
                brother.genes.push_back(father_chromosome.genes[i]);
            }
        }
    }

    for (int i = 0, counter = 0; i < mother_chromosome.genes.size(); i++)
    {
        if (!contains(sister, mother_chromosome.genes[i]))
        {
            if (i < sister.genes.size())
            {
                sister.genes.insert(sister.genes.begin() + counter++, mother_chromosome.genes[i]);
            }
            else
            {
                sister.genes.push_back(mother_chromosome.genes[i]);
            }
        }
    }
}

void crossover(Generation &generation)
{
    auto magicNumber = 55;
    //#pragma omp parallel for
    for (int i = 0; i < POPULATION_SIZE - magicNumber; i += 2)
    {
        Chromosome brother, sister;
        two_point_crossover(generation.population[i], generation.population[i + 1], brother, sister);
        generation.population.push_back(brother);
        generation.population.push_back(sister);
    }
}
void fitness_evalutaion_generation(Generation &generation)
{
    for (int j = 0; j < generation.population.size(); j++)
    {
        fitness_evaluation(generation.population[j]);
    }
}

void next_generation(Generation &generation)
{
    fitness_evalutaion_generation(generation);
    selection_criteria(generation);
    crossover(generation);
    mutate(generation, MUTATION_FACTOR);
    sort(generation.population.begin(), generation.population.end(), fitter);
}