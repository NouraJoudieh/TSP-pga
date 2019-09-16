cp main.cpp tsp-pga.cpp
#echo "#define CHROMO_LEN 29" | cat - tsp-pga.cpp > temp.cpp && mv temp.cpp tsp-pga.cpp
mpicxx -fopenmp tsp-pga.cpp -o tsp-pga
mpiexec -n 3 ./tsp-pga -f wi10.tsp -p 1000 -g 500 -s 10 -m 16 > result10.csv
rm tsp-pga.cpp
#libreoffice --calc result10.csv &

