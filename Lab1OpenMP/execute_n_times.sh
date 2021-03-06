#!/bin/bash

if [[ "$#" -ne 1 || $1 -lt 1 ]]
then
    echo "Enter a valid number of repetitions for each flag"
    exit -1
fi

all:
	g++ main.cpp -o main -fopenmp

clean:
	rm main

echo ./main.exe $1 >> execution_times.txt

all:
	g++ main.cpp -o main -fopenmp -O

clean:
	rm main

echo ./main.exe $1 >> execution_times.txt

all:
	g++ main.cpp -o main -fopenmp -O2

clean:
	rm main

echo ./main.exe $1 >> execution_times.txt

all:
	g++ main.cpp -o main -fopenmp -O3

clean:
	rm main

echo ./main.exe $1 >> execution_times.txt

all:
	g++ main.cpp -o main -fopenmp -Ofast

clean:
	rm main

echo ./main.exe $1 >> execution_times.txt




