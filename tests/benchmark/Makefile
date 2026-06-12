# --- Universal Logic ---
# Detect the first .cpp file in the folder
SRC := $(wildcard *.cpp)
# Create a name for the output based on the filename (removes .cpp)
NAME := $(SRC:.cpp=)

# --- Compiler Settings ---
# Uses the exact flags from your original files 
default:
	g++ -std=c++17 -O2 -Iinclude qubit_dynamics.cpp -I/home/mohammed-maher/Dropbox/LiQuID/include     -L/home/mohammed-maher/Dropbox/LiQuID/libliquid     -lliquid -lpthread -ldl -o Simulation.out

clear:
	rm -f *.out *.dat *.pdf *.csv
