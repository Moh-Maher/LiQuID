#!/bin/bash
#----------------------------------------------
# Exit immediately if a command exits with a non-zero status
#----------------------------------------------
set -e

echo "=================================================="
echo " Starting Benchmark Reproducibility Pipeline"
echo "=================================================="
#----------------------------------------------
# Step 1: Create results and figs directories if they don't exist
#----------------------------------------------
echo "[1/4] Preparing output directories..."
mkdir -p results
mkdir -p figs

# Step 2: Clean old binaries and compile the C++ source files
echo -e "\n[2/4] Compiling C++ benchmarks..."
make clean
make all
#----------------------------------------------
# Step 3: Run the compiled C++ benchmarks
#----------------------------------------------
echo -e "\n[3/4] Running benchmarks..."

echo "Running bench1a_sem_scaling..."
./bench1a_sem_scaling

echo "Running bench1b_adaptive_stopping..."
./bench1b_adaptive_stopping

echo "Running bench2_variance_dependence..."
./bench2_variance_dependence

echo "Running bench3_fixed_vs_adaptive..."
./bench3_fixed_vs_adaptive

#----------------------------------------------
# Step 4: Run the Python plotting scripts
#----------------------------------------------
echo -e "\n[4/4] Generating plots..."

PYTHON_EXEC=$(which python3 || which python)
if [ -z "$PYTHON_EXEC" ]; then
    echo "Error: Python is not installed or not in PATH."
    exit 1
fi

echo "Generating plot for bench1 adaptive stopping..."
$PYTHON_EXEC bench1b_adaptive_stopping_plot.py

echo "Generating plot for bench2 variance dependence..."
$PYTHON_EXEC bench2_variance_dependence_plot.py

echo "Generating plot for bench3 fixed vs adaptive..."
$PYTHON_EXEC bench3_fixed_vs_adaptive_plot.py

#-----------------------------------------------------------------------------
# Organize C++ outputs: Move data files (.txt, .csv, .dat) to the results folder
# Note: If your C++ files already save directly to results/, this step safely skips.
#-----------------------------------------------------------------------------
if ls *.csv *.txt *.dat 1> /dev/null 2>&1; then
    echo "Moving generated data files to 'results/' folder..."
    mv *.csv *.txt *.dat results/ 2>/dev/null || true
fi
#-----------------------------------------------------------------------------
# Organize Python outputs: Move image files (.png, .jpg, .pdf) to the figs folder
#-----------------------------------------------------------------------------
if ls *.png *.jpg *.jpeg *.pdf 1> /dev/null 2>&1; then
    echo "Moving generated plots to 'figs/' folder..."
    mv *.png *.jpg *.jpeg *.pdf figs/ 2>/dev/null || true
fi

echo -e "\n=================================================="
echo " Automation Complete!"
echo " Data saved in: ./results/"
echo " Plots saved in: ./figs/"
echo "=================================================="
