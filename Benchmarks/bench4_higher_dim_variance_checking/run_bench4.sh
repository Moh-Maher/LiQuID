#!/bin/bash

# Exit immediately if any command fails
set -e

# Step 0: Clean old binaries and compile the C++ source files
echo -e "\n[0] Compiling C++ benchmarks..."
make clean
make all
echo "=================================================="
echo " Starting Benchmark 4: Higher-Dimensional Variance"
echo "=================================================="

# Step 1: Reference Run
echo -n "Step 1/4: Running reference stage... "
./bench4_higher_dim_variance ref
echo "Done."

# Step 2: Adaptive Sweeps for all observables (indices 0 to 4)
echo "Step 2/4: Running parameter sweeps (observables 0-4)..."
for i in {0..4}; do
    echo "--- Sweep for observable index $i ---"
    ./bench4_higher_dim_variance sweep $i
done
echo "All sweeps completed."

# Step 3: Finalize (Ratios, Fixed-N comparison, and Summary)
echo -n "Step 3/4: Running finalize stage... "
./bench4_higher_dim_variance finalize
echo "Done."

# Step 4: Generate Publication-Quality Plots
echo -n "Step 4/4: Generating plots via Python script... "
python bench4_higher_dim_variance_plot.py
echo "Done."

echo "=================================================="
echo " Benchmark 4 and plotting completed successfully!"
echo "=================================================="
