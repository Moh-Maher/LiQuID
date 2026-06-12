#!/bin/bash

# 1. Clean and Build
clear
echo "Preparing build..."

if ls *.dat *.out *.pdf >/dev/null 2>&1; then
    echo "Generated files found — cleaning first..."
    make clear
else
    echo "No generated files found — skipping clean."
fi

echo "Building..."
make -j 3 || exit 1


# 2. Identify the Binary
BINARY=$(ls *.out | head -n 1)
if [ -z "$BINARY" ]; then
    echo "No executable (.out) found!"
    exit 1
fi

# 3. Run the C++ Simulation first to generate raw data
echo "Running Simulation: $BINARY..."
./$BINARY

# 4. Run Data Generation (QuTip scripts)
# This finds scripts that start with 'QuTip' and runs them first
echo "Generating comparison data..."
for gen_script in QuTip*.py; do
    if [ -f "$gen_script" ]; then
        echo "Executing $gen_script..."
        python3 "$gen_script"
    fi
done

# 5. Run Plotting Scripts
# This runs the remaining scripts (plot.py, rabi.py, fixd_plot.py, etc.)
echo "Generating plots..."
for plot_script in *.py; do
    # Skip the QuTip scripts because they already ran
    if [[ -f "$plot_script" && ! "$plot_script" =~ ^QuTip ]]; then
        echo "Executing $plot_script..."
        python3 "$plot_script"
    fi
done

echo "-----------------------------------------------------"
echo "Done. All data generated and plotted."
