CXX      := g++
CXXFLAGS := -std=c++17 -O3 -fopenmp

LIQUID_ROOT := ..
INCLUDES    := -I$(LIQUID_ROOT)/include
SOURCES     := $(LIQUID_ROOT)/src/core/rng.cpp

TARGETS := \
    bench1a_sem_scaling \
    bench1b_adaptive_stopping \
    bench2_variance_dependence \
    bench3_fixed_vs_adaptive

all: $(TARGETS)

%: %.cpp
	$(CXX) $< $(SOURCES) $(CXXFLAGS) $(INCLUDES) -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
