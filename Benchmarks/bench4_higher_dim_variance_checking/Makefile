CXX      := g++
CXXFLAGS := -std=c++17 -O3 -fopenmp

LIQUID_ROOT := ../..
INCLUDES    := -I$(LIQUID_ROOT)/include
SOURCES     := $(LIQUID_ROOT)/src/core/rng.cpp

TARGETS := \
    bench4_higher_dim_variance \

all: $(TARGETS)

%: %.cpp
	$(CXX) $< $(SOURCES) $(CXXFLAGS) $(INCLUDES) -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
