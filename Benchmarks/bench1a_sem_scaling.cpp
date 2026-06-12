// benchmarks/bench1a_sem_scaling.cpp
// -----------------------------------------------------------------------------------
// Benchmark 1a: Statistical Convergence Validation
//
// Protocol:
//   Fix system (two-level decay), run ensembles of N trajectories for
//   N = 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000.
//   For each N, record: mean, SEM, rel_SEM, |mean − exact|.
//   Verify that SEM ∝ 1/√N to within statistical noise.
//
// Output: bench_convergence.csv
//   N,mean_sz,exact_sz,abs_error,sem,rel_sem,wall_s
//
// Expected physics:
//   Two-level decay: ⟨σ_z(T=1)⟩ = 2·exp(−1) − 1 ≈ −0.2642
//   SEM ∝ 1/√N  ⟹  log(SEM) = −½ log(N) + const  (slope should be −0.5)
// -----------------------------------------------------------------------------------

#include "liquid/liquid.hpp"
#include <algorithm>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <array>
#include <numeric>
using namespace liquid;
using namespace liquid::ensemble;

static DenseOpenSystem make_sys() {
    DenseOperator H(2);
    H(0,0) = Scalar{0.5, 0}; H(1,1) = Scalar{-0.5, 0};
    DenseOperator L(2);
    L(1,0) = Scalar{1.0, 0};
    std::vector<DenseOperator> ops; ops.push_back(std::move(L));
    return DenseOpenSystem(std::move(H), LindbladSet<DenseTag>(std::move(ops)));
}

struct ConvPoint {
	std::size_t N;
	double mean_sz;
	double exact_sz;
	double abs_error;
	double sem;
	double sem_sqrtN;
	double rel_sem;
	double wall_s;
};

static ConvPoint run_N(std::size_t N, double T, Seed seed) {
    DenseOperator sz_op(2);
    sz_op(0,0) = Scalar{1,0}; sz_op(1,1) = Scalar{-1,0};

    auto sim = SimulationBuilder{}
        .hamiltonian([]{ DenseOperator H(2); H(0,0)=Scalar{0.5,0}; H(1,1)=Scalar{-0.5,0}; return H; }())
        .collapse_operator([]{ DenseOperator L(2); L(1,0)=Scalar{1.0,0}; return L; }())
        .observe("sz", std::move(sz_op))
        .seed(seed)
        .dt(1e-3)
        .min_trajectories(N)
        .max_trajectories(N)
        .build();

    StateVector psi0(2); psi0[0] = Scalar{1,0};

    auto t0 = std::chrono::steady_clock::now();
    auto result = sim.run(psi0, 0.0, T);
    double wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    const double exact = 2.0 * std::exp(-1.0 * T) - 1.0;
	
	const auto& obs = result.observables.at(0);

	ConvPoint pt;
	pt.N         = N;
	pt.mean_sz   = obs.mean;
	pt.exact_sz  = exact;
	pt.abs_error = std::abs(obs.mean - exact);
	pt.sem       = obs.sem;
	pt.sem_sqrtN = obs.sem * std::sqrt(static_cast<double>(N));
	pt.rel_sem   = obs.rel_sem;
	pt.wall_s    = wall_s;

	return pt;
}

template<typename T>
T median5(std::array<T,5>& x)
{
	std::sort(x.begin(), x.end());
	return x[2];
}
int main() {
    const double T = 1.0;   // ⟨σ_z(1)⟩ = 2e^{-1} − 1 ≈ −0.2642
    const double exact = 2.0 * std::exp(-T) - 1.0;

    // Trajectory counts spanning 4 orders of magnitude
    const std::size_t Ns[] = {
        10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
        10000, 20000, 50000, 100000
    };


    FILE* f = std::fopen("bench1a_sem_scaling.csv", "w");
    std::fprintf(f, "N,mean_sz,exact_sz,abs_error,sem,sem_sqrtN,rel_sem,wall_s\n");

	std::vector<double> Ns_fit;
	std::vector<double> sems_fit;

	std::vector<double> sem_sqrtN_values;

	int inside_3sigma = 0;
	int total_points  = 0;
    // Run each N with 5 different seeds and report median
    // (removes seed-specific outliers from the convergence plot)
    	for (std::size_t N : Ns)
	{
		std::array<double,5> means{};
		std::array<double,5> sems{};
		std::array<double,5> rel_sems{};
		std::array<double,5> walls{};
		std::array<double,5> errors{};
		std::array<double,5> sem_sqrtNs{};

		for (int k = 0; k < 5; ++k)
		{
			auto pt = run_N(
				N,
				T,
				static_cast<Seed>(N * 1234ULL + k)
			);

			means[k]       = pt.mean_sz;
			sems[k]        = pt.sem;
			rel_sems[k]    = pt.rel_sem;
			walls[k]       = pt.wall_s;
			errors[k]      = pt.abs_error;
			sem_sqrtNs[k]  = pt.sem_sqrtN;
		}

		const double mean_sz   = median5(means);
		const double sem       = median5(sems);
		const double rel_sem   = median5(rel_sems);
		const double wall_s    = median5(walls);
		const double abs_error = median5(errors);
		const double sem_sqrtN = median5(sem_sqrtNs);

		std::printf(
			"%-8zu %+12.6f %+12.6f %-12.2e %-12.2e %-10.4f\n",
			N,
			mean_sz,
			exact,
			abs_error,
			sem,
			wall_s
		);

		std::fprintf(
			f,
			"%zu,%.8f,%.8f,%.6e,%.6e,%.6e,%.6e,%.6f\n",
			N,
			mean_sz,
			exact,
			abs_error,
			sem,
			sem_sqrtN,
			rel_sem,
			wall_s
		);

		if (N >= 100)
		{
			Ns_fit.push_back(std::log(static_cast<double>(N)));
			sems_fit.push_back(std::log(sem));
		}

		sem_sqrtN_values.push_back(sem_sqrtN);

		if (abs_error <= 3.0 * sem)
			++inside_3sigma;

		++total_points;
	}

    std::fclose(f);
    
 	double sx = 0.0;
	double sy = 0.0;
	double sxx = 0.0;
	double sxy = 0.0;

	for (std::size_t i = 0; i < Ns_fit.size(); ++i)
	{
		sx  += Ns_fit[i];
		sy  += sems_fit[i];
		sxx += Ns_fit[i] * Ns_fit[i];
		sxy += Ns_fit[i] * sems_fit[i];
	}

	const double n = static_cast<double>(Ns_fit.size());

	const double slope =
		(n * sxy - sx * sy) /
		(n * sxx - sx * sx);

	const double intercept =
		(sy - slope * sx) / n;
		double ss_tot = 0.0;
	double ss_res = 0.0;

	const double mean_y = sy / n;

	for (std::size_t i = 0; i < Ns_fit.size(); ++i)
	{
		const double pred =
			intercept + slope * Ns_fit[i];

		ss_tot +=
			(sems_fit[i] - mean_y) *
			(sems_fit[i] - mean_y);

		ss_res +=
			(sems_fit[i] - pred) *
			(sems_fit[i] - pred);
	}

	const double r2 =
		1.0 - ss_res / ss_tot;
		
		const double mean_sem_sqrtN =
		std::accumulate(
			sem_sqrtN_values.begin(),
			sem_sqrtN_values.end(),
			0.0
		) /
		sem_sqrtN_values.size();

	double max_dev = 0.0;

	for (double x : sem_sqrtN_values)
	{
		max_dev =
			std::max(
				max_dev,
				std::abs(x - mean_sem_sqrtN)
			);
	}

	const double variation_pct =
		100.0 * max_dev / mean_sem_sqrtN;
	std::printf("\n");
	std::printf("Statistical Validation Results\n");
	std::printf("----------------------------------------\n");
	std::printf("Slope(log SEM vs log N) = %.4f\n", slope);
	std::printf("R^2                     = %.4f\n", r2);
	std::printf("SEM*sqrt(N) variation   = %.2f %%\n", variation_pct);
	std::printf(
		"Inside 3σ envelope      = %d/%d\n",
		inside_3sigma,
		total_points
	);
	
	return 0;
}
