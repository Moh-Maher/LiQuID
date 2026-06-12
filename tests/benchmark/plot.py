import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(filename='../../build/benchmarks.csv'):
    df = pd.read_csv(filename)
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. RHS Comparison
    rhs = df[df['type'] == 'rhs']
    sns.barplot(data=rhs, x='label', y='value1', ax=axes[0], palette='magma')
    axes[0].set_title('Solver Efficiency')
    axes[0].set_ylabel('Mean RHS Evals')

    # 2. Scaling
    scale = df[df['type'] == 'scaling']
    axes[1].plot(scale['label'], scale['value2'], marker='o', label='Actual Speedup')
    axes[1].plot(scale['label'], scale['label'], '--', color='gray', label='Ideal')
    axes[1].set_title('Thread Scaling')
    axes[1].set_xlabel('Threads')
    axes[1].set_ylabel('Speedup (x)')
    axes[1].legend()

    # 3. Accuracy vs Cost
    acc = df[df['type'] == 'accuracy']
    axes[2].loglog(acc['value1'], acc['value2'], marker='s', color='green')
    axes[2].set_title('Error vs Computational Cost')
    axes[2].set_xlabel('Mean RHS Evals')
    axes[2].set_ylabel('Absolute Error')
    axes[2].grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()
