import numpy as np, qutip as qt, pandas as pd
import matplotlib.pyplot as plt

# QuTiP system (identical parameters)
w=2*np.pi; th=0.2*np.pi; g1=0.5; g2=0.2; nth=0.5
H = w*(np.cos(th)*qt.sigmaz() + np.sin(th)*qt.sigmax())
c_ops = [
  np.sqrt(g1*(nth+1))*qt.sigmam(),
  np.sqrt(g1*nth)*qt.sigmap(),
  np.sqrt(g2)*qt.sigmaz()
]
psi0 = qt.basis(2,0)
tlist = np.linspace(0,10,200)

# mesolve (density matrix — exact)
r_me = qt.mesolve(H, psi0, tlist, c_ops,
         [qt.sigmax(), qt.sigmay(), qt.sigmaz()])

# mcsolve (Monte Carlo — stochastic, like LiQuID)
r_mc = qt.mcsolve(H, psi0, tlist, c_ops,
         [qt.sigmax(), qt.sigmay(), qt.sigmaz()], ntraj=500)

# Load and plot LiQuID results
liq = pd.read_csv('qubit_dynamics_liquid.csv')
bloch = pd.read_csv('qubit_dynamics_bloch.csv')

fig, axes = plt.subplots(3, 1, figsize=(10,8), sharex=True)
labels = ['<σ_x>', '<σ_y>', '<σ_z>']
me_cols = [r_me.expect[0], r_me.expect[1], r_me.expect[2]]
mc_cols = [r_mc.expect[0], r_mc.expect[1], r_mc.expect[2]]
liq_cols= [liq.sx, liq.sy, liq.sz]
liq_sems= [liq.sx_sem, liq.sy_sem, liq.sz_sem]
bl_cols = [bloch.sx, bloch.sy, bloch.sz]
for i, ax in enumerate(axes):
  ax.plot(tlist, me_cols[i], 'k-', lw=2, label='QuTiP mesolve')
  ax.plot(tlist, mc_cols[i], 'b--', lw=1, label='QuTiP mcsolve')
  ax.plot(bloch.t, bl_cols[i], 'g:', lw=1.5, label='Bloch eq.')
  ax.errorbar(liq.t, liq_cols[i], yerr=liq_sems[i],
      fmt='ro', ms=5, capsize=3, label='LiQuID MCWF')
  ax.set_ylabel(labels[i]); ax.legend(loc='upper right')
axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.savefig('qubit_dynamics_comparison.pdf', dpi=150)
plt.show()
