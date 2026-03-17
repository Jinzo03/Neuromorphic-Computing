from brian2 import SpikeGeneratorGroup, run, SpikeMonitor, ms, second
import numpy as np

# Example events: list of (neuron_index, time_in_seconds)
events = [
    (0, 0.001),
    (5, 0.005),
    (0, 0.010),
    # ...
]

# Convert to Brian-friendly arrays
indices = np.array([e[0] for e in events], dtype=int)
times   = np.array([e[1] for e in events]) * second

# Derive N_in from the highest neuron index referenced
N_in = int(indices.max()) + 1  # ← fix: was undefined

G_in = SpikeGeneratorGroup(N_in, indices, times)
spikemon = SpikeMonitor(G_in)

run(0.05 * second)
print("Recorded spikes:", spikemon.count[:10])