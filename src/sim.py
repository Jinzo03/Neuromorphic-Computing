from brian2 import SpikeGeneratorGroup, run, SpikeMonitor, ms, second

# Example events: list of (neuron_index, time_in_seconds)
events = [
    (0, 0.001),
    (5, 0.005),
    (0, 0.010),
    # ...
]

# convert to Brian-friendly arrays
indices = [e[0] for e in events]
times = [e[1] * second for e in events]

G_in = SpikeGeneratorGroup(N_in, indices, times)  # N_in is total input neurons
spikemon = SpikeMonitor(G_in)

run(0.05*second)
print("Recorded spikes:", spikemon.count[:10])