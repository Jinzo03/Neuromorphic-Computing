from brian2 import (start_scope, SpikeGeneratorGroup, NeuronGroup, Synapses,
                    SpikeMonitor, StateMonitor, run, ms, second, prefs)
import numpy as np
from synth_dvs_to_brian import generate_moving_bar_events, events_to_spikegenerator_args

# FIX 1: suppress Cython warning, use numpy backend explicitly
prefs.codegen.target = 'numpy'

W, H = 32, 32
events = generate_moving_bar_events(width=W, height=H, duration_s=0.5, fps=200,
                                    bar_width=4, speed_px_per_s=120, direction='right', polarity=0)
indices, times = events_to_spikegenerator_args(events, W, H)

start_scope()

N_in = W * H * 2
G_in = SpikeGeneratorGroup(N_in, indices, times)

eqs = "dv/dt = (-v) / (10*ms) : 1"
N_rec = 128
G = NeuronGroup(N_rec, eqs, threshold='v>1', reset='v=0', method='exact')

S = Synapses(G_in, G, on_pre='v += 0.15')  # FIX 2: 0.02 → 0.15 (neurons can now reach threshold)
S.connect(p=0.1)                            # FIX 2: 0.03 → 0.1  (more connections per neuron)

sm_in = SpikeMonitor(G_in)
sm = SpikeMonitor(G)
vm = StateMonitor(G, 'v', record=[0, 1, 2])

run(0.6 * second)
print("Input spikes:", sm_in.num_spikes, "SNN spikes:", sm.num_spikes)