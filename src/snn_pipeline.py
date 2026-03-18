"""
step3_snn_pipeline.py
Run: python step3_snn_pipeline.py
Dependencies: numpy, matplotlib, scikit-learn, brian2
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import start_scope, SpikeGeneratorGroup, NeuronGroup, Synapses, SpikeMonitor, StateMonitor, run, ms, second, prefs

# ensure numpy codegen target for Brian2 (safer on many systems)
prefs.codegen.target = 'numpy'

# ---------- Synthetic event generator ----------
def generate_moving_bar_events(width=32, height=32,
                               duration_s=0.4, fps=200,
                               bar_width=4, speed_px_per_s=100,
                               direction='right', polarity=0,
                               temporal_jitter=0.0):
    dt = 1.0 / fps
    times = np.arange(0, duration_s, dt)
    events = []
    if direction == 'right':
        start_center_x = -bar_width//2
        vx = abs(speed_px_per_s)
    else:
        start_center_x = width + bar_width//2
        vx = -abs(speed_px_per_s)
    for t in times:
        center_x = start_center_x + vx * t
        xs = np.arange(int(np.floor(center_x - bar_width/2)), int(np.ceil(center_x + bar_width/2))+1)
        ys = np.arange(0, height)
        xs = xs[(xs >= 0) & (xs < width)]
        for x in xs:
            for y in ys:
                tt = float(t)
                if temporal_jitter > 0.0:
                    tt += np.random.normal(scale=temporal_jitter)
                    if tt < 0:
                        tt = 0.0
                events.append((int(x), int(y), float(tt), int(polarity)))
    return events

def events_to_spikegenerator_args(events, width, height):
    Npix = width * height
    indices = []
    times = []
    for (x,y,t,pol) in events:
        i = y * width + x + pol * Npix
        indices.append(int(i))
        times.append(float(t))
    indices = np.array(indices, dtype=np.int64)
    times = np.array(times) * second
    return indices, times

# ---------- Run a single trial in Brian2 ----------
def run_snn_trial(indices, times, width, height, sim_duration_s,
                  N_rec=128, conn_p=0.03, input_weight=0.03, recur_weight=0.02):
    start_scope()
    N_in = width * height * 2  # two polarity channels
    G_in = SpikeGeneratorGroup(N_in, indices=indices, times=times)
    # LIF
    eqs = "dv/dt = (-v) / (10*ms) : 1"
    G = NeuronGroup(N_rec, eqs, threshold='v>1', reset='v=0', method='exact')
    S = Synapses(G_in, G, on_pre='v += w_pre')
    S.connect(p=conn_p)
    S.w_pre = input_weight
    R = Synapses(G, G, on_pre='v += w_rec')
    R.connect(p=0.02)
    R.w_rec = recur_weight
    sm_in = SpikeMonitor(G_in)
    sm = SpikeMonitor(G)
    vm = StateMonitor(G, 'v', record=[0])
    run(sim_duration_s * second)
    in_spikes_t = np.array(sm_in.t / second)
    in_spikes_i = np.array(sm_in.i)
    out_spikes_t = np.array(sm.t / second)
    out_spikes_i = np.array(sm.i)
    counts = np.bincount(out_spikes_i, minlength=N_rec)
    return {
        'in_t': in_spikes_t, 'in_i': in_spikes_i,
        'out_t': out_spikes_t, 'out_i': out_spikes_i,
        'counts': counts.astype(float),
        'vm_t': np.array(vm.t/second), 'vm_v': vm.v[0].copy()
    }

# ---------- Build dataset ----------
def build_dataset(n_trials_per_class=40, width=32, height=32,
                  sim_duration_s=0.4, fps=200, bar_width=4, base_speed=100.0):
    X_counts = []
    labels = []
    trial_spike_data = []
    for cls_idx, direction in enumerate(['right','left']):
        for tnum in range(n_trials_per_class):
            speed = base_speed + np.random.normal(scale=8.0)
            events = generate_moving_bar_events(width=width, height=height, duration_s=sim_duration_s,
                                                fps=fps, bar_width=bar_width, speed_px_per_s=speed,
                                                direction=direction, polarity=0, temporal_jitter=0.002)
            indices, times = events_to_spikegenerator_args(events, width, height)
            sim = run_snn_trial(indices, times, width, height, sim_duration_s,
                                N_rec=128, conn_p=0.03, input_weight=0.03, recur_weight=0.02)
            X_counts.append(sim['counts'])
            labels.append(cls_idx)
            if tnum == 0:
                trial_spike_data.append({'direction':direction, 'sim':sim, 'width':width, 'height':height})
    X = np.vstack(X_counts)
    y = np.array(labels, dtype=int)
    return X, y, trial_spike_data

# ---------- Simple closed-form ridge ----------
def train_ridge(X_train, y_train, alpha=1.0):
    N, D = X_train.shape
    Xb = np.hstack([X_train, np.ones((N,1))])
    t = y_train.astype(float).reshape(-1,1)
    I = np.eye(D+1)
    I[-1,-1] = 0.0
    W = np.linalg.inv(Xb.T.dot(Xb) + alpha * I).dot(Xb.T).dot(t)
    w = W[:-1,0]
    b = float(W[-1,0])
    return w, b

def predict_counts(X, w, b):
    logits = X.dot(w) + b
    preds = (logits >= 0.5).astype(int)
    return preds, logits

# ---------- Latency computation ----------
def compute_latency_for_trial(sim_data, w, b, sim_duration_s, min_hold_s=0.03):
    dt = 0.001
    times = np.arange(0, sim_duration_s+1e-9, dt)
    N_rec = len(w)
    cum_counts = np.zeros((len(times), N_rec))
    out_t = sim_data['out_t']
    out_i = sim_data['out_i']
    for n in range(N_rec):
        spikes_n = out_t[out_i == n]
        inds = np.searchsorted(times, spikes_n, side='right') - 1
        if len(inds) > 0:
            for idx in inds:
                if 0 <= idx < len(times):
                    cum_counts[idx:, n] += 1.0
    logits_t = cum_counts.dot(w) + b
    preds_t = (logits_t >= 0.5).astype(int)
    final_pred = preds_t[-1]
    hold_bins = int(np.ceil(min_hold_s / dt))
    earliest_t = None
    for i in range(len(times)-hold_bins):
        if preds_t[i] == final_pred and np.all(preds_t[i:i+hold_bins] == final_pred):
            earliest_t = times[i]
            break
    if earliest_t is None:
        earliest_t = sim_duration_s
    return earliest_t, logits_t, times

# ---------- Main pipeline ----------
def main():
    print("Building dataset (this will run many small Brian2 sims)...")
    X, y, trial_spike_data = build_dataset(n_trials_per_class=40, width=32, height=32, sim_duration_s=0.4)
    print("Dataset built:", X.shape)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    w, b = train_ridge(X_train, y_train, alpha=1.0)
    preds_train, _ = predict_counts(X_train, w, b)
    preds_test, _ = predict_counts(X_test, w, b)
    train_acc = (preds_train == y_train).mean()
    test_acc = (preds_test == y_test).mean()
    print(f"Train acc: {train_acc*100:.2f}%  Test acc (counts readout): {test_acc*100:.2f}%")

    # latency: re-simulate test trials to compute time-to-decision
    print("Computing latency by re-simulating test trials...")
    latencies = []
    final_preds = []
    for i in range(len(X_test)):
        direction = 'right' if y_test[i] == 0 else 'left'
        events = generate_moving_bar_events(width=32, height=32, duration_s=0.4, fps=200,
                                            bar_width=4, speed_px_per_s=100.0 + np.random.normal(scale=8.0),
                                            direction=direction, polarity=0, temporal_jitter=0.002)
        indices, times = events_to_spikegenerator_args(events, 32, 32)
        sim = run_snn_trial(indices, times, 32, 32, 0.4, N_rec=128, conn_p=0.03, input_weight=0.03, recur_weight=0.02)
        latency, logits_t, times_grid = compute_latency_for_trial(sim, w, b, 0.4, min_hold_s=0.03)
        latencies.append(latency)
        final_pred = int((logits_t[-1] >= 0.5))
        final_preds.append(final_pred)
    latencies = np.array(latencies)
    final_preds = np.array(final_preds)
    test_acc_resim = (final_preds == y_test).mean()
    print(f"Re-simulated test acc (final): {test_acc_resim*100:.2f}%")
    print(f"Mean latency: {latencies.mean()*1000.0:.1f} ms  Median: {np.median(latencies)*1000.0:.1f} ms")

    # ---------- Plots ----------
    sample = trial_spike_data[0]['sim']
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(sample['in_t']*1000.0, sample['in_i'], '.')
    plt.xlabel('Time (ms)'); plt.ylabel('Input neuron index'); plt.title('Input spikes (sample)')
    plt.subplot(1,2,2)
    plt.plot(sample['out_t']*1000.0, sample['out_i'], '.')
    plt.xlabel('Time (ms)'); plt.ylabel('Hidden neuron index'); plt.title('Hidden spikes (sample)')
    plt.tight_layout()

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, preds_test)
    plt.figure(figsize=(4,3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['right','left'])
    disp.plot(values_format='d', cmap=None, colorbar=False)
    plt.title('Confusion matrix (test set)')

    plt.figure(figsize=(5,3))
    plt.hist(latencies*1000.0, bins=15)
    plt.xlabel('Latency (ms)'); plt.ylabel('Count'); plt.title('Latency distribution')

    avg_counts = X_train.mean(axis=0)
    plt.figure(figsize=(6,3))
    plt.imshow(avg_counts.reshape(1, -1), aspect='auto', origin='lower')
    plt.xlabel('Hidden neuron index'); plt.yticks([]); plt.title('Average spike count per hidden neuron (train)')

    plt.show()

if __name__ == "__main__":
    main()