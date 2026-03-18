"""
step3_snn_pipeline_updated.py

Complete pipeline for synthetic moving-bar DVS -> Brian2 SNN -> time-binned readout -> evaluation + latency.

Run:
    python step3_snn_pipeline_updated.py

Dependencies:
    numpy, matplotlib, scikit-learn, brian2

Notes:
- This script is designed for local execution. The notebook environment used by ChatGPT
  may not have brian2 installed, so run locally in a virtualenv as needed.
- If you see many zero-spike trials, increase input_weight, conn_p, or tau (in eqs).
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import (
    start_scope,
    SpikeGeneratorGroup,
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    StateMonitor,
    run,
    ms,
    second,
    prefs,
)

# Ensure Brian2 uses numpy codegen target (more robust on many systems)
prefs.codegen.target = "numpy"

# sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------- Synthetic event generator ----------
def generate_moving_bar_events(
    width=32,
    height=32,
    duration_s=0.4,
    fps=200,
    bar_width=4,
    speed_px_per_s=100,
    direction="right",
    polarity=0,
    temporal_jitter=0.0,
):
    """
    Generate deterministic synthetic events for a vertical bright bar moving horizontally.
    Returns list of (x, y, t_seconds, polarity).
    """
    dt = 1.0 / fps
    times = np.arange(0, duration_s, dt)
    events = []
    if direction == "right":
        start_center_x = -bar_width // 2
        vx = abs(speed_px_per_s)
    else:
        start_center_x = width + bar_width // 2
        vx = -abs(speed_px_per_s)
    for t in times:
        center_x = start_center_x + vx * t
        xs = np.arange(
            int(np.floor(center_x - bar_width / 2)),
            int(np.ceil(center_x + bar_width / 2)) + 1,
        )
        ys = np.arange(0, height)
        xs = xs[(xs >= 0) & (xs < width)]
        for x in xs:
            for y in ys:
                tt = float(t)
                if temporal_jitter > 0.0:
                    tt += np.random.normal(scale=temporal_jitter)
                    tt = max(0.0, tt)
                events.append((int(x), int(y), float(tt), int(polarity)))
    return events


def events_to_spikegenerator_args(events, width, height, min_dt_s=0.0002):
    """
    Convert events -> SpikeGeneratorGroup args (indices, times).
    - Enforce minimal inter-spike gap for each neuron (to satisfy Brian2 resolution).
    - Return indices (np.array int) and times (brian2 seconds array).
    """
    Npix = width * height
    indices = []
    times = []
    for (x, y, t, pol) in events:
        i = y * width + x + pol * Npix
        indices.append(int(i))
        times.append(float(t))
    indices = np.array(indices, dtype=np.int64)
    times = np.array(times, dtype=float)

    # sort by time
    sort_order = np.argsort(times, kind="stable")
    indices = indices[sort_order]
    times = times[sort_order]

    # enforce min spacing per neuron
    neuron_last_time = {}
    for k in range(len(indices)):
        n = int(indices[k])
        t = float(times[k])
        if n in neuron_last_time:
            min_allowed = neuron_last_time[n] + min_dt_s
            if t < min_allowed:
                t = min_allowed
        neuron_last_time[n] = t
        times[k] = t

    # re-sort (nudging might reorder)
    sort_order2 = np.argsort(times, kind="stable")
    indices = indices[sort_order2]
    times = times[sort_order2] * second
    return indices, times


# ---------- SNN single-trial runner ----------
def run_snn_trial(
    indices,
    times,
    width,
    height,
    sim_duration_s,
    N_rec=128,
    conn_p=0.05,
    input_weight=0.06,
    recur_weight=0.03,
    tau_ms=10.0,
):
    """
    Run a single Brian2 simulation trial and return spike arrays and counts.
    Prints a debug line showing input/output spike counts (comment out if too verbose).
    """
    start_scope()
    N_in = width * height * 2  # two polarity channels
    # clip spike times to avoid out-of-range spikes
    valid_mask = (times / second) < sim_duration_s
    indices = indices[valid_mask]
    times = times[valid_mask]

    G_in = SpikeGeneratorGroup(N_in, indices=indices, times=times)
    eqs = f"dv/dt = (-v) / ({tau_ms}*ms) : 1"
    G = NeuronGroup(N_rec, eqs, threshold="v>1", reset="v=0", method="exact")

    # Input -> hidden
    S = Synapses(G_in, G, model="w : 1", on_pre="v += w")
    S.connect(p=conn_p)
    S.w = input_weight

    # Recurrent
    R = Synapses(G, G, model="w : 1", on_pre="v += w")
    R.connect(p=0.02)
    R.w = recur_weight

    sm_in = SpikeMonitor(G_in)
    sm = SpikeMonitor(G)
    vm = StateMonitor(G, "v", record=[0])  # record first neuron voltage for demo
    run(sim_duration_s * second)

    in_spikes_t = np.array(sm_in.t / second)
    in_spikes_i = np.array(sm_in.i)
    out_spikes_t = np.array(sm.t / second)
    out_spikes_i = np.array(sm.i)
    counts = np.bincount(out_spikes_i, minlength=N_rec)

    # debug print to detect sparse trials
    total_in = len(in_spikes_t)
    total_out = len(out_spikes_t)
    print(f"[debug] in_spikes={total_in} out_spikes={total_out} sum_hidden_counts={counts.sum():.0f}")

    return {
        "in_t": in_spikes_t,
        "in_i": in_spikes_i,
        "out_t": out_spikes_t,
        "out_i": out_spikes_i,
        "counts": counts.astype(float),
        "vm_t": np.array(vm.t / second),
        "vm_v": vm.v[0].copy(),
    }


# ---------- Time-binned feature extractor ----------
def extract_binned_hidden_counts(sim_out, N_rec, sim_duration_s, n_bins=8):
    """
    Convert sim_out (containing 'out_t' and 'out_i') into flattened (N_rec * n_bins,) features.
    Each neuron's spikes are binned into n_bins uniform time bins across sim_duration_s.
    """
    bins = np.linspace(0.0, sim_duration_s, n_bins + 1)
    features = np.zeros((N_rec, n_bins), dtype=float)
    out_t = sim_out["out_t"]
    out_i = sim_out["out_i"]
    if out_t.size == 0:
        return features.ravel()
    # find bin index for each spike
    bin_inds = np.searchsorted(bins, out_t, side="right") - 1
    bin_inds = np.clip(bin_inds, 0, n_bins - 1)
    for neuron, bidx in zip(out_i, bin_inds):
        if 0 <= neuron < N_rec:
            features[int(neuron), int(bidx)] += 1.0
    return features.ravel()


# ---------- Bin-based latency computation (using classifier) ----------
def compute_latency_binned(sim_data, clf, true_label, sim_duration_s, n_bins=8, min_hold_s=0.03):
    """
    Compute earliest time (in seconds) when the running binned readout equals true_label
    and holds for min_hold_s. Running readout is computed at bin granularity:
      - at bin k we use counts for bins 0..k and zeros for k+1..n_bins-1 (partial observation).
    Returns earliest_time, array_of_logits_per_bin, bins (bin edges).
    """
    bins = np.linspace(0.0, sim_duration_s, n_bins + 1)  # length n_bins+1
    bin_duration = bins[1] - bins[0]
    # compute per-neuron per-bin counts
    N_rec = int(clf.coef_.shape[1] // n_bins)
    # build a per-neuron-per-bin matrix
    per_bin_counts = np.zeros((N_rec, n_bins), dtype=float)
    out_t = sim_data["out_t"]
    out_i = sim_data["out_i"]
    if out_t.size > 0:
        bin_inds = np.searchsorted(bins, out_t, side="right") - 1
        bin_inds = np.clip(bin_inds, 0, n_bins - 1)
        for neuron, bidx in zip(out_i, bin_inds):
            if 0 <= neuron < N_rec:
                per_bin_counts[int(neuron), int(bidx)] += 1.0

    logits_per_bin = np.zeros(n_bins, dtype=float)
    preds_per_bin = np.zeros(n_bins, dtype=int)

    # For each bin k, construct a feature vector using counts for bins <= k and zeros elsewhere
    for k in range(n_bins):
        feat = np.zeros((N_rec, n_bins), dtype=float)
        if k >= 0:
            feat[:, : k + 1] = per_bin_counts[:, : k + 1]
        feat_flat = feat.ravel().reshape(1, -1)
        # Ensure classifier shape matches
        logits = clf.decision_function(feat_flat)  # decision function is linear for LR
        # decision_function returns shape (n_samples,) for binary
        logits_per_bin[k] = float(logits)
        preds_per_bin[k] = int((logits >= 0).astype(int))
    # Determine earliest bin index where pred == true_label and holds for min_hold_s
    hold_bins = max(1, int(np.ceil(min_hold_s / bin_duration)))
    earliest_bin = None
    for k in range(0, n_bins - hold_bins + 1):
        if preds_per_bin[k] == true_label and np.all(preds_per_bin[k : k + hold_bins] == true_label):
            earliest_bin = k
            break
    if earliest_bin is None:
        # not stable within trial
        earliest_time = sim_duration_s
    else:
        earliest_time = bins[earliest_bin]  # start time of that bin
    return earliest_time, logits_per_bin, bins


# ---------- Build dataset ----------
def build_dataset(
    n_trials_per_class=40,
    width=32,
    height=32,
    sim_duration_s=0.4,
    fps=200,
    bar_width=4,
    base_speed=100.0,
    N_rec=128,
    n_bins=8,
):
    """
    Run many trials (Brian2 sims) and return:
      - X: (n_trials, N_rec * n_bins) features (binned hidden counts)
      - y: labels (0=right, 1=left)
      - trial_spike_data: list of first trial per class (for plotting/debug)
    """
    X_features = []
    labels = []
    trial_spike_data = []
    for cls_idx, direction in enumerate(["right", "left"]):
        for tnum in range(n_trials_per_class):
            speed = base_speed + np.random.normal(scale=8.0)
            events = generate_moving_bar_events(
                width=width,
                height=height,
                duration_s=sim_duration_s,
                fps=fps,
                bar_width=bar_width,
                speed_px_per_s=speed,
                direction=direction,
                polarity=0,
                temporal_jitter=0.002,
            )
            indices, times = events_to_spikegenerator_args(events, width, height)
            sim = run_snn_trial(
                indices,
                times,
                width,
                height,
                sim_duration_s,
                N_rec=N_rec,
                conn_p=0.05,
                input_weight=0.06,
                recur_weight=0.03,
                tau_ms=15.0,
            )
            features = extract_binned_hidden_counts(sim, N_rec=N_rec, sim_duration_s=sim_duration_s, n_bins=n_bins)
            X_features.append(features)
            labels.append(cls_idx)
            if tnum == 0:
                trial_spike_data.append({"direction": direction, "sim": sim, "width": width, "height": height})
    X = np.vstack(X_features)
    y = np.array(labels, dtype=int)
    return X, y, trial_spike_data


# ---------- Train logistic readout ----------
def train_logistic(X_train, y_train, C=1.0, max_iter=1000):
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
    clf.fit(X_train, y_train)
    return clf


# ---------- Main pipeline ----------
def main():
    print("Building dataset and running SNN trials (this will take some time)...")
    N_rec = 128
    n_bins = 8
    X, y, trial_spike_data = build_dataset(
        n_trials_per_class=40, width=32, height=32, sim_duration_s=0.4, N_rec=N_rec, n_bins=n_bins
    )
    print("Dataset built:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print("Training logistic regression readout (time-binned features)...")
    clf = train_logistic(X_train, y_train, C=0.5, max_iter=1000)
    preds_train = clf.predict(X_train)
    preds_test = clf.predict(X_test)
    train_acc = (preds_train == y_train).mean()
    test_acc = (preds_test == y_test).mean()
    print(f"Train acc: {train_acc*100:.2f}%  Test acc (binned readout): {test_acc*100:.2f}%")

    # Re-simulate test trials and compute latency per trial using bin-based running evidence
    print("Computing latency by re-simulating test trials (bin-based measure)...")
    latencies = []
    final_preds = []
    for i in range(len(X_test)):
        # resimulate a fresh trial with similar variability
        direction = "right" if y_test[i] == 0 else "left"
        events = generate_moving_bar_events(
            width=32,
            height=32,
            duration_s=0.4,
            fps=200,
            bar_width=4,
            speed_px_per_s=100.0 + np.random.normal(scale=8.0),
            direction=direction,
            polarity=0,
            temporal_jitter=0.002,
        )
        indices, times = events_to_spikegenerator_args(events, 32, 32)
        sim = run_snn_trial(
            indices,
            times,
            32,
            32,
            0.4,
            N_rec=N_rec,
            conn_p=0.05,
            input_weight=0.06,
            recur_weight=0.03,
            tau_ms=15.0,
        )
        # compute latency in seconds (bin-based)
        latency_s, logits_per_bin, bins = compute_latency_binned(sim, clf, true_label=y_test[i], sim_duration_s=0.4, n_bins=n_bins, min_hold_s=0.03)
        latencies.append(latency_s)
        # final pred based on full observation (all bins)
        feat_full = extract_binned_hidden_counts(sim, N_rec=N_rec, sim_duration_s=0.4, n_bins=n_bins).reshape(1, -1)
        final_pred = int(clf.predict(feat_full)[0])
        final_preds.append(final_pred)

    latencies = np.array(latencies)
    final_preds = np.array(final_preds)
    test_acc_resim = (final_preds == y_test).mean()
    print(f"Re-simulated Test acc (final predicted using full trial): {test_acc_resim*100:.2f}%")
    print(
        f"Mean latency: {latencies.mean()*1000.0:.1f} ms  Median: {np.median(latencies)*1000.0:.1f} ms"
    )

    # ---- Visualizations ----
    # sample rasters (first stored trial)
    sample = trial_spike_data[0]["sim"]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if sample["in_t"].size > 0:
        plt.plot(sample["in_t"] * 1000.0, sample["in_i"], ".")
    plt.xlabel("Time (ms)")
    plt.ylabel("Input neuron index")
    plt.title("Input spikes (sample trial)")

    plt.subplot(1, 2, 2)
    if sample["out_t"].size > 0:
        plt.plot(sample["out_t"] * 1000.0, sample["out_i"], ".")
    plt.xlabel("Time (ms)")
    plt.ylabel("Hidden neuron index")
    plt.title("Hidden spikes (sample trial)")
    plt.tight_layout()

    # confusion matrix using classifier on X_test
    cm = confusion_matrix(y_test, preds_test)
    plt.figure(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["right", "left"])
    disp.plot(values_format="d", cmap=None, colorbar=False)
    plt.title("Confusion matrix (test set)")

    # latency histogram
    plt.figure(figsize=(5, 3))
    plt.hist(latencies * 1000.0, bins=15)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency distribution (re-simulated test set)")

    # average spike-count heatmap (train set aggregated per neuron across bins)
    avg_counts = X_train.mean(axis=0)  # shape (N_rec * n_bins,)
    plt.figure(figsize=(8, 3))
    plt.imshow(avg_counts.reshape(N_rec, n_bins), aspect="auto", origin="lower")
    plt.xlabel("Time bin index")
    plt.ylabel("Hidden neuron index")
    plt.title("Average spike count per neuron per time-bin (train set)")
    plt.colorbar(label="avg spike count")

    plt.show()

    # final summary printed
    print("SUMMARY:")
    print(f"  Train acc (readout): {train_acc*100:.2f}%")
    print(f"  Test acc  (readout): {test_acc*100:.2f}%")
    print(f"  Re-simulated test acc (final pred): {test_acc_resim*100:.2f}%")
    print(f"  Mean latency (ms): {latencies.mean()*1000.0:.1f}")
    print(f"  Median latency (ms): {np.median(latencies)*1000.0:.1f}")


if __name__ == "__main__":
    main()