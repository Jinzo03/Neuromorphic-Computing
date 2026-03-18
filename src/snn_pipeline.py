"""
step3_snn_pipeline_tuned.py

Tuned pipeline for synthetic moving-bar DVS -> Brian2 SNN -> time-binned readout -> evaluation + latency.
Includes diagnostics for failed re-simulated trials.

Run:
    python step3_snn_pipeline_tuned.py

Dependencies:
    numpy, matplotlib, scikit-learn, brian2

Notes:
- This script may take some time (many small Brian2 sims). For quick debugging lower
  n_trials_per_class in main() to 10 or 20.
- If you see many zero-spike trials, try increasing input_weight or conn_p further.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings

# Use numpy codegen target for Brian2 (robust)
prefs.codegen.target = "numpy"

# silence harmless sklearn convergence warnings in long runs (optional)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


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
    Convert events -> (indices, times) for Brian2 SpikeGeneratorGroup.
    Enforce minimal inter-spike gap per neuron to satisfy Brian2 resolution.
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
    conn_p=0.06,
    input_weight=0.08,
    recur_weight=0.03,
    tau_ms=20.0,
):
    """
    Run one Brian2 simulation and return spike arrays + counts.
    Debug print included to detect sparse trials.
    """
    start_scope()
    N_in = width * height * 2
    # clip spike times to [0, sim_duration_s)
    valid_mask = (times / second) < sim_duration_s
    indices = indices[valid_mask]
    times = times[valid_mask]

    G_in = SpikeGeneratorGroup(N_in, indices=indices, times=times)
    eqs = f"dv/dt = (-v) / ({tau_ms}*ms) : 1"
    G = NeuronGroup(N_rec, eqs, threshold="v>1", reset="v=0", method="exact")

    S = Synapses(G_in, G, model="w : 1", on_pre="v += w")
    S.connect(p=conn_p)
    S.w = input_weight

    R = Synapses(G, G, model="w : 1", on_pre="v += w")
    R.connect(p=0.02)
    R.w = recur_weight

    sm_in = SpikeMonitor(G_in)
    sm = SpikeMonitor(G)
    vm = StateMonitor(G, "v", record=[0])
    run(sim_duration_s * second)

    in_spikes_t = np.array(sm_in.t / second)
    in_spikes_i = np.array(sm_in.i)
    out_spikes_t = np.array(sm.t / second)
    out_spikes_i = np.array(sm.i)
    counts = np.bincount(out_spikes_i, minlength=N_rec)

    # debug
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
def extract_binned_hidden_counts(sim_out, N_rec, sim_duration_s, n_bins=12):
    bins = np.linspace(0.0, sim_duration_s, n_bins + 1)
    features = np.zeros((N_rec, n_bins), dtype=float)
    out_t = sim_out["out_t"]
    out_i = sim_out["out_i"]
    if out_t.size == 0:
        return features.ravel()
    bin_inds = np.searchsorted(bins, out_t, side="right") - 1
    bin_inds = np.clip(bin_inds, 0, n_bins - 1)
    for neuron, bidx in zip(out_i, bin_inds):
        if 0 <= neuron < N_rec:
            features[int(neuron), int(bidx)] += 1.0
    return features.ravel()


# ---------- Bin-based latency computation (safe scalar handling) ----------
def compute_latency_binned(sim_data, clf, true_label, sim_duration_s, n_bins=12, min_hold_s=0.03):
    bins = np.linspace(0.0, sim_duration_s, n_bins + 1)
    bin_duration = bins[1] - bins[0]
    D = clf.coef_.shape[1]
    N_rec = int(D // n_bins)

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

    for k in range(n_bins):
        feat = np.zeros((N_rec, n_bins), dtype=float)
        feat[:, : k + 1] = per_bin_counts[:, : k + 1]
        feat_flat = feat.ravel().reshape(1, -1)
        logits_arr = clf.decision_function(feat_flat)
        logits_val = float(logits_arr[0])  # extract scalar safely
        logits_per_bin[k] = logits_val
        preds_per_bin[k] = int(logits_val >= 0.0)

    hold_bins = max(1, int(np.ceil(min_hold_s / bin_duration)))
    earliest_bin = None
    for k in range(0, n_bins - hold_bins + 1):
        if preds_per_bin[k] == true_label and np.all(preds_per_bin[k : k + hold_bins] == true_label):
            earliest_bin = k
            break
    earliest_time = sim_duration_s if earliest_bin is None else float(bins[earliest_bin])
    return earliest_time, logits_per_bin, bins


# ---------- Build dataset (tuned) ----------
def build_dataset(
    n_trials_per_class=120,
    width=32,
    height=32,
    sim_duration_s=0.4,
    fps=200,
    bar_width=4,
    base_speed=100.0,
    N_rec=128,
    n_bins=12,
):
    X_features = []
    labels = []
    trial_spike_data = []
    for cls_idx, direction in enumerate(["right", "left"]):
        for tnum in range(n_trials_per_class):
            speed = base_speed + np.random.normal(scale=12.0)
            events = generate_moving_bar_events(
                width=width,
                height=height,
                duration_s=sim_duration_s,
                fps=fps,
                bar_width=bar_width,
                speed_px_per_s=speed,
                direction=direction,
                polarity=0,
                temporal_jitter=0.004,
            )
            indices, times = events_to_spikegenerator_args(events, width, height)
            sim = run_snn_trial(
                indices,
                times,
                width,
                height,
                sim_duration_s,
                N_rec=N_rec,
                conn_p=0.06,
                input_weight=0.08,
                recur_weight=0.03,
                tau_ms=20.0,
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
def train_logistic(X_train, y_train, C=0.1, max_iter=2000):
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
    clf.fit(X_train, y_train)
    return clf


# ---------- Diagnostics plotting for failed resims ----------
def plot_trial_debug(sim, clf, true_label, n_bins=12, sim_duration_s=0.4):
    latency_s, logits_per_bin, bins = compute_latency_binned(sim, clf, true_label=true_label, sim_duration_s=sim_duration_s, n_bins=n_bins)
    plt.figure(figsize=(9,4))
    plt.subplot(1,3,1)
    if sim["in_t"].size > 0:
        plt.plot(sim["in_t"] * 1000.0, sim["in_i"], ".")
    plt.title("input spikes"); plt.xlabel("ms")
    plt.subplot(1,3,2)
    if sim["out_t"].size > 0:
        plt.plot(sim["out_t"] * 1000.0, sim["out_i"], ".")
    plt.title("hidden spikes"); plt.xlabel("ms")
    plt.subplot(1,3,3)
    plt.plot((bins[:-1]) * 1000.0, logits_per_bin, "-o")
    plt.axhline(0, color="k", linestyle="--")
    plt.title(f"logits per bin (latency={latency_s*1000.0:.0f} ms)")
    plt.tight_layout()
    plt.show()


# ---------- Main pipeline ----------
def main():
    print("Building dataset and running SNN trials (this will take a while)...")
    N_rec = 128
    n_bins = 12
    X, y, trial_spike_data = build_dataset(
        n_trials_per_class=120, width=32, height=32, sim_duration_s=0.4, N_rec=N_rec, n_bins=n_bins
    )
    print("Dataset built:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print("Training logistic regression readout (time-binned features)...")
    clf = train_logistic(X_train, y_train, C=0.1, max_iter=2000)
    preds_train = clf.predict(X_train)
    preds_test = clf.predict(X_test)
    train_acc = (preds_train == y_train).mean()
    test_acc = (preds_test == y_test).mean()
    print(f"Train acc: {train_acc*100:.2f}%  Test acc (binned readout): {test_acc*100:.2f}%")

    # Re-simulate test trials and compute latency per trial using bin-based running evidence
    print("Computing latency by re-simulating test trials (bin-based measure)...")
    latencies = []
    final_preds = []
    sims_for_test = []
    for i in range(len(X_test)):
        direction = "right" if y_test[i] == 0 else "left"
        events = generate_moving_bar_events(
            width=32,
            height=32,
            duration_s=0.4,
            fps=200,
            bar_width=4,
            speed_px_per_s=100.0 + np.random.normal(scale=12.0),
            direction=direction,
            polarity=0,
            temporal_jitter=0.004,
        )
        indices, times = events_to_spikegenerator_args(events, 32, 32)
        sim = run_snn_trial(
            indices,
            times,
            32,
            32,
            0.4,
            N_rec=N_rec,
            conn_p=0.06,
            input_weight=0.08,
            recur_weight=0.03,
            tau_ms=20.0,
        )
        sims_for_test.append(sim)
        latency_s, logits_per_bin, bins = compute_latency_binned(sim, clf, true_label=y_test[i], sim_duration_s=0.4, n_bins=n_bins, min_hold_s=0.03)
        latencies.append(latency_s)
        feat_full = extract_binned_hidden_counts(sim, N_rec=N_rec, sim_duration_s=0.4, n_bins=n_bins).reshape(1, -1)
        final_pred = int(clf.predict(feat_full)[0])
        final_preds.append(final_pred)

    latencies = np.array(latencies)
    final_preds = np.array(final_preds)
    test_acc_resim = (final_preds == y_test).mean()
    print(f"Re-simulated Test acc (final predicted using full trial): {test_acc_resim*100:.2f}%")
    print(f"Mean latency: {latencies.mean()*1000.0:.1f} ms  Median: {np.median(latencies)*1000.0:.1f} ms")

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

    # explicit confusion matrix plot for stored-test
    cm = confusion_matrix(y_test, preds_test)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix (stored-test)")
    plt.colorbar()
    classes = ["right", "left"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), "d"), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # confusion matrix for resimulated preds
    cm_resim = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm_resim, interpolation="nearest", cmap="Oranges")
    plt.title("Confusion matrix (resimulated-test)")
    plt.colorbar()
    thresh = cm_resim.max() / 2.0 if cm_resim.max() > 0 else 1.0
    for i in range(cm_resim.shape[0]):
        for j in range(cm_resim.shape[1]):
            plt.text(j, i, format(int(cm_resim[i, j]), "d"), horizontalalignment="center",
                     color="white" if cm_resim[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # latency histogram
    plt.figure(figsize=(5, 3))
    plt.hist(latencies * 1000.0, bins=20)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency distribution (re-simulated test set)")

    # average spike-count heatmap (train set aggregated per neuron across bins)
    avg_counts = X_train.mean(axis=0)  # shape (N_rec * n_bins,)
    plt.figure(figsize=(8, 4))
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

    # ----- Diagnostics: inspect failing resim trials -----
    failed_idx = np.where(final_preds != y_test)[0]
    print("\nDIAGNOSTICS:")
    print("Resim confusion matrix:\n", cm_resim)
    print("Resim preds distribution:", np.bincount(final_preds))
    print("True distribution:", np.bincount(y_test))
    print(f"Number of failed resim trials: {len(failed_idx)} / {len(y_test)}")
    if len(failed_idx) > 0:
        print("Failed indices (first 10):", failed_idx[:10])
        # show up to 4 failing trial visualizations
        for idx in failed_idx[:4]:
            print(f"\nPlotting failed test idx={idx}, true={y_test[idx]}")
            plot_trial_debug(sims_for_test[idx], clf, true_label=y_test[idx], n_bins=n_bins, sim_duration_s=0.4)


if __name__ == "__main__":
    main()