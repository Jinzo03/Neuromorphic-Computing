# Neuromorphic Computing: Spiking Neural Network Simulation

A reproducible project that simulates a **spiking neural network (SNN)** for **event-based motion direction classification** using **Brian2**.

The notebook and scripts in this repository generate synthetic Dynamic Vision Sensor (DVS)-style events, feed them into a leaky integrate-and-fire (LIF) spiking network, and train a time-binned logistic regression readout to classify **right vs left motion**.

---

## Project summary

This project explores the core ideas of **neuromorphic computing**:

- **Spiking Neural Networks (SNNs)** instead of standard frame-based neural networks
- **Event-based sensors** instead of regular image frames
- **Time-driven response** and early decision-making based on spike timing

The final version of the notebook achieved:

- **Dataset shape:** `(240, 2048)`
- **Class balance:** `120 right / 120 left`
- **Train accuracy:** `100%`
- **Test accuracy:** `100%`
- **Re-simulation test accuracy:** `100%`
- **Mean latency:** `14.7 ms`
- **Median latency:** `14.0 ms`

These results show that the model can classify the motion direction correctly and make very fast decisions from event-driven input.

---

## Repository contents

Typical files in this project:

- `neuromorphic_snn_v6.ipynb` — final notebook with the complete pipeline
- `results_v6/` — saved figures and exported metrics
- `README.md` — this documentation
- `requirements.txt` — dependencies

Example output files saved in `results_v6/`:

- `confusion_matrices.png`
- `event_maps.png`
- `latency.png`
- `membrane_voltages.png`
- `per_trial_metrics.csv`
- `rasters.png`
- `spike_count_heatmaps.png`

---

## Main idea

The project follows this pipeline:

1. **Generate synthetic DVS-style events**
   - A vertical bar moves across a 32×32 sensor.
   - Each event is represented as `(x, y, t, polarity)`.

2. **Convert events into input spikes**
   - Each pixel is mapped to an input neuron.
   - Events are fed into Brian2 using `SpikeGeneratorGroup`.

3. **Simulate a spiking neural network**
   - A recurrent **leaky integrate-and-fire (LIF)** hidden layer processes the spikes.
   - Input spikes and hidden spikes are recorded.

4. **Extract time-binned features**
   - Hidden spikes are counted in time bins.
   - These counts preserve temporal information.

5. **Train a readout classifier**
   - A logistic regression model classifies the motion direction.

6. **Measure latency**
   - The code computes the earliest stable decision time.
   - This shows how quickly the SNN can respond to event-based input.

---

## Why this project is interesting

This project demonstrates the main advantages of neuromorphic computing:

- **Low latency**: decisions are made quickly from sparse events
- **Temporal processing**: the model reacts to spike timing, not just static images
- **Biological inspiration**: the network uses spiking units and membrane integration
- **Efficiency-oriented design**: event-driven input is naturally sparse

---

## Requirements

- Python 3.9 or newer
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `brian2`
- `jupyter` (for the notebook)
- `tqdm` (for progress bars)
---

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv snn-env
```

Activate it:

```bash
# Linux / macOS
source snn-env/bin/activate

# Windows PowerShell
# .\snn-env\Scripts\Activate.ps1
```

Install packages:

```bash
pip install --upgrade pip
pip install numpy matplotlib pandas scikit-learn brian2 jupyter
```

---

## How to run

### Run the notebook
Open the notebook and run all cells in order:

```bash
jupyter notebook neuromorphic_snn_v6.ipynb
```

The script/notebook will:

- generate the synthetic dataset,
- simulate the SNN,
- train the readout,
- evaluate accuracy,
- compute latency,
- save plots and a CSV summary into `results_v6/`.

---

## Final results

Final notebook results:

- **Dataset shape:** `(240, 2048)`
- **Class balance:** `120 right / 120 left`
- **Train acc:** `100.00%`
- **Test acc:** `100.00%`
- **Re-sim test acc:** `100.00%`
- **Mean latency:** `14.7 ms`
- **Median latency:** `14.0 ms`
- **Std latency:** `4.1 ms`

Per class latency:

- **Right**
  - Mean: `10.8 ms`
  - Median: `10.7 ms`
  - Std: `0.9 ms`

- **Left**
  - Mean: `18.6 ms`
  - Median: `18.9 ms`
  - Std: `1.8 ms`

---

## What the code is doing

### Input generation
A moving bar is converted into event spikes. This simulates the behavior of an event camera, where only changes in the scene generate outputs.

### SNN simulation
The Brian2 network uses:

- `SpikeGeneratorGroup` for event input
- `NeuronGroup` for hidden LIF neurons
- `Synapses` for sparse connections
- `SpikeMonitor` for spike recording
- `StateMonitor` for membrane voltage traces

### Readout
Instead of using raw total spikes only, the notebook splits hidden spikes into time bins and uses those binned counts as features for the classifier.

### Latency measurement
The notebook checks when the classifier first makes a stable correct decision, which is a direct way to measure response speed.

---

## Results interpretation

These final results mean:

- the synthetic motion task is clearly separable,
- the hidden spiking layer captures the temporal structure of the input,
- the readout can classify the direction with no errors on the final setup,
- the system makes decisions very quickly, which is consistent with the neuromorphic computing idea.

---

## Future work

Possible next steps include:

- testing on real event-based datasets such as **N-MNIST** or **DVS128 Gesture**
- replacing the synthetic generator with a real event-camera loader
- using STDP or surrogate-gradient learning for the SNN itself
- comparing against a standard frame-based ANN baseline
- mapping the model to neuromorphic hardware for energy measurements

---


## License

- MIT
- Apache 2.0
- no license if the repository is private

---

## Acknowledgements

This project was developed as part of a student neuromorphic computing assignment and uses:

- **Brian2** for SNN simulation
- **scikit-learn** for classification
- **matplotlib** for plots
- **numpy** for numerical computation
- **pandas** for result export
