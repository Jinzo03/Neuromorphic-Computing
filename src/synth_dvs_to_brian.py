import numpy as np
import matplotlib.pyplot as plt
from brian2 import SpikeGeneratorGroup, SpikeMonitor, ms, second, run, start_scope
from brian2 import NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor, StateMonitor, start_scope, run

def generate_moving_bar_events(width=32, height=32,
                               duration_s=0.5,
                               fps=200, 
                               bar_width=4,
                               speed_px_per_s=80,
                               direction='right',
                               polarity=0):
    """
    Generate simple synthetic events representing a bright vertical bar moving horizontally.
    Returns a list of events as tuples (x, y, t_seconds, polarity).
    """
    if direction not in ('right', 'left'):
        raise ValueError("Direction must be 'right' or 'left'")
    dt = 1.0 / fps
    times = np.arange(0, duration_s, dt)
    events = []
    if direction == 'right':
        start_center_x = -bar_width // 2
        vx = abs(speed_px_per_s)
    else:
        start_center_x = width + bar_width // 2
        vx = -abs(speed_px_per_s)

    for t in times:
        center_x = start_center_x + vx * t
        xs = np.arange(int(np.floor(center_x - bar_width / 2)), int(np.ceil(center_x + bar_width / 2)))
        ys = np.arange(0, height)
        xs = xs[(xs >= 0) & (xs < width)]
        for x in xs:
            for y in ys:
                events.append((int(x), int(y), float(t), int(polarity)))
    return events


def events_to_spikegenerator_args(events, width, height):
    """
    Convert events (x,y,t,pol) to SpikeGeneratorGroup arrays.
    Polarity channel mapping: index += polarity * (width*height)
    """
    Npix = width * height
    indices = []
    times = []
    for (x, y, t, pol) in events:
        i = y * width + x + pol * Npix
        indices.append(int(i))
        times.append(float(t))
    indices = np.array(indices, dtype=np.int64)
    times = np.array(times) * second
    return indices, times

    
def plot_event_frame(events, width, height, title="Event sample"):
    counts = np.zeros((height, width), dtype=int)
    for (x, y, t, pol) in events:
        counts[y, x] += 1
    plt.figure(figsize=(4, 4))
    plt.imshow(counts, origin='lower', cmap='hot', vmin=0)  # FIX: anchor colormap at 0
    plt.title(title)
    plt.colorbar(label='event count')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    W, H = 32, 32
    N_in = int(W * H * 2)
    events_right = generate_moving_bar_events(width=W, height=H, duration_s=0.5,
                                              fps=200, bar_width=4, speed_px_per_s=120,
                                              direction='right', polarity=0)
    events_left = generate_moving_bar_events(width=W, height=H, duration_s=0.5,
                                             fps=200, bar_width=4, speed_px_per_s=120,
                                             direction='left', polarity=0)
    print("Right events:", len(events_right), "Left events:", len(events_left))
    plot_event_frame(events_right, W, H, title="Right motion events")

    idxs, times = events_to_spikegenerator_args(events_right, W, H)



