##### Routine to compute Takens' embedding from amplitude data #####
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# FFT-based tau estimation
# -----------------------------
def dominant_frequency_fft(signal, dt=1.0):
    """
    Return the dominant nonzero frequency in the signal using FFT.
    Assumes uniform sampling with time step dt.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)   # remove DC component

    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.abs(np.fft.rfft(signal))

    # Ignore the zero-frequency (DC) bin
    if len(spectrum) <= 1:
        raise ValueError("Signal is too short for FFT frequency estimation.")

    peak_idx = np.argmax(spectrum[1:]) + 1
    f_dom = freqs[peak_idx]
    return f_dom, freqs, spectrum


def estimate_tau_from_fft(signal, dt=1.0, fraction_of_period=0.25):
    """
    Estimate tau from the dominant period.
    By default, tau = period / 4, a common Takens-style heuristic.
    """
    f_dom, freqs, spectrum = dominant_frequency_fft(signal, dt=dt)

    if f_dom <= 0:
        raise ValueError("Dominant frequency is non-positive.")

    period = 1.0 / f_dom
    tau_time = fraction_of_period * period
    tau_samples = int(np.round(tau_time / dt))

    return tau_samples, f_dom, period, freqs, spectrum


# -----------------------------
# Takens embedding
# -----------------------------
def takens_embedding(dat, tau, m):
    delay_vecs = []
    for i in range((m - 1) * tau, len(dat)):
        delay_vec = []
        for j in range(0, m * tau, tau):
            delay_vec.append(dat[i - j])
        delay_vecs.append(delay_vec)
    return np.array(delay_vecs)


# -----------------------------
# Read in data
# -----------------------------
data = np.loadtxt('amplitude.dat', dtype=float)

# If you know the sampling interval, put it here.
# If each sample is one unit apart, dt = 1.0 is fine.
dt = 1.0

# -----------------------------
# Hard-coded parameters
# -----------------------------
tau_hardcoded = 8
m = 7

# -----------------------------
# FFT-based tau estimate
# -----------------------------
tau_fft, f_dom, period, freqs, spectrum = estimate_tau_from_fft(
    data, dt=dt, fraction_of_period=0.25
)

print(f"Dominant frequency from FFT: {f_dom:.6f} cycles/unit time")
print(f"Dominant period: {period:.6f} time units")
print(f"Estimated tau from FFT: {tau_fft} samples")
print(f"Hard-coded tau: {tau_hardcoded} samples")

# -----------------------------
# Compare embeddings
# -----------------------------
emb_hard = takens_embedding(data, tau_hardcoded, m)
emb_fft = takens_embedding(data, tau_fft, m)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(emb_hard[:, 0], emb_hard[:, 2], '.', markersize=1)
axes[0].set_xlabel(r"$x(t)$")
axes[0].set_ylabel(r"$x(t-2\tau)$")
axes[0].set_title(f"Takens Reconstruction\nhard-coded tau = {tau_hardcoded}")

axes[1].plot(emb_fft[:, 0], emb_fft[:, 2], '.', markersize=1)
axes[1].set_xlabel(r"$x(t)$")
axes[1].set_ylabel(r"$x(t-2\tau)$")
axes[1].set_title(f"Takens Reconstruction\nFFT-estimated tau = {tau_fft}")

plt.tight_layout()
plt.show()

# -----------------------------
# Optional: plot FFT spectrum
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(freqs, spectrum)
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("FFT Spectrum")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()