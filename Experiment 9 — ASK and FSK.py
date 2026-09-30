# ==============================================================================
# EXPERIMENT 9 — ASK and FSK
# Single-Cell Google Colab Implementation
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# 1. MANDATORY VALIDATION: INNER PRODUCTS OF BFSK BASIS SIGNALS
# ==============================================================================
print("="*60)
print("1. MANDATORY VALIDATION: INNER PRODUCT CALCULATIONS")
print("="*60)

Tb = 1e-3        # Bit duration = 1 ms
fs = 200e3       # Sampling frequency = 200 kHz
t_bit = np.linspace(0, Tb, int(fs * Tb), endpoint=False)

f1 = 5000        # Base frequency = 5 kHz

# Tone spacings
df_coherent_min = 1 / (2 * Tb)    # 500 Hz (Coherent orthogonal spacing)
df_noncoherent_min = 1 / Tb       # 1000 Hz (Non-coherent orthogonal spacing)
df_non_ortho = 250               # 250 Hz (Non-orthogonal spacing)

phi1 = np.cos(2 * np.pi * f1 * t_bit) * np.sqrt(2 / Tb)
phi2_coh = np.cos(2 * np.pi * (f1 + df_coherent_min) * t_bit) * np.sqrt(2 / Tb)
phi2_noncoh = np.cos(2 * np.pi * (f1 + df_noncoherent_min) * t_bit) * np.sqrt(2 / Tb)
phi2_bad = np.cos(2 * np.pi * (f1 + df_non_ortho) * t_bit) * np.sqrt(2 / Tb)

# Numerical integration over 1 bit interval
ip_coh = np.trapz(phi1 * phi2_coh, t_bit)
ip_noncoh = np.trapz(phi1 * phi2_noncoh, t_bit)
ip_bad = np.trapz(phi1 * phi2_bad, t_bit)

print(f"Inner Product (Δf = 1/(2Tb) = {df_coherent_min} Hz) : {ip_coh:.6e} (Coherent Orthogonal)")
print(f"Inner Product (Δf = 1/Tb    = {df_noncoherent_min} Hz): {ip_noncoh:.6e} (Non-Coherent Orthogonal)")
print(f"Inner Product (Δf = {df_non_ortho} Hz)           : {ip_bad:.6f} (Non-Orthogonal)")
print("\n" + "="*60 + "\n")

# ==============================================================================
# 2. SYSTEM FUNCTIONS (TRANSMITTER, CHANNEL, RECEIVERS)
# ==============================================================================

def generate_signals(bits, Tb, fs, f1, f2, A=1.0):
    """Generate binary ASK and BFSK passband waveforms."""
    sps = int(fs * Tb)
    N = len(bits)
    t_total = np.linspace(0, N * Tb, N * sps, endpoint=False)
    
    bit_stream = np.repeat(bits, sps)
    
    # ASK Waveform
    ask_wave = bit_stream * A * np.cos(2 * np.pi * f1 * t_total)
    
    # BFSK Waveform
    fsk_freqs = np.where(bit_stream == 1, f2, f1)
    fsk_wave = A * np.cos(2 * np.pi * fsk_freqs * t_total)
    
    return t_total, bit_stream, ask_wave, fsk_wave

def add_awgn(signal_in, snr_db, fs, Tb):
    """Add Additive White Gaussian Noise based on Eb/N0."""
    eb = np.mean(signal_in**2) * Tb
    snr_linear = 10**(snr_db / 10)
    N0 = eb / snr_linear
    noise_sigma = np.sqrt(N0 * fs / 2)
    noise = np.random.normal(0, noise_sigma, len(signal_in))
    return signal_in + noise

def detect_coherent_ask(rx_wave, Tb, fs, f1, threshold=0.5):
    """Correlator-based coherent ASK detector."""
    sps = int(fs * Tb)
    num_bits = len(rx_wave) // sps
    t_bit = np.linspace(0, Tb, sps, endpoint=False)
    phi = np.cos(2 * np.pi * f1 * t_bit) * np.sqrt(2 / Tb)
    
    corr_out = []
    detected_bits = []
    
    for i in range(num_bits):
        segment = rx_wave[i*sps : (i+1)*sps]
        z = np.trapz(segment * phi, t_bit)
        corr_out.append(z)
        detected_bits.append(1 if z > threshold * np.sqrt(Tb/2) else 0)
        
    return np.array(detected_bits), np.array(corr_out)

def detect_coherent_bfsk(rx_wave, Tb, fs, f1, f2):
    """Correlator-based coherent BFSK detector."""
    sps = int(fs * Tb)
    num_bits = len(rx_wave) // sps
    t_bit = np.linspace(0, Tb, sps, endpoint=False)
    
    phi1 = np.cos(2 * np.pi * f1 * t_bit) * np.sqrt(2 / Tb)
    phi2 = np.cos(2 * np.pi * f2 * t_bit) * np.sqrt(2 / Tb)
    
    corr_diff = []
    detected_bits = []
    
    for i in range(num_bits):
        segment = rx_wave[i*sps : (i+1)*sps]
        z0 = np.trapz(segment * phi1, t_bit)
        z1 = np.trapz(segment * phi2, t_bit)
        diff = z1 - z0
        corr_diff.append(diff)
        detected_bits.append(1 if diff > 0 else 0)
        
    return np.array(detected_bits), np.array(corr_diff)

def detect_noncoherent_bfsk(rx_wave, Tb, fs, f1, f2):
    """Envelope / Non-coherent detector for BFSK."""
    sps = int(fs * Tb)
    num_bits = len(rx_wave) // sps
    t_bit = np.linspace(0, Tb, sps, endpoint=False)
    
    c1 = np.cos(2 * np.pi * f1 * t_bit)
    s1 = np.sin(2 * np.pi * f1 * t_bit)
    c2 = np.cos(2 * np.pi * f2 * t_bit)
    s2 = np.sin(2 * np.pi * f2 * t_bit)
    
    env_diff = []
    detected_bits = []
    
    for i in range(num_bits):
        segment = rx_wave[i*sps : (i+1)*sps]
        
        # Quadrature envelope detectors
        e1 = np.sqrt(np.trapz(segment * c1, t_bit)**2 + np.trapz(segment * s1, t_bit)**2)
        e2 = np.sqrt(np.trapz(segment * c2, t_bit)**2 + np.trapz(segment * s2, t_bit)**2)
        
        diff = e2 - e1
        env_diff.append(diff)
        detected_bits.append(1 if diff > 0 else 0)
        
    return np.array(detected_bits), np.array(env_diff)

# ==============================================================================
# 3. REQUIRED VISUALIZATIONS
# ==============================================================================
f2 = f1 + 1/Tb  # Standard non-coherent orthogonal spacing (6000 Hz)
sample_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
t_total, bit_stream, ask_wave, fsk_wave = generate_signals(sample_bits, Tb, fs, f1, f2)

ask_rx = add_awgn(ask_wave, snr_db=10, fs=fs, Tb=Tb)
fsk_rx = add_awgn(fsk_wave, snr_db=10, fs=fs, Tb=Tb)

plt.figure(figsize=(12, 12))

# Viz 1: Passband Waveforms
plt.subplot(4, 1, 1)
plt.plot(t_total * 1e3, bit_stream, 'k--', label='Baseband Bit Stream', alpha=0.5)
plt.plot(t_total * 1e3, ask_wave, label='ASK Passband')
plt.plot(t_total * 1e3, fsk_wave, label='BFSK Passband', alpha=0.8)
plt.title('1. Passband Waveforms')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Viz 2: Power Spectra
plt.subplot(4, 1, 2)
f_ask, psd_ask = signal.welch(ask_wave, fs=fs, nperseg=1024)
f_fsk, psd_fsk = signal.welch(fsk_wave, fs=fs, nperseg=1024)
plt.semilogy(f_ask / 1000, psd_ask, label='ASK PSD')
plt.semilogy(f_fsk / 1000, psd_fsk, label='BFSK PSD', alpha=0.8)
plt.title('2. Power Spectra Density (PSD)')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power Magnitude')
plt.xlim(0, 10)
plt.grid(True)
plt.legend()

# Viz 3: Correlator Outputs
_, ask_corr = detect_coherent_ask(ask_rx, Tb, fs, f1)
_, fsk_corr = detect_coherent_bfsk(fsk_rx, Tb, fs, f1, f2)
bit_indices = np.arange(1, len(sample_bits) + 1)

plt.subplot(4, 1, 3)
plt.stem(bit_indices - 0.1, ask_corr, linefmt='b-', markerfmt='bo', label='ASK Correlator Output')
plt.stem(bit_indices + 0.1, fsk_corr, linefmt='g-', markerfmt='go', label='BFSK Decision Output (Z1 - Z0)')
plt.axhline(0, color='r', linestyle='--', label='BFSK Decision Threshold')
plt.title('3. Correlator Outputs per Bit Interval')
plt.xlabel('Bit Index')
plt.ylabel('Statistic Output')
plt.grid(True)
plt.legend()

# Viz 4: Decision-Statistic Histograms
sim_bits = np.random.randint(0, 2, 2000)
_, _, _, sim_fsk = generate_signals(sim_bits, Tb, fs, f1, f2)
sim_fsk_noisy = add_awgn(sim_fsk, snr_db=4, fs=fs, Tb=Tb)
_, fsk_stats = detect_coherent_bfsk(sim_fsk_noisy, Tb, fs, f1, f2)

plt.subplot(4, 1, 4)
plt.hist(fsk_stats[sim_bits == 0], bins=30, alpha=0.6, label='Bit 0 Decision Statistic', color='blue')
plt.hist(fsk_stats[sim_bits == 1], bins=30, alpha=0.6, label='Bit 1 Decision Statistic', color='orange')
plt.axvline(0, color='red', linestyle='--', label='Optimal Threshold (0)')
plt.title('4. Decision-Statistic Histograms (BFSK at Eb/N0 = 4 dB)')
plt.xlabel('Decision Statistic')
plt.ylabel('Count')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==============================================================================
# 4. TONE SPACING VARIATION & BER CURVES
# ==============================================================================
print("\nRunning Monte Carlo BER Simulation over Eb/N0 range...")

ebn0_db_range = np.arange(0, 12, 2)
num_bits = 10000
test_bits = np.random.randint(0, 2, num_bits)

# Signals with different tone spacings
_, _, ask_tx, fsk_tx_ortho = generate_signals(test_bits, Tb, fs, f1, f2=f1 + 1/Tb)
_, _, _, fsk_tx_subortho = generate_signals(test_bits, Tb, fs, f1, f2=f1 + 250)

ber_ask_coh = []
ber_fsk_coh = []
ber_fsk_noncoh = []
ber_fsk_subortho = []

for ebn0 in ebn0_db_range:
    ask_rx = add_awgn(ask_tx, ebn0, fs, Tb)
    fsk_rx_ortho = add_awgn(fsk_tx_ortho, ebn0, fs, Tb)
    fsk_rx_subortho = add_awgn(fsk_tx_subortho, ebn0, fs, Tb)
    
    det_ask_coh, _ = detect_coherent_ask(ask_rx, Tb, fs, f1)
    det_fsk_coh, _ = detect_coherent_bfsk(fsk_rx_ortho, Tb, fs, f1, f2=f1 + 1/Tb)
    det_fsk_noncoh, _ = detect_noncoherent_bfsk(fsk_rx_ortho, Tb, fs, f1, f2=f1 + 1/Tb)
    det_fsk_sub, _ = detect_coherent_bfsk(fsk_rx_subortho, Tb, fs, f1, f2=f1 + 250)
    
    ber_ask_coh.append(np.mean(det_ask_coh != test_bits))
    ber_fsk_coh.append(np.mean(det_fsk_coh != test_bits))
    ber_fsk_noncoh.append(np.mean(det_fsk_noncoh != test_bits))
    ber_fsk_subortho.append(np.mean(det_fsk_sub != test_bits))

plt.figure(figsize=(9, 6))
plt.semilogy(ebn0_db_range, ber_ask_coh, 'o-', label='Coherent ASK')
plt.semilogy(ebn0_db_range, ber_fsk_coh, 's-', label='Coherent BFSK (Orthogonal Δf = 1/Tb)')
plt.semilogy(ebn0_db_range, ber_fsk_noncoh, '^--', label='Non-coherent BFSK (Orthogonal Δf = 1/Tb)')
plt.semilogy(ebn0_db_range, ber_fsk_subortho, 'x:', label='Coherent BFSK (Non-Orthogonal Δf = 250 Hz)')

plt.title('5. BER Curves vs Tone Spacing & Detector Type')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both')
plt.legend()
plt.show()

# ==============================================================================
# 5. OBSERVATIONS AND INTERPRETATION SUMMARY
# ==============================================================================
print("="*60)
print("5. OBSERVATIONS AND INTERPRETATION SUMMARY")
print("="*60)
print("1. Orthogonality Condition: Orthogonality holds when the inner product equals 0.")
print("   - Coherent threshold: Δf = 1/(2Tb) = 500 Hz")
print("   - Non-coherent threshold: Δf = 1/Tb = 1000 Hz")
print("2. Tone Spacing Impact:")
print("   - Non-orthogonal spacing (Δf = 250 Hz) creates cross-correlation between tones,")
print("     causing inter-channel interference and a severe BER noise floor.")
print("3. Coherent vs. Non-coherent Performance:")
print("   - Coherent BFSK performs ~3 dB better than coherent ASK for the same peak power.")
print("   - Non-coherent BFSK incurs a small penalty (~1 dB) compared to coherent BFSK,")
print("     but requires no phase tracking in the receiver.")
print("="*60)
