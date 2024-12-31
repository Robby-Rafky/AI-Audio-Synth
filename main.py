import numpy as np
import scipy.io.wavfile as wav
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate, freqz
from scipy.signal import butter, filtfilt


# Read audio file and normalize the data
def read_audio(file_path):
    rate, data = wav.read(file_path)
    data = data / (2 ** 15) if data.dtype == np.int16 else data / (2 ** 31)
    return rate, data


# Estimate LPC coefficients using autocorrelation method
def autocorrelation_lpc(data, order):
    r = correlate(data, data, mode='full')
    r = r[len(r) // 2: len(r) // 2 + order + 1]
    R = toeplitz(r[:-1])
    r_vector = -r[1:]
    a = np.linalg.solve(R, r_vector)
    return np.concatenate(([1], a))


# Estimate LPC coefficients using Burg's method
def burg_method(data, order):
    a = np.zeros(order + 1)
    a[0] = 1.0
    ef = np.copy(data)
    eb = np.copy(data)

    for m in range(1, order + 1):
        # Calculate reflection coefficient
        num = -2.0 * np.dot(eb[m:], ef[:-m])
        den = np.dot(ef[:-m], ef[:-m]) + np.dot(eb[m:], eb[m:])
        k = num / den

        # Update LPC coefficients
        a_prev = np.copy(a)
        for i in range(1, m):
            a[i] = a_prev[i] + k * a_prev[m - i]
        a[m] = k

        # Update forward and backward errors
        ef_tmp = np.copy(ef)
        ef[:-m] += k * eb[m:]
        eb[m:] += k * ef_tmp[:-m]

    return a


# Wrapper function to estimate LPC coefficients using specified method
def estimate_lpc(data, order=16, method='burg'):
    if method == 'autocorr':
        lpc_coeffs = autocorrelation_lpc(data, order)
    elif method == 'burg':
        lpc_coeffs = burg_method(data, order)

    return lpc_coeffs


# Find formant frequencies from LPC coefficients
def find_formants(lpc_coeffs, rate):
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) >= 0]

    angles = np.angle(roots)
    freqs = sorted(angles * (rate / (2 * np.pi)))

    formant_freqs = []
    for freq in freqs:
        if freq > 90:
            formant_freqs.append(freq)

    return formant_freqs


# Estimate fundamental frequency using autocorrelation
def estimate_fundamental_frequency(data, rate):
    corr = correlate(data, data, mode='full')
    corr = corr[len(corr) // 2:]
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    fundamental_freq = rate / peak
    return fundamental_freq


# Generate impulse train for excitation signal
def generate_impulse_train(fundamental_freq, rate, duration=1, jitter=0.02):
    impulse_train = np.zeros(max(1, int(rate * duration)))
    period = int(rate / fundamental_freq)
    for i in range(0, len(impulse_train), period):
        jitter_offset = int(jitter * period * (np.random.rand() - 0.5))
        impulse_position = min(max(i + jitter_offset, 0), len(impulse_train) - 1)
        impulse_train[impulse_position] = 1
    return impulse_train


# Filter impulse train using LPC coefficients
def filter_impulse_train(impulse_train, lpc_coeffs):
    return lfilter([1], lpc_coeffs, impulse_train)


if __name__ == "__main__":
    # Constants
    WORD = "head_f"
    LPC_ORDER = 32
    LPC_METHOD = 'burg'
    WINDOW_SIZE = 20  # Hz
    LOWPASS_FREQ = 4000  # Hz

    # Define the input audio file
    audio_file = f"Speech/{WORD}.wav"

    # Read the audio file
    rate, waveform = read_audio(audio_file)
    print(f"\nSampling Rate: {rate} Hz")
    print(f"Waveform Length: {len(waveform)} samples\n")

    # Estimate LPC coefficients
    lpc_coeffs = estimate_lpc(waveform, order=LPC_ORDER, method=LPC_METHOD)
    print(f"Estimated LPC Coefficients (order {LPC_ORDER}, method [{LPC_METHOD}]):")
    print(lpc_coeffs)

    # Find formant frequencies from LPC coefficients
    formants = find_formants(lpc_coeffs, rate)[:3]  # Get the first three formants
    formants = [float(f) for f in formants]  # Convert np.float64 to regular float
    print(f"\nFormant Frequencies (Hz): {formants}\n")

    # Estimate fundamental frequency
    fundamental_freq = estimate_fundamental_frequency(waveform, rate)
    print(f"Fundamental Frequency (Hz): {fundamental_freq}")

    # Generate excitation signal using impulse train
    excitation_signal = generate_impulse_train(fundamental_freq, rate, duration=len(waveform) / rate)
    envelope = np.abs(waveform)

    # Apply low-pass filter to extract amplitude envelope
    b, a = butter(N=4, Wn=LOWPASS_FREQ / (rate / 2), btype='low')
    envelope = filtfilt(b, a, envelope)

    # Smooth the amplitude envelope
    window_size = int((WINDOW_SIZE / 1000) * rate)
    smoothed_envelope = np.convolve(envelope, np.ones(window_size) / window_size, mode='same')
    smoothed_envelope = smoothed_envelope / np.max(smoothed_envelope)

    # Modulate the excitation signal with the smoothed envelope
    excitation_signal = excitation_signal * smoothed_envelope
    excitation_signal = excitation_signal / np.max(np.abs(excitation_signal))

    # Filter the excitation signal using LPC coefficients
    filtered_impulse_train = filter_impulse_train(excitation_signal, lpc_coeffs)

    # Normalize the filtered signal
    filtered_impulse_train = filtered_impulse_train / np.max(np.abs(filtered_impulse_train))

    # Save the synthesized speech to a file
    output_path = f"Output/{WORD}_{LPC_METHOD}_filtered.wav"
    wav.write(output_path, rate, np.int16(filtered_impulse_train * 32767))

    # Plot the original waveform
    plt.figure(figsize=(10, 4))
    plt.plot(waveform, color='blue', lw=1)
    plt.title(f"[{WORD}] Original Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Plot the synthesized waveform
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_impulse_train, color='orange', lw=1)
    plt.title(f"[{WORD}] Synthesized Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Compute the amplitude spectrum of the original waveform
    fft_spectrum = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(len(fft_spectrum), 1 / rate)
    fft_magnitude = 20 * np.log10(np.abs(fft_spectrum[:len(fft_spectrum) // 2]))

    # Compute the LPC filter response
    w, h = freqz(1, lpc_coeffs, worN=8000, fs=rate)
    h_db = 20 * np.log10(abs(h))

    # Plot the amplitude specturm with the LPC and formants overlayed
    plt.figure(figsize=(10, 6))
    plt.plot(fft_freqs[:len(fft_freqs) // 2], fft_magnitude, color='blue', lw=1, label='Amplitude Spectrum')
    scaling_factor = np.mean(fft_magnitude) - np.mean(h_db)
    plt.plot(w, h_db + scaling_factor, color='red', lw=2, label='LPC Frequency Response')
    for formant in formants:
        plt.axvline(formant, color='magenta', linestyle='--', lw=1)
        plt.title(f"[{WORD}] Amplitude Spectrum of Speech Segment with LPC Filter Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the LPC filter response as a function of frequency
    plt.figure(figsize=(10, 6))
    plt.plot(w, abs(h), color='green', lw=2, label='LPC Filter Response')
    plt.title(f"[{WORD}] LPC Filter Response as a Function of Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.show()
