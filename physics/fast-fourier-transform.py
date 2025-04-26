

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(np):
    # Generate a sample signal: sum of two sine waves
    def generate_signal(t):
        freq1 = 5  # Frequency of first sine wave (Hz)
        freq2 = 20 # Frequency of second sine wave (Hz)
        return np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    return (generate_signal,)


@app.cell
def _(generate_signal, np):
    # Parameters
    sampling_rate = 64  # Hz
    duration = 2         # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = generate_signal(t)
    return sampling_rate, signal, t


@app.cell
def _(np):
    def fft(x):
        N = len(x)
        if N <= 1:
            return x
        if N % 2 != 0:
            raise ValueError("Size of x must be a power of 2")
    
        # Recursive call
        even = fft(x[::2])
        odd = fft(x[1::2])
    
        # Combine
        T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
        return [even[k] + T[k] for k in range(N // 2)] + \
               [even[k] - T[k] for k in range(N // 2)]
    return (fft,)


@app.cell
def _(np):
    def fftfreq(N, d=1.0):
        freqs = np.zeros(N)
        for k in range(N):
            if k < N // 2:
                freqs[k] = k / (N * d)
            else:
                freqs[k] = (k - N) / (N * d)
        return freqs

    return (fftfreq,)


@app.cell
def _(fft, fftfreq, sampling_rate, signal):
    # Apply Fast Fourier Transform (FFT)
    # fft_result = np.fft.fft(signal)
    # fft_freq = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    fft_result = fft(signal)
    fft_freq = fftfreq(len(signal), d=1/sampling_rate)
    return fft_freq, fft_result


@app.cell
def _(fft_freq, fft_result, np):
    # Take only the positive frequencies
    positive_freqs = fft_freq > 0
    fft_magnitude = np.abs(fft_result)
    return fft_magnitude, positive_freqs


@app.cell
def _(fft_freq, fft_magnitude, plt, positive_freqs, signal, t):
    # Plotting
    plt.figure(figsize=(12, 6))

    # Original Signal
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Signal (Time Domain)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # FFT Magnitude Spectrum
    plt.subplot(2, 1, 2)
    plt.plot(fft_freq[positive_freqs], fft_magnitude[positive_freqs])
    plt.title('FFT of Signal (Frequency Domain)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
