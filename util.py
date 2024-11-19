import numpy as np


def generate_bit_sequence(n):
    """Generates a sequence of `n` random bits."""
    return np.random.randint(2, size=n, dtype=np.uint8)


def generate_awgn(s, snr_db):
    """Generates complex AWGN with a given SNR, in dB, relative to the signal `s`."""
    snr = 10 ** (snr_db / 10)
    n = len(s)

    # signal power can alternatively be estimated as np.var(s) if the mean is near zero
    p_signal = np.mean(np.abs(s) ** 2)
    p_noise = p_signal / snr

    # samples from the normal distribution have a power level (variance) of 1. to achieve a variance
    # of `p_noise`, we must adjust the standard deviation of the distribution by scaling it by
    # `sqrt(p_noise)`. furthermore, the noise power should be distributed equally between the real
    # and imaginary portions. thus, we scale by `sqrt(p_noise / 2)`
    return np.sqrt(p_noise / 2) * (np.random.randn(n) + 1j * np.random.randn(n))


def generate_pulse_shaping_filter(samples_per_symbol, ntaps=101, rolloff=0.35):
    """Generates a root-raised cosine pulse shaping filter.

    Pulse shaping should occur after upsampling. `samples_per_symbol` is the upsampling rate. A
    higher upsampling rate produce a wider filter.

    `ntaps` controls the number of coefficients, and thus, the computational complexity of the
    filter. more taps will ensure the filter decays to zero. `ntaps` should be odd so that there is
    a center tap.

    `rolloff` is a value between 0 and 1, usually between 0.2 and 0.5 that controls the "sharpness"
    of the filter's edges. a "sharper" filter (smaller rolloff) will use less bandwidth. however,
    it will take longer to decay to zero in the time domain, and thus require more taps.

    The returned filter has a gain of `samples_per_symbol`.
    """
    beta = rolloff
    Ts = samples_per_symbol
    t = np.arange(-(ntaps // 2), ntaps // 2 + 1)
    h = np.zeros(ntaps)

    if beta == 0:
        h = np.sinc(t / Ts)
    else:
        # there are 3 cases depending on the value of t. get the indices where each case is true.
        t0 = np.where(np.isclose(t, 0))
        t1 = np.where(np.isclose(np.abs(t), Ts / (4 * beta)))
        t2 = np.setdiff1d(np.arange(ntaps), np.union1d(t0, t1))

        # see https://en.wikipedia.org/wiki/Root-raised-cosine_filter
        h[t0] = 1 + beta * (4 / np.pi - 1)
        h[t1] = (beta / np.sqrt(2)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
        )
        h[t2] = (
            np.sin(np.pi * t[t2] / Ts * (1 - beta))
            + 4 * beta * t[t2] / Ts * np.cos(np.pi * t[t2] / Ts * (1 + beta))
        ) / (np.pi * t[t2] / Ts * (1 - (4 * beta * t[t2] / Ts) ** 2))

    return h


def fft(s, fs=1.0, n=None):
    """Perform a `n`-point (default `len(s)`) Fast Fourier Transform of `s` at sample rate `fs`.

    Returns a tuple containing the frequency bins and the (magnitude) response.
    """
    bins = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    response = np.fft.fftshift(np.abs(np.fft.fft(s, n)))
    return bins, response


def firfilter(h, s):
    """Apply the FIR filter `h` to `s`."""
    return np.convolve(s, h, mode="same")


def upsample(s, n):
    """Upsample a signal `s` by a factor `n` by zero-stuffing."""
    upsampled = np.zeros(len(s) * n, dtype=s.dtype)
    upsampled[::n] = s

    return upsampled
