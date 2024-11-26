import numpy as np

import util


def nary_to_binary(nary, n):
    """Converts the `n`-bit sequence `nary` to a binary sequence of length `n * len(nary)`."""
    return (
        ((np.array(nary).reshape(-1, 1) & (1 << np.arange(n - 1, -1, -1))) > 0)
        .astype(int)
        .flatten()
    )


def nearest_point_index(x, y):
    """Returns the indices for which the values in `y` are nearest the values in `x`."""
    return np.argmin(np.abs(y[:, None] - x), axis=0)


def demodulate(constellation, samples):
    """Demodulates complex samples `samples`."""
    constellation_indices = nearest_point_index(samples, constellation)
    order = int(np.log2(len(constellation)))
    result = nary_to_binary(constellation_indices, order)

    return result.item() if result.size == 1 else result


def quantize(constellation, samples):
    """Quantizes complex points `samples` to points in `constellation`."""
    constellation_indices = nearest_point_index(samples, constellation)
    result = constellation[constellation_indices]

    return result.item() if result.size == 1 else result


def coarse_frequency_correction(samples, fs, constellation):
    """Perform a coarse frequency correction on `samples` by removing the most prominent frequency
    from `samples` after repeated squaring.
    """
    squared_symbols = samples ** len(constellation)
    freqs, response = util.fft(squared_symbols, fs=fs, n=1024 * 1024)
    max_freq = freqs[np.argmax(response)]

    return util.frequency_offset(samples, fs, -max_freq / len(constellation))


def fine_frequency_correction(samples, constellation, ki=0.01, kp=0.1):
    """Perform a fine frequency correction on `samples` using a decision-directed Costas loop."""
    corrected_samples = np.empty_like(samples)
    errors = np.empty_like(samples, dtype=float)

    phase = 0
    integral = 0

    for i, sample in enumerate(samples):
        # adjust the input sample by the inverse of the estimated phase offset
        corrected_samples[i] = sample * np.exp(-1j * phase)

        # determine the phase error
        decision = quantize(constellation, corrected_samples[i])
        error = np.angle(corrected_samples[i] * np.conj(decision))

        # update the estimated phase offset
        integral += error
        errors[i] = ki * integral + kp * error

        phase += errors[i]
        phase = np.mod(phase, 2 * np.pi)

    return corrected_samples, errors


def recover_symbols_early_late(
    samples, samples_per_symbol, constellation=None, loop_gain=0.3
):
    """Recover symbols from `samples` oversampled at `samples_per_symbol` using Early-Late Gate.

    Forms a feedback loop to synchronize to any timing offset. The feedback strength can be adjusted
    by `loop_gain`. Pass `constellation` to perform decision-directed recovery.

    returns an array of symbol estimates and an array containing the symbol recovery feedback.
    """
    recovered_symbols = []
    errors = []

    # estimated phase offset in (fractional) samples
    mu = 0
    # sample index of current symbol estimate
    i = samples_per_symbol

    while i < len(samples) - samples_per_symbol // 2:
        # get the current symbol estimates for the early, prompt, and late samples
        early = samples[i - samples_per_symbol // 2]
        prompt = samples[i]
        late = samples[i + samples_per_symbol // 2]

        # take the symbol estimate at the current predicted location
        recovered_symbols.append(prompt)

        # optionally perform decision-directed recovery
        if constellation is not None:
            early = quantize(constellation, early)
            late = quantize(constellation, late)

        # compute error
        error_real = (np.real(early) - np.real(late)) * np.real(prompt)
        error_imag = (np.imag(early) - np.imag(late)) * np.imag(prompt)
        error = error_real + error_imag

        # update the estimated phase offset
        mu -= error * loop_gain
        errors.append(mu)

        # move the integer part of mu into i
        i += samples_per_symbol + int(mu)
        mu -= int(mu)

    return np.array(recovered_symbols), np.array(errors)


def recover_symbols_gardner(
    samples, samples_per_symbol, constellation=None, loop_gain=0.3
):
    """Recover symbols from `samples` oversampled at `samples_per_symbol` using Gardner TED.

    Forms a feedback loop to synchronize to any timing offset. The feedback strength can be adjusted
    by `loop_gain`. Pass `constellation` to perform decision-directed recovery.

    returns an array of symbol estimates and an array containing the symbol recovery feedback.
    """
    recovered_symbols = []
    errors = []

    # estimated phase offset in (fractional) samples
    mu = 0
    # sample index of current symbol estimate
    i = samples_per_symbol // 2

    while i < len(samples) - samples_per_symbol // 2:
        # get the current symbol estimates for the early, prompt, and late samples
        early = samples[i - samples_per_symbol // 2]
        prompt = samples[i]
        late = samples[i + samples_per_symbol // 2]

        # take the symbol estimate at the current predicted location
        recovered_symbols.append(early)

        # optionally perform decision-directed recovery
        if constellation is not None:
            early = quantize(constellation, early)
            late = quantize(constellation, late)

        # compute error
        error_real = (np.real(late) - np.real(early)) * np.real(prompt)
        error_imag = (np.imag(late) - np.imag(early)) * np.imag(prompt)
        error = error_real + error_imag

        # update the estimated phase offset
        mu -= error * loop_gain
        errors.append(mu)

        # move the integer part of mu into i
        i += samples_per_symbol + int(mu)
        mu -= int(mu)

    return np.array(recovered_symbols), np.array(errors)


def recover_symbols_mueller_muller(
    samples, samples_per_symbol, constellation, loop_gain=0.3
):
    """Recover symbols from `samples` oversampled at `samples_per_symbol` using Mueller & Muller
    TED.

    Forms a feedback loop to synchronize to any timing offset. The feedback strength can be adjusted
    by `loop_gain`.

    returns an array of symbol estimates and an array containing the symbol recovery feedback.
    """
    recovered_symbols = []
    errors = []

    # estimated phase offset in (fractional) samples
    mu = 0
    # sample index of current symbol estimate
    i = samples_per_symbol

    while i < len(samples):
        # get the current symbol estimates for the current and previous symbols
        prev_estimate = samples[i - samples_per_symbol]
        current_estimate = samples[i]

        # take the symbol estimate at the current predicted location
        recovered_symbols.append(current_estimate)

        # compute error and adjust timing offset
        current_decision = quantize(constellation, current_estimate)
        prev_decision = quantize(constellation, prev_estimate)

        # compute error
        error_real = np.real(current_decision) * np.real(prev_estimate) - np.real(
            prev_decision
        ) * np.real(current_estimate)
        error_imag = np.imag(current_decision) * np.imag(prev_estimate) - np.imag(
            prev_decision
        ) * np.imag(current_estimate)
        error = error_real + error_imag

        # update the estimated phase offset
        mu -= error * loop_gain
        errors.append(mu)

        # move the integer part of mu into i
        i += samples_per_symbol + int(mu)
        mu -= int(mu)

    return np.array(recovered_symbols), np.array(errors)
