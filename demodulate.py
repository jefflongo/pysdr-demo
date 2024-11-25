import numpy as np


def quantize(constellation, sample):
    """Quantizes a complex point `sample` to a point in `constellation`."""
    return constellation[np.argmin(np.abs(constellation - sample))]


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
        errors.append(error)

        # update the estimated phase offset
        mu -= error * loop_gain

        # move the integer part of mu into i
        i += samples_per_symbol + int(mu)
        mu -= int(mu)

    return recovered_symbols, np.array(errors)


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
        errors.append(error)

        # update the estimated phase offset
        mu -= error * loop_gain

        # move the integer part of mu into i
        i += samples_per_symbol + int(mu)
        mu -= int(mu)

    return recovered_symbols, np.array(errors)


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
        errors.append(error)

        # update the estimated phase offset
        mu -= error * loop_gain

        # move the integer part of mu into i
        i += samples_per_symbol + int(mu)
        mu -= int(mu)

    return recovered_symbols, np.array(errors)
