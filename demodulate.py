import numpy as np


def quantize(constellation, sample):
    """Quantizes a complex point `sample` to a point in `constellation`."""
    return constellation[np.argmin[np.abs(constellation - sample)]]


def recover_symbols_early_late(samples, samples_per_symbol, loop_gain=0.1):
    """Recover symbols from `samples` oversampled at `samples_per_symbol` using Early-Late Gate.

    Forms a feedback loop to synchronize to any timing offset. The feedback strength can be adjusted
    by `loop_gain`.

    returns an array of symbol estimates.
    """
    recovered_symbols = []

    mu = 0
    i = samples_per_symbol
    while i < len(samples) - samples_per_symbol:
        # read sample at the predicted sample and adjacent samples spaced by half the symbol period
        prompt = samples[i]
        early = samples[i - samples_per_symbol // 2]
        late = samples[i + samples_per_symbol // 2]

        # take the symbol estimate at the current prediction
        recovered_symbols.append(prompt)

        # compute error and adjust timing offset
        error = np.real(late - early)
        mu += loop_gain * error
        i += samples_per_symbol + int(mu)

    return np.array(recovered_symbols)


def recover_symbols_gardner(samples, samples_per_symbol, loop_gain=0.1):
    """Recover symbols from `samples` oversampled at `samples_per_symbol` using Gardner TED.

    Forms a feedback loop to synchronize to any timing offset. The feedback strength can be adjusted
    by `loop_gain`.

    returns an array of symbol estimates.
    """
    recovered_symbols = []

    mu = 0
    i = samples_per_symbol
    while i < len(samples) - samples_per_symbol:
        # read sample at the predicted sample and adjacent samples spaced by half the symbol period
        prompt = samples[i]
        early = samples[i - samples_per_symbol // 2]
        late = samples[i + samples_per_symbol // 2]

        # take the symbol estimate at the current prediction
        recovered_symbols.append(prompt)

        # compute error and adjust timing offset
        error = np.real((late - early) * np.conj(prompt))
        mu += loop_gain * error
        i += samples_per_symbol + int(mu)

    return np.array(recovered_symbols)
