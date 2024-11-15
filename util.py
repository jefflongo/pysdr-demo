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


def generate_gray_code(n):
    """Generates the first `n` elements of the gray code."""
    return np.array(list(map(lambda x: x ^ (x >> 1), np.arange(n))))


def binary_to_nary(bits, n):
    """Converts the binary sequence `bits` to an `n`-bit sequence of length `len(bits) / n`."""
    if len(bits) % n:
        raise ValueError("Bit sequence length must be a multiple of n")

    return bits.reshape(-1, n).dot(1 << np.arange(n - 1, -1, -1))


def modulate_psk(bits, order):
    """Modulates `bits` with PSK modulation of order `order`.
    i.e. BPSK for order=1, QPSK for order=2, 8-PSK for order=3.

    Returns a tuple containing the constellation and the complex symbols.
    The constellation is represented as a numpy array of complex coordinates where each coordinate's
    index represents its symbol value.
    """
    nbits = 1 << order

    # generate the phase angles for the given order of PSK modulation
    # i.e. this will be [0, 180] degrees for BPSK and [0, 90, 180, 270] degrees for QPSK.
    phase_angles = np.linspace(0, 2 * np.pi, nbits, endpoint=False)

    # generate the complex coordinates for the constellation
    # alternatively: constellation = np.cos(phase_angles) + 1j * np.sin(phase_angles)
    constellation = np.exp(1j * phase_angles)

    # reorder the constellation such that it is gray coded
    gray_code = generate_gray_code(nbits)
    constellation_gray_coded = np.empty_like(constellation)
    constellation_gray_coded[gray_code] = constellation

    # convert the bit array to an array of n-bit values, i.e. 1-bit for BPSK, 2-bit for QPSK
    bits_nary = binary_to_nary(bits, order)

    # index the constellation to convert the n-bit values to complex symbols
    symbols = constellation_gray_coded[bits_nary]

    return constellation_gray_coded, symbols


def modulate_qam(bits, order, expand_grid=False):
    """Modulates `bits` with QAM modulation of order `order`.
    i.e. 4-QAM for order=2, 16-QAM for order=4, 64-QAM for order=6. Only square QAMs are supported.

    Returns a tuple containing the constellation and the complex symbols.
    The constellation is represented as a numpy array of complex coordinates where each coordinate's
    index represents its symbol value.
    """
    nbits = 1 << order

    shapef = np.sqrt(nbits)
    if not shapef.is_integer():
        raise ValueError("Only square QAMs are supported")

    shape = int(shapef)

    # whether to pack the modulation within the unit circle or expand the grid for better spacing
    start, end = (
        (1 - shape, shape - 1) if expand_grid else (-np.sqrt(2) / 2, np.sqrt(2) / 2)
    )

    # generate the complex coordinates for the constellation
    grid = np.linspace(start, end, shape, endpoint=True)
    constellation = np.array([complex(real, imag) for real in grid for imag in grid])

    # due to the grid geometry of QAM, gray coding must traverse through the grid in a snaking
    # pattern. in other words, reverse every odd grouping of `shape` elements.
    gray_code = generate_gray_code(nbits)
    gray_code_snake = gray_code.reshape(-1, shape)
    gray_code_snake[1::2] = gray_code_snake[1::2, ::-1]
    gray_code_snake = gray_code_snake.flatten()

    # reorder the constellation such that it is gray coded
    constellation_gray_coded = np.empty_like(constellation)
    constellation_gray_coded[gray_code_snake] = constellation

    # convert the bit array to an array of n-bit values, i.e. 2-bit for 4-QAM, 4-bit for 16-QAM
    bits_nary = binary_to_nary(bits, order)

    # index the constellation to convert the n-bit values to complex symbols
    symbols = constellation_gray_coded[bits_nary]

    return constellation_gray_coded, symbols
