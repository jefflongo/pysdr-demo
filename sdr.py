import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

import util

#### CONFIGURATION #################################################################################

# modulation configuration
N_SYMBOLS = 100
MODULATION_ORDER = 1
MODULATE_QAM = False

# sampling configuration
SYMBOL_RATE_HZ = 100
UPSAMPLE_RATE = 8
SAMPLE_RATE_HZ = SYMBOL_RATE_HZ * UPSAMPLE_RATE

# noise configuration
APPLY_NOISE = True
SNR_DB = 20

# plotting configuration
PLOT_PULSE_SHAPING_FILTER = True
PLOT_PULSE_SHAPED_SYMBOLS = True
PLOT_CONSTELLATION = True

#### TRANSMITTER ###################################################################################

# generate bit sequence
bits = util.generate_bit_sequence(N_SYMBOLS * MODULATION_ORDER)

# modulate
constellation, symbols = (
    util.modulate_qam(bits, MODULATION_ORDER)
    if MODULATE_QAM
    else util.modulate_psk(bits, MODULATION_ORDER)
)

# upsample
symbols_upsampled = util.upsample(symbols, UPSAMPLE_RATE)

# pulse shape
h = util.generate_pulse_shaping_filter(UPSAMPLE_RATE)
symbols_pulse_shaped = np.convolve(symbols_upsampled, h)

#### WIRELESS CHANNEL ##############################################################################

# apply noise
if APPLY_NOISE:
    noise = util.generate_awgn(symbols_pulse_shaped, SNR_DB)
    symbols_pulse_shaped_with_noise = symbols_pulse_shaped + noise
else:
    symbols_pulse_shaped_with_noise = symbols_pulse_shaped

# TODO: frequency/timing offset

#### RECEIVER ######################################################################################

#### PLOTS #########################################################################################

matplotlib.rcParams["figure.figsize"] = (12, 10)

if PLOT_PULSE_SHAPING_FILTER:
    fig, (ax1, ax2) = plt.subplots(2)

    t = np.arange(-(len(h) // 2), len(h) // 2 + 1)
    ax1.axhline(0, color="gray")
    ax1.set_title("Pulse Shaping Filter Impulse Response")
    ax1.set_xlabel("t / Ts")
    ax1.xaxis.set_major_locator(MultipleLocator(UPSAMPLE_RATE))
    ax1.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x / UPSAMPLE_RATE)}")
    )
    ax1.grid()
    ax1.plot(t, h, ".-")

    freqs, response = util.fft(h, fs=SAMPLE_RATE_HZ, n=1024)
    ax2.set_title("Pulse Shaping Filter Frequency Response")
    ax2.set_xlabel("f (Hz)")
    ax2.grid()
    ax2.plot(freqs, response)

    fig.tight_layout()
    plt.show()

if PLOT_PULSE_SHAPED_SYMBOLS:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    t = np.arange(len(symbols))
    ax1.set_title("Symbols")
    ax1.set_xlabel("t")
    ax1.step(t, np.real(symbols), label="I")
    ax1.step(t, np.imag(symbols), label="Q")
    ax1.legend()

    ax2.set_title("Symbols (Pulse Shaped)")
    ax2.set_xlabel("t")
    ax2.plot(np.real(symbols_pulse_shaped), label="I")
    ax2.plot(np.imag(symbols_pulse_shaped), label="Q")
    ax2.legend()

    symbol_freq, symbol_response = util.fft(
        symbols_upsampled, fs=SAMPLE_RATE_HZ, n=1024
    )
    ax3.set_title("Symbols (Frequency Domain)")
    ax3.set_xlabel("f (Hz)")
    ax3.grid()
    ax3.plot(symbol_freq, symbol_response)

    symbol_ps_freq, symbol_ps_response = util.fft(
        symbols_pulse_shaped, fs=SAMPLE_RATE_HZ, n=1024
    )
    ax4.set_title("Symbols (Pulse Shaped, Frequency Domain)")
    ax4.set_xlabel("f (Hz)")
    ax4.grid()
    ax4.plot(symbol_ps_freq, symbol_ps_response)

    fig.tight_layout()
    plt.show()

if PLOT_CONSTELLATION:
    ax = plt.subplot()
    ax.set_title("Symbol Constellation")
    ax.set_xlabel("I")
    ax.set_ylabel("Q", rotation=0)
    ax.set_box_aspect(1)
    ax.axis("equal")

    if MODULATE_QAM:
        ax.grid()
        locator = MultipleLocator(np.sqrt(2) / (np.sqrt(1 << MODULATION_ORDER) - 1))
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(locator)
    else:
        ax.add_patch(
            plt.Circle((0, 0), 1, color="k", linewidth=1, fill=False, alpha=0.3)
        )

    # TODO: plot received symbols
    ax.plot(np.real(constellation), np.imag(constellation), "r.")
    for i, symbol in enumerate(constellation):
        ax.annotate(xy=(symbol.real, symbol.imag), text=f"{i:>0{MODULATION_ORDER}b}")

    plt.show()
