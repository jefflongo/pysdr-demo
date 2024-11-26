import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

import demodulate
import modulate
import util

#### CONFIGURATION #################################################################################

# modulation configuration
N_SYMBOLS = 10000
MODULATION_ORDER = 1
MODULATE_QAM = False

# sampling configuration
SYMBOL_RATE_HZ = 10000
UPSAMPLE_RATE = 8
SAMPLE_RATE_HZ = SYMBOL_RATE_HZ * UPSAMPLE_RATE

# noise configuration
APPLY_NOISE = True
SNR_DB = 20
DELAY_SAMPLES = UPSAMPLE_RATE // 2
FREQUENCY_OFFSET_HZ = int(0.01 * SAMPLE_RATE_HZ)

# plotting configuration
PLOT_PULSE_SHAPING_FILTER = False
PLOT_PULSE_SHAPED_SYMBOLS = False
PLOT_CONSTELLATION = True

#### TRANSMITTER ###################################################################################

# generate bit sequence
bits = util.generate_bit_sequence(N_SYMBOLS * MODULATION_ORDER)

# modulate
constellation, symbols = (
    modulate.modulate_qam(bits, MODULATION_ORDER)
    if MODULATE_QAM
    else modulate.modulate_psk(bits, MODULATION_ORDER)
)

# upsample
symbols_upsampled = util.upsample(symbols, UPSAMPLE_RATE)

# pulse shape
h = util.generate_pulse_shaping_filter(UPSAMPLE_RATE)
symbols_pulse_shaped = util.firfilter(symbols_upsampled, h)

#### WIRELESS CHANNEL ##############################################################################

# delay
transmitted_signal = np.concatenate(
    [np.zeros(DELAY_SAMPLES, dtype=complex), symbols_pulse_shaped]
)

# apply noise
if APPLY_NOISE:
    noise = util.generate_awgn(transmitted_signal, SNR_DB)
    transmitted_signal = transmitted_signal + noise

# apply frequency offset
transmitted_signal = util.frequency_offset(
    transmitted_signal, SAMPLE_RATE_HZ, FREQUENCY_OFFSET_HZ
)

#### RECEIVER ######################################################################################

# apply matched filter
GAIN_CORRECTION = UPSAMPLE_RATE
received_signal = util.firfilter(transmitted_signal, h)
received_signal /= GAIN_CORRECTION

# coarse frequency correction
received_signal_with_frequency_correction = (
    demodulate.coarse_frequency_correction(
        received_signal, SAMPLE_RATE_HZ, constellation
    )
    if not MODULATE_QAM or MODULATION_ORDER <= 2
    else received_signal
)

# clock recovery
recovered_symbols, symbol_recovery_fb = demodulate.recover_symbols_mueller_muller(
    received_signal_with_frequency_correction, UPSAMPLE_RATE, constellation
)

# fine frequency correction
recovered_symbols, freq_recovery_fb = (
    demodulate.fine_frequency_correction(recovered_symbols, constellation)
    if not MODULATE_QAM or MODULATION_ORDER <= 2
    else (recovered_symbols, None)
)

recovered_bits = demodulate.demodulate(constellation, recovered_symbols)
bit_error_rate = np.sum(np.abs(bits - recovered_bits[-len(bits) :]))
print(f"Bit Error Rate: {bit_error_rate} / {len(bits)}")

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

    t = np.linspace(0, len(symbols) * SAMPLE_RATE_HZ, len(symbols_upsampled))

    color1, color2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
    ax1.set_title("Symbols (Upsampled)")
    ax1.set_xlabel("t")
    ax1.vlines(t, 0, np.real(symbols_upsampled), colors=color1, alpha=0.7)
    ax1.scatter(t, np.real(symbols_upsampled), label="I", marker=".", alpha=0.7)
    ax1.vlines(t, 0, np.imag(symbols_upsampled), colors=color2, alpha=0.7)
    ax1.scatter(t, np.imag(symbols_upsampled), label="Q", marker=".", alpha=0.7)
    ax1.legend()

    ax2.sharex(ax1)
    ax2.set_title("Symbols (Pulse Shaped)")
    ax2.set_xlabel("t")
    ax2.plot(t, np.real(symbols_pulse_shaped), label="I", alpha=0.7)
    ax2.plot(t, np.imag(symbols_pulse_shaped), label="Q", alpha=0.7)
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
    ax4.sharex(ax3)
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

    ax.plot(
        np.real(received_signal),
        np.imag(received_signal),
        ".",
        alpha=0.05,
        label="Received signal",
    )
    ax.plot(
        np.real(recovered_symbols),
        np.imag(recovered_symbols),
        ".",
        label="Recovered symbols",
    )
    ax.plot(
        np.real(constellation),
        np.imag(constellation),
        "ro",
        label="Optimal symbols",
    )
    ax.legend()
    for i, symbol in enumerate(constellation):
        ax.annotate(xy=(symbol.real, symbol.imag), text=f"{i:>0{MODULATION_ORDER}b}")

    plt.show()
