import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

import util

n_symbols = 100
modulation_order = 1
modulate_qam = False
snr_db = 20

plot_constellation = True

bits = util.generate_bit_sequence(n_symbols * modulation_order)

constellation, symbols = (
    util.modulate_qam(bits, modulation_order)
    if modulate_qam
    else util.modulate_psk(bits, modulation_order)
)

noise = util.generate_awgn(symbols, snr_db)
symbols += noise

if plot_constellation:
    _, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.axis("equal")

    if modulate_qam:
        ax.grid(True)
        locator = MultipleLocator(np.sqrt(2) / (np.sqrt(1 << modulation_order) - 1))
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(locator)
    else:
        ax.add_patch(
            plt.Circle((0, 0), 1, color="k", linewidth=1, fill=False, alpha=0.3)
        )

    ax.plot(np.real(symbols), np.imag(symbols), ".")

    ax.plot(np.real(constellation), np.imag(constellation), "r.")
    for i, symbol in enumerate(constellation):
        ax.annotate(xy=(symbol.real, symbol.imag), text=f"{i:>0{modulation_order}b}")

    plt.show()
