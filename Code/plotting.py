import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use( './Code/presentation_plot.mplstyle')



def PlotPs(k_values, delta_PS, xHI, num):
    """
    Plots the N number of power spectrum for different neutral fraction values.

    input:
    k_values: array of k values (h/Mpc)
    delta_PS: array of the power spectrum values for each xHI
    xHI: array of neutral fraction values corresponding to each power spectrum
    num: number of power spectra to plot

    """

    fig, ax = plt.subplots(figsize=(6, 4.5))

    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=min(xHI), vmax=max(xHI))

    for i in range(num):
        ax.plot(k_values, delta_PS[i],
                color=cmap(norm(xHI[i])),
                label=f'xHI={xHI[i]}',
                alpha=0.8,
                linewidth=1.5,
                marker='o',
                markersize=4
                )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, label=r'Neutral Fraction $\rm x_{{HI}}$', pad=0.03,)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel(r'$\Delta^2(k)$ [mK$^2$]')

    # plt.legend(fontsize=8, ncol=2)
    plt.show()