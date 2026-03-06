import matplotlib.pyplot as plt
from matplotlib import cm
from getdist import plots, MCSamples

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

def PlotCorner(data, true_params):
        true_values = true_params
        param_labels = [r'M_{\rm (h,min)}', r' N_{\rm ion}', r" R_{\rm mfp}"]
        names = ["M_{(h,min)}", "N_{ion}", "R_{mfp}"]

        samples1 = MCSamples(
            samples=data,
            names=names,
            labels=param_labels,
            # label='Power Spectrum_ANN',
            # settings={'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.5}
        )
        # samples2 = MCSamples(
        #     samples=data2.T,
        #     names=names,
        #     labels=param_labels,
        #     # label='Power Spectrum_BNN',
        #     # settings={'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.5}
        # )


        g = plots.get_subplot_plotter()
        g.settings.line_labels = False
        g.settings.scaling = False
        g.settings.axes_fontsize = 12
        g.settings.axes_labelsize = 12
        # g.settings.auto_add_legend = False  
        # g.settings.axes_labelsize = 18
        # g.settings.lab_fontsize = 18
        # g.settings.legend_fontsize = 25
        # g.settings.title_limit_fontsize = 16
        # g.settings.tick_labelsize = 14

        # Apply the settings to the plotter
        # g = plots.get_subplot_plotter(settings=custom_settings)

        # Make the triangle plot
        # g = plots.get_subplot_plotter()
        g.triangle_plot(
            [samples1],
            filled=True,
            contour_colors=["deepskyblue"],
            markers=true_params,
            marker_args={'color': 'red', 'markeredgewidth': 12},
        )

        # Set axis limits
        # g.subplots[0][0].set_xlim(320, 650)
        # g.subplots[1][1].set_xlim(120, 200)
        # g.subplots[2][2].set_xlim(10, 15)

        # g.add_legend(['Power Spectrum BNN', 'Bispectrum BNN'],legend_loc = (-0.5,2.5) ,fontsize=15)

        # Add a main title
        plt.suptitle(r'[$\rm M_{{h,min}}(10^8 M_\odot)$ = {:.2f}, $\rm N_{{ion}}$ = {:.2f}, $\rm R_{{mfp}} (Mpc)$ = {:.2f},]'
                     .format(*true_values), ha='center', fontsize=18, y=1.05)

        # plt.suptitle(r'$\rm 500$ dataset ',ha='center', fontsize=18, y=1.05)
        # plt.savefig("../Plots/test3_PS_vs_BS_500data.pdf", format="pdf",dpi=500,transparent=False, bbox_inches="tight",)
        plt.show()