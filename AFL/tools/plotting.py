"""
Helper functions for plotting AFL entropy
"""

import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

def subplot_array(entropy, shape, inter_labels, id_entropy=None, plot_time=-1, intra_labels=None, suptitle=None, figsize=(12,8), dpi=90):
    """
    Plots a set of entropies in each subplot.

    Arguments
    entropy[i,j,k]: The AFL entropies to plot.
        i indexes the subplot
        j indexes the intraplot entropy lines
        k indexes the timestep
    shape: (x,y) tuple for the dimensions of the subplot array
    inter_labels: list of subplot titles

    Optional Keyword Arguments
    id_entropy (1D array): AFL entropy of identity dynamics.
        If provided, plots in the upper left only.
    plot_time: maximum time to plot to
        By default, we plot the entirety of entropy
    intra_labels: labels for intraplot entropies
        If provided a 1D array, this plots one legend in the upper left subplot
        (NOT YET SUPPORTED) If provided a 2D array, this plots a legend in each subplot
    suptitle: suptitle for the whole plot
    figsize: figsize tuple to be passed to matplotlib
    dpi: dpi to be passed to matplotlib
    """
    vnum = shape[1]
    hnum = shape[0]

    interplots = entropy.shape[0]
    intraplots = entropy.shape[1]

    # Argument checks
    if vnum * hnum != interplots:
        warn('Requested {0} plots but data implies {1} plots.'.format(vnum*hnum, interplots))

    assert inter_labels.size == interplots,\
        'Expected {0} interplot labels but received {1}.'.format(interplots, inter_labels.size)
    
    assert (intra_labels is None) or (intra_labels.size == intraplots),\
        'Expected {0} intraplot labels but received {1}.'.format(intraplots, intra_labels.size)
    
    if plot_time > 0:
        assert plot_time <= entropy.shape[2], 'plot_time exceeds the time interval of the data'
        entropy = entropy[:,:,:plot_time]
    times = np.arange(entropy.shape[2]) + 1

    # Plotting
    fig, axes = plt.subplots(vnum, hnum, figsize=figsize, dpi=dpi, sharey=True)
    
    for i in np.arange(min(vnum*hnum, interplots)):
        ax = axes[i//hnum, i%hnum]
        ax.grid()
        ax.set_title(inter_labels[i])
        for j in np.arange(intraplots):
            line, = ax.plot(times, entropy[i,j,:])
            if intra_labels is not None:
                line.set(label=intra_labels[j])

    if id_entropy is not None:
        id_time = min(id_entropy.size, times[-1])
        axes[0,0].plot(times[:id_time], id_entropy[:id_time], 'k--', label='identity')
    
    for i in np.arange(hnum):
        axes[-1,i].set_xlabel('Time Step')
    for i in np.arange(vnum):
        axes[i,0].set_ylabel('AFL Entropy')
    
    if intra_labels is not None:
        axes[0,0].legend(framealpha=0.75)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20)

    return fig

def growth_plot(entropy, bounds, trans_time, show_bounds=False, id_entropy=None, labels=None, \
                LR_labels=None, true_agc=None, suptitle=None, figsize=(12,10), dpi=90):
    """
    Plots AFL entropy and its growth.
    Upper left: AFL entropy
    Upper right: AFL growth rate at early times
    Lower left: AFL approaching the bound at late times
    Lower right: AFL asymptotic growth constant

    Arguments
    entropy[i,j]: The AFL entropies to plot
        i indexes the intraplot entropy lines
        j indexes the time step
    bounds: Maximal entropy value. Can be a scalar or a vector to match the intraplot lines.
    trans_time (int): Transition point from 'early' to 'late' times.
        A good choice is around twice the Thouless time, which is ~log(dimension).

    Optional Keyword Arguments:
    show_bounds: Toggle whether to plot the entropy bounds
    id_entropy (1D array): AFL entropy of identity dynamics.
    labels: Labels for the intraplot lines.
    LR_labels: tuple for labelling the lower right graph of asymptotic growth constants
        (label for the x-axis, x-tick numbers)
    true_agc: the true values of the asymptotic growth constants
    suptitle: suptitle for the whole plot
    figsize: figsize tuple to be passed to matplotlib
    dpi: dpi to be passed to matplotlib
    """
    intraplots = entropy.shape[0]
    times = np.arange(entropy.shape[1]) + 1

    # Argument checks
    assert (labels is None) or (labels.size == intraplots), \
        'Expected {0} intraplot labels but received {1}.'.format(intraplots, labels.size)
    
    scalar_bound = np.isscalar(bounds)
    if scalar_bound:
        bounds = np.full(intraplots, bounds)
    else:
        assert bounds.size == intraplots, 'Expected 1 or {0} bounds but received {1}.'.format(intraplots, bounds.size)

    assert (trans_time < times[-1]) and (trans_time > 0), 'Transition time is invalid.'

    assert (LR_labels is None) or (len(LR_labels[1]) == intraplots), \
        'Expected {0} xticks labels but received {1}.'.format(intraplots, len(LR_labels[1]))
    
    assert (true_agc is None) or (true_agc.size == intraplots), \
        'Expected {0} true_agc values labels but received {1}.'.format(intraplots, true_agc.size)

    # asymptotic growth constants
    agc = np.empty(intraplots)

    # Plotting
    #plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    UL = axes[0,0]
    UR = axes[0,1]
    LL = axes[1,0]
    LR = axes[1,1]

    for i in np.arange(2):
        for j in np.arange(2):
            axes[i,j].grid()
            axes[i,j].set_xlabel('Time Step')

    for i in np.arange(intraplots):
        # upper left: AFL entropy
        line, = UL.plot(times[:trans_time], entropy[i,:trans_time], ls='-', marker='.')
        if labels is not None:
            line.set(label=labels[i])
        if show_bounds and not scalar_bound:
            UL.hlines(bounds[i], 1, trans_time, linestyles=':', colors=line.get_color())

        # upper right: early time growth
        growth = np.diff(entropy[i,:trans_time+1])
        UR.plot(times[:trans_time], growth, ls='-', marker='.')

        # lower left: late time growth
        to_max = bounds[i] - entropy[i,trans_time:]
        LL.semilogy(times[trans_time:], to_max, ls='-', marker='.')

        # S(t) --> S_max - const*exp(-at), so to_max[t] ~ exp(-at)
        # exp(-a(t-1))/exp(-at) = exp(a)
        agc[i] = np.log(to_max[-2] / to_max[-1])
        #agc[i] = to_max[-1] / to_max[-2]

    # lower right: asymptotic growth constants
    if LR_labels is not None:
        LR.set_xlabel(LR_labels[0])
        intraplot_vals = LR_labels[1]
    else:
        LR.set_xlabel('Intraplot Line Index')
        intraplot_vals = np.arange(intraplots)
    ''' Temporarily removing data AGC plotting. '''
    #LR.plot(intraplot_vals, agc, 'r^', label='From Data')
    if true_agc is not None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        reps = np.ceil(intraplots / len(colors)).astype(int)
        colors = np.tile(colors, reps)
        LR.scatter(intraplot_vals, true_agc, c=colors[:intraplots])
        #LR.tick_params(axis='x', labelsize=8)
        LR.set_axisbelow(True)
        #LR.plot(intraplot_vals, true_agc, 'bo', label='From Matrix')
        #LR.legend(framealpha=1)

    # Other plotting things
    if id_entropy is not None:
        id_time = min(id_entropy.size, trans_time)
        UL.plot(times[:id_time], id_entropy[:id_time], 'k--', label='identity')

    if show_bounds and scalar_bound:
        UL.hlines(bounds[0], 1, trans_time, linestyles=':', colors='k')

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20)

    if labels is not None:
        UL.legend(framealpha=1, loc='lower right')
    UL.set_title('AFL Entropy')
    UL.set_ylabel('AFL Entropy')

    UR.set_title('Early Time AFL Growth')
    UR.set_ylabel('AFL Growth Rate')

    LL.set_title('Late Time AFL Growth')
    LL.set_ylabel('Difference from Maximal Entropy')

    LR.set_title('Asymptotic Growth Constant')
    LR.set_ylabel('Asymptotic Growth Constant')

    fig.subplots_adjust(hspace=0.25)

    return fig
