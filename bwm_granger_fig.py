import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/admin/int-brain-lab/bwm_dmn')
import matplotlib.pyplot as plt
from ibl_style.style import figure_style
from ibl_style.utils import MM_TO_INCH, get_coords, add_label
import figrid as fg
import numpy as np
from pathlib import Path

from bwm_dmn.granger import (plot_gc, heatmap_adjacency, scatter_direction, plot_graph, plot_strip_pairs,
                             scatter_similarity, plot_multi_graph)

figure_style()

f_size_l = 7
f_size = 7
f_size_s = 6
f_size_xs = 5


def adjust_subplots(fig, adjust=5, extra=2):
    width, height = fig.get_size_inches() / MM_TO_INCH
    if not isinstance(adjust, int):
        assert len(adjust) == 4
    else:
        adjust = [adjust] *  4
    fig.subplots_adjust(top=1 - adjust[0] / height, bottom=(adjust[1] + extra) / height,
                        left=adjust[2] / width, right=1 - adjust[3] / width)


def shift_plot(ax, x, y):
    pos = ax.get_position()
    pos.x0 = pos.x0 + x
    pos.x1 = pos.x1 + x
    pos.y0 = pos.y0 + y
    pos.y1 = pos.y1 + y
    ax.set_position(pos)

figure_style()
fig = plt.figure(figsize=(183 * MM_TO_INCH, 170 * MM_TO_INCH))
width, height = fig.get_size_inches() / MM_TO_INCH
xspans1 = get_coords(width, ratios=[1, 1, 1, 1], space=[20, 10, 15], pad=10, span=(0, 1))
xspans2 = get_coords(width, ratios=[1, 1], space=15, pad=10, span=(0, 1))
xspans3 = get_coords(width, ratios=[1], space=15, pad=10, span=(0, 1))
xspans4 = get_coords(width, ratios=[1, 1, 1], space=15, pad=10, span=(0, 1))

yspans = get_coords(height, ratios=[1, 5, 4, 1], space=[0, 10, 15], pad=5, span=(0, 1))
yspans2 = get_coords(height, ratios=[1, 1, 1], space=10, pad=0, span=[0.58, 1])



axs = {
    'a': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=[0.04, 0.12]),
    'b': fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=[0.04, 0.12]),
    'c': fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=[0.04, 0.17]),
    'd': fg.place_axes_on_grid(fig, xspan=xspans1[3], yspan=[0.04, 0.15]),
    'e': fg.place_axes_on_grid(fig, xspan=[0, 0.5], yspan=[0.17, 0.64]),
    #'e': fg.place_axes_on_grid(fig, xspan=[0, 0.5], yspan=[0.15, 0.68]),
    'f': fg.place_axes_on_grid(fig, xspan=[0.4, 0.85], yspan=[0.21, 0.62], dim=(4, 5), hspace=0.1, wspace=0),
    'h': fg.place_axes_on_grid(fig, xspan=[0.92, 1], yspan=[0.24, 0.60], dim=(3, 1), hspace=0.45),
    'g_1': fg.place_axes_on_grid(fig, xspan=[0.02, 1], yspan=yspans2[0]),
    'g_2': fg.place_axes_on_grid(fig, xspan=[0.02, 1], yspan=yspans2[1]),
    'g_3': fg.place_axes_on_grid(fig, xspan=[0.02, 1], yspan=yspans2[2]),
}

labels = []
padx = 10
pady = 3
labels.append(add_label('a', fig, xspans1[0], [0.05, 0.12], 25, 8, fontsize=8))
labels.append(add_label('b', fig, xspans1[1], [0.05, 0.12], 15, 8, fontsize=8))
labels.append(add_label('c', fig, xspans1[2], [0.05, 0.12], 5, 8, fontsize=8))
labels.append(add_label('d', fig, xspans1[3], [0.05, 0.12], 10, 8, fontsize=8))
labels.append(add_label('e', fig, xspans2[0], [0.2, 0.64], 35, pady, fontsize=8))
labels.append(add_label('f', fig, xspans2[1], [0.2, 0.64], 30, pady, fontsize=8))
labels.append(add_label('g', fig, xspans3[0], [0.6, 1], 15, 0, fontsize=8))
labels.append(add_label('h', fig, [0.92, 1], [0.24, 0.62], padx, pady, fontsize=8))
fg.add_labels(fig, labels)



adjust_subplots(fig, [2, 13, 3, 5])

plot_gc('af55d16f-0e31-4073-bdb5-26da54914aa2', single_pair=True, axs=[axs['a'], axs['b']])
axs['a'].set_xlabel('Time (s)', fontsize=f_size, va='center')
axs['a'].set_ylabel('Firing rate (Hz)', fontsize=f_size)
axs['a'].set_title('Example segment', fontsize=f_size_l)

for text in axs['a'].texts:
    text.set_fontsize(f_size_s)
    y0 = 15 if text.get_text() == 'CP' else 20
    pos = text.get_position()
    text.set_position((pos[0] - 1, pos[1] + y0))

axs['b'].set_xlabel('Frequency (Hz)', fontsize=f_size, va='center')
axs['b'].set_ylabel('Pairwise spectral \n granger prediction', fontsize=f_size)
leg = axs['b'].legend_
leg.set_frame_on(False)
for text in leg.get_texts():
    text.set_fontsize(f_size_s)
    a, b = text.get_text().split(' --> ')
    text.set_text(f'{a} to {b}')


heatmap_adjacency(ax=axs['c'], ms=0.05)
xlabels = axs['c'].get_xticklabels()
axs['c'].xaxis.set_tick_params(labelsize=f_size_xs)
axs['c'].yaxis.set_tick_params(labelsize=f_size_xs)

scatter_direction(ax=axs['d'], annotate=True)
title = axs['d'].get_title()
axs['d'].set_title(title, fontsize=f_size_s)
xlabel = axs['d'].get_xlabel()
axs['d'].set_xlabel('A to B', fontsize=f_size, va='center')
ylabel = axs['d'].get_ylabel()
axs['d'].set_ylabel('B to A', fontsize=f_size, va='center')
for text in axs['d'].texts:
    text.set_fontsize(f_size_s)

axs_g = np.array([axs['g_1'], axs['g_2'], axs['g_3']]) # , axs['g_4']])
plot_strip_pairs(markersize=0.8, axs=axs_g)
for i, ax in enumerate(axs_g):
    if i == 0:
        ax.spines['left'].set_bounds(0, 0.012)
        ax.set_yticks([0, 0.01])
    else:
        ax.spines['left'].set_bounds(0, 0.03)
        ax.set_yticks([0, 0.01, 0.02, 0.03])
    ax.xaxis.set_tick_params(labelsize=f_size_xs)
    ax.set_ylabel('Granger', fontsize=f_size, va='center')
    ax.set_zorder(5 - i)
    for i, text in enumerate(ax.texts):
        text.set_fontsize(f_size_xs)
        pos = text.get_position()
        text.set_position((pos[0], pos[1] -0.01))
    shift_plot(ax, 0.02, 0.005)
    ax.set_ylim(0, 0.038)

ylabel = axs['g_1'].yaxis.get_label()
ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1] -0.3))
ylabel = axs['g_2'].yaxis.get_label()
ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1] -0.1))
ylabel = axs['g_3'].yaxis.get_label()
ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1] -0.1))


axs['h_1'] =axs['h'][0]
axs['h_2'] =axs['h'][1]
axs['h_3'] =axs['h'][2]
scatter_similarity(axs=[axs['h_1'], axs['h_2'], axs['h_3']])
for ax in [axs['h_1'], axs['h_2'], axs['h_3']]:
    ylabel = ax.get_ylabel()
    ax.set_ylabel(ylabel.capitalize(), fontsize=f_size, va='center')
    xlabel = ax.get_xlabel()
    ax.set_xlabel(xlabel.capitalize(), fontsize=f_size, va='center')
    title = ax.get_title()
    ax.set_title('')
    ax.text(0.5, 0.6, title, transform=ax.transAxes, fontsize=f_size_s, ha='center')
axs['h_3'].xaxis.set_tick_params(rotation=0)
axs['h_3'].set_zorder(50)


plot_multi_graph(axs=np.array(axs['f']), sa=5)

j = 0
col = 0
axs['f'] = np.array(axs['f'])
for i in range(axs['f'].flatten().size):

    row = np.mod(j, 4)
    if j != 0 and row == 0:
        col += 1

    ax = axs['f'][row,col]
    if i == 0:
        ax_anchor = ax

    for text in ax.texts:
        text.remove()
    title = ax.get_title()
    ax.set_title('')
    ax.text(col * 1.05 + 0.5, 1.01 - (row * 1.1), title, transform=ax_anchor.transAxes, fontsize=f_size_xs,
            ha='center', va='bottom')
    ax.set_zorder(10)
    j += 1


plot_graph(ax=axs['e'], sa=5)
for i, text in enumerate(axs['e'].texts):
    if np.mod(i, 2) == 0:
        text.set_fontsize(f_size_xs)
    else:
        text.remove()
axs['e'].set_zorder(10)
shift_plot(axs['e'], -0.06, 0)

ed_save_path = Path('/Users/admin/bwm/ed')
ed_save_path.mkdir(exist_ok=True, parents=True)
save_name = 'n6_ed_fig4_granger'
fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)