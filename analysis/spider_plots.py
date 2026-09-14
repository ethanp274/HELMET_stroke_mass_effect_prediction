import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import os

# Output directories for figures (git-ignored, so absent in a fresh clone).
for _output_dir in ('results/figures',):
    os.makedirs(_output_dir, exist_ok=True)


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    data = [
        ['AUROC', 'AUPRC', 'Specificity', 'Sensitivity', 'Accuracy'],
        ('Performance on 8-Hour Task\n(Mass General Brigham Cohort)\n', [
            [0.966, 0.875, 0.902, 0.577, 0.829],
            [0.762, 0.612, 0.756, 0.577, 0.467],
            [0.805, 0.593, 0.876, 0.321, 0.616],
            [0.575, 0.373, 0.851, 0.321, 0.235]]),
        ('Performance on 24-Hour Task\n(Mass General Brigham Cohort)\n', [
            [0.967, 0.872, 0.940, 0.912, 0.873],
            [0.941, 0.852, 0.978, 0.912, 0.812],
            [0.780, 0.576, 0.843, 0.376, 0.590],
            [0.547, 0.371, 0.869, 0.376, 0.251]]),
        ('Performance on 8-Hour Task\n(Boston Medical Center Cohort)\n', [
            [0.925, 0.805, 0.941, 0.921, 0.753],
            [0.921, 0.798, 0.992, 0.921, 0.733],
            [0.585, 0.389, 0.782, 0.479, 0.379],
            [0.588, 0.402, 1.000, 0.479, 0.289]]),
        ('Performance on 24-Hour Task\n(Boston Medical Center Cohort)\n', [
            [0.697, 0.469, 0.806, 0.874, 0.486],
            [0.707, 0.484, 0.950, 0.874, 0.487],
            [0.513, 0.330, 0.623, 0.633, 0.292],
            [0.511, 0.353, 0.796, 0.633, 0.304]])
    ]
    return data


tnr_font = {'fontname':'Times New Roman'}

if __name__ == '__main__':
    N = 5
    theta = radar_factory(N, frame='polygon')
    mpl.rc('font',family='Times New Roman')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange']

    # Plot the four cases from the example data on separate Axes
    for ax, (title, case_data) in zip(axs.flat, data):
        #ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.6), horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.1, label='_nolegend_')
        ax.set_thetagrids(theta * 180/np.pi, labels=spoke_labels, **tnr_font, fontsize='small')
        ax.set_rlim(0, 1)
        ax.set_rticks([0, 0.25, 0.5, 0.75, 1.0], labels = ['', '0.25', '0.50', '0.75', '1.00'], **tnr_font, fontsize='x-small') 
        ax.set_rlabel_position(0)
        for tick in ax.yaxis.get_major_ticks():
            print(tick._label)
            

    # add legend relative to top-left plot
    labels = ('HELMET', 'HELMET (Filtered)', 'EDEMA', 'EDEMA (Filtered)')
    legend = axs[0, 0].legend(labels, loc=(1.3, -0.25),
                              labelspacing=0.1, fontsize='small')

    # save individual plots

    labs = ['mgb_8', 'mgb_24', 'bmc_8', 'bmc_24']

    plt.show()


    '''
    for i, (title, case_data) in zip(range(4), data):
        plt.figure()
        ax = plt.subplot(111, projection='radar')

        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.1, label='_nolegend_')

        ax.set_thetagrids(theta * 180/np.pi, labels=spoke_labels, **tnr_font, fontsize='small')
        ax.set_rlim(0, 1)
        ax.set_rticks([0, 0.25, 0.5, 0.75, 1.0], labels = ['', '0.25', '0.50', '0.75', '1.00'], **tnr_font, fontsize='x-small')
        ax.set_rlabel_position(0)

        plt.savefig(f'results/figures/radar_plot_{labs[i]}.png', dpi=450)
        plt.close()
    '''
