import matplotlib.pyplot as plt

rcparams = {
    'text.usetex': False,   # keep native mathtext, no TeX runtime required
    'font.family': 'serif',
    'font.serif': [
        'Times New Roman',   # available on Ubuntu (mscorefonts)
        'TeX Gyre Termes',   # free Times clone (fonts-texgyre)
        'Nimbus Roman',      # Ghostscript Times clone (gsfonts)
        'Times',             # available on macOS
        'DejaVu Serif'       # guaranteed fallback
    ],

    # --- Math font alignment ---
    'mathtext.fontset': 'custom',     # use our overrides below
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',

    # --- Figure + text sizing ---
    'figure.figsize': (3.25, 2.0086104634371584),
    'figure.constrained_layout.use': True,
    'figure.autolayout': False,
    'savefig.pad_inches': 0.015,

    'font.size': 12,
    'axes.labelsize': 15,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 12,
}


plt.rcParams.update(rcparams)
