from matplotlib import colors as mcolors
import six
import numpy as np


def get_named_colors():
    if hasattr(mcolors, 'BASE_COLORS'):
        return get_named_colors_new()
    else:
        return get_named_colors_old()


def get_named_colors_new():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    return [name for hsv, name in by_hsv]


def get_named_colors_old():
    colors_ = list(six.iteritems(mcolors.cnames))

    # Add the single letter colors.
    for name, rgb in six.iteritems(mcolors.ColorConverter.colors):
        hex_ = mcolors.rgb2hex(rgb)
        colors_.append((name, hex_))

    # Transform to hex color values.
    hex_ = [color[1] for color in colors_]
    # Get the rgb equivalent.
    rgb = [mcolors.hex2color(color) for color in hex_]
    # Get the hsv equivalent.
    hsv = [mcolors.rgb_to_hsv(color) for color in rgb]

    # Split the hsv values to sort.
    hue = [color[0] for color in hsv]
    sat = [color[1] for color in hsv]
    val = [color[2] for color in hsv]

    # Sort by hue, saturation and value.
    ind = np.lexsort((val, sat, hue))
    sorted_colors = [colors_[i] for i in ind]
    return [name for name, hx in sorted_colors]
