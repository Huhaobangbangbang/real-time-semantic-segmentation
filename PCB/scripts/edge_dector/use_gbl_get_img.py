import gerber
from gerber.render import GerberCairoContext
import os
from gerber import load_layer
from gerber.render import RenderSettings, theme
from gerber.render.cairo_backend import GerberCairoContext


our_settings = RenderSettings(color=theme.COLORS['white'], alpha=0.85)
copper = load_layer('/Users/huhao/Downloads/test.gtl')

# Create a new drawing context
ctx = GerberCairoContext()

# Draw the copper layer. render_layer() uses the default color scheme for the
# layer, based on the layer type. Copper layers are rendered as
ctx.render_layer(copper)

# Write output to png file
#ctx.dump('/Users/huhao/Downloads/test.png')