import gerber
from gerber.render import GerberCairoContext
import os
from gerber import load_layer
from gerber.render import RenderSettings, theme
from gerber.render.cairo_backend import GerberCairoContext
def create_png(img_path):
    copper = load_layer(img_path)
    ctx = GerberCairoContext()
    ctx.render_layer(copper)
    # # Write output to png file
    ctx.dump('/Users/huhao/Downloads/template/png/test3.png')



def create_svg(img_path):
    """利用gl2生成svg"""
    import gerber
    from gerber.render import GerberCairoContext

    # Read gerber and Excellon files
    top_copper = gerber.read(img_path)
    nc_drill = gerber.read('example.txt')

    # Rendering context
    ctx = GerberCairoContext()

    # Create SVG image
    top_copper.render(ctx)
    nc_drill.render(ctx, '/Users/huhao/Downloads/template/png/test.svg')

if __name__ == '__main__':
    img_path = '/Users/huhao/Downloads/template/1082s433367a.gl2'
    #create_png(img_path)
    create_svg(img_path)
