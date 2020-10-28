import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs

def get_goes_ccrs(goes_ds):
    return ccrs.Geostationary(satellite_height=goes_ds.goes_imager_projection.perspective_point_height,
                              central_longitude=goes_ds.goes_imager_projection.longitude_of_projection_origin,
                              sweep_axis=goes_ds.goes_imager_projection.sweep_angle_axis)
def get_goes_extent(goes_ds):
    h = goes_ds.goes_imager_projection.perspective_point_height
    img_extent=(goes_ds.x[0]*h, goes_ds.x[-1]*h, goes_ds.y[-1]*h, goes_ds.y[0]*h)
    return img_extent

def goes_subplot(goes_ds, *args, fig=None, cbar_size="5%" , cbar_pad=0.1 , **kwargs):
    if fig is None:
        fig=plt.gcf()
    crs = get_goes_ccrs(goes_ds)
    img_extent = get_goes_extent(goes_ds)

    ax = fig.add_subplot(*args, projection=crs, **kwargs)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.new_horizontal(size=cbar_size, pad=cbar_pad, axes_class=plt.Axes)

    ax._imshow=ax.imshow.__get__(ax)

    def colorbar(self, *args, **kwargs):
        fig.add_axes(cax)
        cbar = plt.colorbar(*args, cax=cax, **kwargs)
        return cbar

    def imshow(self, *args, extent=img_extent, **kwargs):
        img = self._imshow(*args, extent=extent, **kwargs)
        return img

    ax.colorbar = colorbar.__get__(ax)
    ax.imshow = imshow.__get__(ax)
    return ax

def goes_figure(goes_ds, *args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    crs = get_goes_ccrs(goes_ds)
    img_extent = get_goes_extent(goes_ds)

    def subplot(self, *args, **kwargs):
        ax = goes_subplot(goes_ds, *args, fig=fig, **kwargs)
        return ax

    fig.subplot = subplot.__get__(fig)
    return fig
