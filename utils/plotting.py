import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import numpy as np

def get_goes_ccrs(goes_ds):
    return ccrs.Geostationary(satellite_height=goes_ds.goes_imager_projection.perspective_point_height,
                              central_longitude=goes_ds.goes_imager_projection.longitude_of_projection_origin,
                              sweep_axis=goes_ds.goes_imager_projection.sweep_angle_axis)
def get_goes_extent(goes_ds):
    h = goes_ds.goes_imager_projection.perspective_point_height
    img_extent=(goes_ds.x[0]*h, goes_ds.x[-1]*h, goes_ds.y[-1]*h, goes_ds.y[0]*h)
    return img_extent

def get_goes_ccrs(goes_ds):
    return ccrs.Geostationary(satellite_height=goes_ds.goes_imager_projection.perspective_point_height,
                              central_longitude=goes_ds.goes_imager_projection.longitude_of_projection_origin,
                              sweep_axis=goes_ds.goes_imager_projection.sweep_angle_axis)
def get_goes_extent(goes_ds):
    h = goes_ds.goes_imager_projection.perspective_point_height
    img_extent=(goes_ds.x.data[0]*h, goes_ds.x.data[-1]*h, goes_ds.y.data[-1]*h, goes_ds.y.data[0]*h)
    return img_extent

def goes_subplot(goes_ds, *args, fig=None, cbar_size="5%" , cbar_pad=0.1 , **kwargs):
    if fig is None:
        fig=plt.gcf()
    crs = get_goes_ccrs(goes_ds)
    img_extent = get_goes_extent(goes_ds)
    h = goes_ds.goes_imager_projection.perspective_point_height

    ax = fig.add_subplot(*args, projection=crs, **kwargs)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.new_horizontal(size=cbar_size, pad=cbar_pad, axes_class=plt.Axes)

    ax._imshow=ax.imshow.__get__(ax)
    ax._contour=ax.contour.__get__(ax)
    ax._contourf=ax.contourf.__get__(ax)
    ax._quiver=ax.quiver.__get__(ax)

    def colorbar(self, *args, **kwargs):
        fig.add_axes(cax)
        cbar = plt.colorbar(*args, cax=cax, **kwargs)
        return cbar
    ax.colorbar = colorbar.__get__(ax)

    def imshow(self, *args, extent=img_extent, **kwargs):
        img = self._imshow(*args, extent=extent, **kwargs)
        return img
    ax.imshow = imshow.__get__(ax)

    def contour(self, data, *args, **kwargs):
        cntr = self._contour(goes_ds.x*h, goes_ds.y*h, data, *args, **kwargs)
        self.set_extent(img_extent, self.projection)
        return cntr
    ax.contour = contour.__get__(ax)

    def contourf(self, data, *args, **kwargs):
        cntr = self._contourf(goes_ds.x*h, goes_ds.y*h, data, *args, **kwargs)
        self.set_extent(img_extent, self.projection)
        return cntr
    ax.contourf = contourf.__get__(ax)

    def quiver(self, u, v, *args, spacing=1, block_method='slice', **kwargs):
        if block_method == 'slice':
            slc = slice(spacing//2, None, spacing)
            quiv = self._quiver(goes_ds.x[slc]*h, goes_ds.y[slc]*h,
                                u[slc,slc], v[slc,slc], *args, **kwargs)
        elif block_method == 'reduce':
            from skimage.measure import block_reduce
            new_x, new_y = block_reduce(goes_ds.x.data*h, (spacing,), np.nanmean), \
                            block_reduce(goes_ds.y.data*h, (spacing,), np.nanmean)
            new_u, new_v = block_reduce(u, (spacing, spacing), np.nanmean), \
                            block_reduce(v, (spacing, spacing), np.nanmean)
            quiv = self._quiver(new_x, new_y, new_u, new_v, *args, **kwargs)
        else:
            raise ValueError("invalid input for 'block_method'")
        return quiv
    ax.quiver = quiver.__get__(ax)

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
