# Tools for working with GLM data. Mostly adapted from glmtools
import xarray as xr
import numpy as np
import pyproj as proj4

from glmtools.io.lightning_ellipse import lightning_ellipse_rev
from lmatools.coordinateSystems import CoordinateSystem
from lmatools.grid.fixed import get_GOESR_coordsys

this_ellps=0

# equatorial and polar radii
ltg_ellps_re, ltg_ellps_rp = lightning_ellipse_rev[this_ellps]

# Functions from GLM notebook for parallax correction
def semiaxes_to_invflattening(semimajor, semiminor):
    """ Calculate the inverse flattening from the semi-major
        and semi-minor axes of an ellipse"""
    rf = semimajor/(semimajor-semiminor)
    return rf

class GeostationaryFixedGridSystemAltEllipse(CoordinateSystem):

    def __init__(self, subsat_lon=0.0, subsat_lat=0.0, sweep_axis='y',
                 sat_ecef_height=35785831.0,
                 semimajor_axis=None,
                 semiminor_axis=None,
                 datum='WGS84'):
        """
        Satellite height is with respect to an arbitray ellipsoid whose
        shape is given by semimajor_axis (equatorial) and semiminor_axis(polar)

        Fixed grid coordinates are in radians.
        """
        rf = semiaxes_to_invflattening(semimajor_axis, semiminor_axis)
        print("Defining alt ellipse for Geostationary with rf=", rf)
        self.ECEFxyz = proj4.Proj(proj='geocent',
            a=semimajor_axis, rf=rf)
        self.fixedgrid = proj4.Proj(proj='geos', lon_0=subsat_lon,
            lat_0=subsat_lat, h=sat_ecef_height, x_0=0.0, y_0=0.0,
            units='m', sweep=sweep_axis,
            a=semimajor_axis, rf=rf)
        self.h=sat_ecef_height

    def toECEF(self, x, y, z):
        X, Y, Z = x*self.h, y*self.h, z*self.h
        return proj4.transform(self.fixedgrid, self.ECEFxyz, X, Y, Z)

    def fromECEF(self, x, y, z):
        X, Y, Z = proj4.transform(self.ECEFxyz, self.fixedgrid, x, y, z)
        return X/self.h, Y/self.h, Z/self.h

class GeographicSystemAltEllps(CoordinateSystem):
    """
    Coordinate system defined on the surface of the earth using latitude,
    longitude, and altitude, referenced by default to the WGS84 ellipse.

    Alternately, specify the ellipse shape using an ellipse known
    to pyproj, or [NOT IMPLEMENTED] specify r_equator and r_pole directly.
    """
    def __init__(self, ellipse='WGS84', datum='WGS84',
                 r_equator=None, r_pole=None):
        if (r_equator is not None) | (r_pole is not None):
            rf = semiaxes_to_invflattening(r_equator, r_pole)
            print("Defining alt ellipse for Geographic with rf", rf)
            self.ERSlla = proj4.Proj(proj='latlong', #datum=datum,
                                     a=r_equator, rf=rf)
            self.ERSxyz = proj4.Proj(proj='geocent', #datum=datum,
                                     a=r_equator, rf=rf)
        else:
            # lat lon alt in some earth reference system
            self.ERSlla = proj4.Proj(proj='latlong', ellps=ellipse, datum=datum)
            self.ERSxyz = proj4.Proj(proj='geocent', ellps=ellipse, datum=datum)
    def toECEF(self, lon, lat, alt):
        projectedData = np.array(proj4.transform(self.ERSlla, self.ERSxyz, lon, lat, alt ))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0,:], projectedData[1,:], projectedData[2,:]

    def fromECEF(self, x, y, z):
        projectedData = np.array(proj4.transform(self.ERSxyz, self.ERSlla, x, y, z ))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0,:], projectedData[1,:], projectedData[2,:]


def get_GOESR_coordsys_alt_ellps(sat_lon_nadir=-75.0):
    goes_sweep = 'x' # Meteosat is 'y'
    datum = 'WGS84'
    sat_ecef_height=35786023.0
    geofixcs = GeostationaryFixedGridSystemAltEllipse(subsat_lon=sat_lon_nadir,
                    semimajor_axis=ltg_ellps_re, semiminor_axis=ltg_ellps_rp,
                    datum=datum, sweep_axis=goes_sweep,
                    sat_ecef_height=sat_ecef_height)
    grs80lla = GeographicSystemAltEllps(r_equator=ltg_ellps_re, r_pole=ltg_ellps_rp,
                                datum='WGS84')
    return geofixcs, grs80lla
