import numpy as np
import math
import copy
from datetime import datetime, timedelta


###############################################################################
# Constants - units of meters, kilograms, seconds, radians
###############################################################################

# Generic Values
arcsec2rad = np.pi/(3600.*180.)

# Earth parameters
wE = 7.2921158553e-5  # rad/s
GME = 398600.4415e9  # m^3/s^2
J2E = 1.082626683e-3

# WGS84 Data (Pratap and Misra P. 103)
Re = 6378137.0   # m
rec_f = 298.257223563

# Light and Flux Parameters
c_light = 299792458.  # m/s
AU_m = 149597870700.  # m
SF = 1367.  # W/m^2 = kg/s^3
C_sunvis = 455.   # W/m^2 = kg/s^3




def check_visibility(X, ):
    
    
    return visible_flag



def cart2kep(cart, GM=GME):
    '''
    This function converts a Cartesian state vector in inertial frame to
    Keplerian orbital elements.
    
    Parameters
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [m]
    cart[1] : y
      Position in y               [m]
    cart[2] : z
      Position in z               [m]
    cart[3] : dx
      Velocity in x               [m/s]
    cart[4] : dy
      Velocity in y               [m/s]
    cart[5] : dz
      Velocity in z               [m/s]
      
    Returns
    ------
    elem : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    elem[0] : a
      Semi-Major Axis             [km]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : theta
      True Anomaly                [deg]    
      
    '''
    
    # Retrieve input cartesian coordinates
    r_vect = cart[0:3].reshape(3,1)
    v_vect = cart[3:6].reshape(3,1)

    # Calculate orbit parameters
    r = np.linalg.norm(r_vect)
    ir_vect = r_vect/r
    v2 = np.linalg.norm(v_vect)**2
    h_vect = np.cross(r_vect, v_vect, axis=0)
    h = np.linalg.norm(h_vect)

    # Calculate semi-major axis
    a = 1./(2./r - v2/GM)     # km
    
    # Calculate eccentricity
    e_vect = np.cross(v_vect, h_vect, axis=0)/GM - ir_vect
    e = np.linalg.norm(e_vect)

    # Calculate RAAN and inclination
    ih_vect = h_vect/h

    RAAN = math.atan2(ih_vect[0,0], -ih_vect[1,0])   # rad
    i = math.acos(ih_vect[2,0])   # rad
    if RAAN < 0.:
        RAAN += 2.*math.pi

    # Apply correction for circular orbit, choose e_vect to point
    # to ascending node
    if e != 0:
        ie_vect = e_vect/e
    else:
        ie_vect = np.array([[math.cos(RAAN)], [math.sin(RAAN)], [0.]])

    # Find orthogonal unit vector to complete perifocal frame
    ip_vect = np.cross(ih_vect, ie_vect, axis=0)

    # Form rotation matrix PN
    PN = np.concatenate((ie_vect, ip_vect, ih_vect), axis=1).T

    # Calculate argument of periapsis
    w = math.atan2(PN[0,2], PN[1,2])  # rad
    if w < 0.:
        w += 2.*math.pi

    # Calculate true anomaly
    cross1 = np.cross(ie_vect, ir_vect, axis=0)
    tan1 = np.dot(cross1.T, ih_vect).flatten()[0]
    tan2 = np.dot(ie_vect.T, ir_vect).flatten()[0]
    theta = math.atan2(tan1, tan2)    # rad
    
    # Update range of true anomaly for elliptical orbits
    if a > 0. and theta < 0.:
        theta += 2.*math.pi
    
    # Convert angles to deg
    i *= 180./math.pi
    RAAN *= 180./math.pi
    w *= 180./math.pi
    theta *= 180./math.pi
    
    # Form output
    elem = np.array([[a], [e], [i], [RAAN], [w], [theta]])
      
    return elem


def kep2cart(elem, GM=GME):
    '''
    This function converts a vector of Keplerian orbital elements to a
    Cartesian state vector in inertial frame.
    
    Parameters
    ------
    elem : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    elem[0] : a
      Semi-Major Axis             [m]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : theta
      True Anomaly                [deg]
      
      
    Returns
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [km]
    cart[1] : y
      Position in y               [km]
    cart[2] : z
      Position in z               [km]
    cart[3] : dx
      Velocity in x               [km/s]
    cart[4] : dy
      Velocity in y               [km/s]
    cart[5] : dz
      Velocity in z               [km/s]  
      
    '''
    
    # Retrieve input elements, convert to radians
    a = float(elem[0])
    e = float(elem[1])
    i = float(elem[2]) * math.pi/180
    RAAN = float(elem[3]) * math.pi/180
    w = float(elem[4]) * math.pi/180
    theta = float(elem[5]) * math.pi/180

    # Calculate h and r
    p = a*(1 - e**2)
    h = np.sqrt(GM*p)
    r = p/(1. + e*math.cos(theta))

    # Calculate r_vect and v_vect
    r_vect = r * \
        np.array([[math.cos(RAAN)*math.cos(theta+w) - math.sin(RAAN)*math.sin(theta+w)*math.cos(i)],
                  [math.sin(RAAN)*math.cos(theta+w) + math.cos(RAAN)*math.sin(theta+w)*math.cos(i)],
                  [math.sin(theta+w)*math.sin(i)]])

    vv1 = math.cos(RAAN)*(math.sin(theta+w) + e*math.sin(w)) + \
          math.sin(RAAN)*(math.cos(theta+w) + e*math.cos(w))*math.cos(i)

    vv2 = math.sin(RAAN)*(math.sin(theta+w) + e*math.sin(w)) - \
          math.cos(RAAN)*(math.cos(theta+w) + e*math.cos(w))*math.cos(i)

    vv3 = -(math.cos(theta+w) + e*math.cos(w))*math.sin(i)
    
    v_vect = -GM/h * np.array([[vv1], [vv2], [vv3]])

    cart = np.concatenate((r_vect, v_vect), axis=0)
    
    return cart


def ecef2enu(r_ecef, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ECEF to ENU frame.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)
    lat = lat*math.pi/180  # rad
    lon = lon*math.pi/180  # rad

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),  math.sin(lon1), 0.],
                   [-math.sin(lon1), math.cos(lon1), 0.],
                   [0.,              0.,             1.]])

    R = np.dot(R1, R3)

    r_enu = np.dot(R, r_ecef)

    return r_enu


def enu2ecef(r_enu, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ENU to ECEF frame.

    Parameters
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)
    lat = lat*math.pi/180  # rad
    lon = lon*math.pi/180  # rad

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),   math.sin(lon1), 0.],
                   [-math.sin(lon1),  math.cos(lon1), 0.],
                   [0.,                           0., 1.]])

    R = np.dot(R1, R3)

    R2 = R.T

    r_ecef = np.dot(R2, r_enu)

    return r_ecef


def ecef2latlonht(r_ecef):
    '''
    This function converts the coordinates of a position vector from
    the ECEF frame to geodetic latitude, longitude, and height.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]

    Returns
    ------
    lat : float
      latitude [deg] [-90,90]
    lon : float
      longitude [deg] [-180,180]
    ht : float
      height [km]
    '''

    a = Re   # m

    # Get components from position vector
    x = float(r_ecef[0])
    y = float(r_ecef[1])
    z = float(r_ecef[2])

    # Compute longitude
    f = 1./rec_f
    e = np.sqrt(2.*f - f**2.)
    lon = math.atan2(y, x) * 180./math.pi  # deg

    # Iterate to find height and latitude
    p = np.sqrt(x**2. + y**2.)  # km
    lat = 0.*math.pi/180.
    lat_diff = 1.
    tol = 1e-12

    while abs(lat_diff) > tol:
        lat0 = copy.copy(lat)  # rad
        N = a/np.sqrt(1 - e**2*(math.sin(lat0)**2))  # m
        ht = p/math.cos(lat0) - N
        lat = math.atan((z/p)/(1 - e**2*(N/(N + ht))))
        lat_diff = lat - lat0

    lat = lat*180./math.pi  # deg

    return lat, lon, ht


def latlonht2ecef(lat, lon, ht):
    '''
    This function converts geodetic latitude, longitude and height
    to a position vector in ECEF.

    Parameters
    ------
    lat : float
      geodetic latitude [deg]
    lon : float
      geodetic longitude [deg]
    ht : float
      geodetic height [m]

    Returns
    ------
    r_ecef = 3x1 numpy array
      position vector in ECEF [m]
    '''

    # Convert to radians
    lat = lat*math.pi/180    # rad
    lon = lon*math.pi/180    # rad

    # Compute flattening and eccentricity
    f = 1/rec_f
    e = np.sqrt(2*f - f**2)

    # Compute ecliptic plane and out of plane components
    C = Re/np.sqrt(1 - e**2*math.sin(lat)**2)
    S = Re*(1 - e**2)/np.sqrt(1 - e**2*math.sin(lat)**2)

    rd = (C + ht)*math.cos(lat)
    rk = (S + ht)*math.sin(lat)

    # Compute ECEF position vector
    r_ecef = np.array([[rd*math.cos(lon)], [rd*math.sin(lon)], [rk]])

    return r_ecef


def compute_ERA(UT1_JD):
    '''
    This function computes the Earth Rotation Angle (ERA) and the ERA rotation
    matrix based on UT1 time. The ERA is modulated to lie within [0,2*pi] and 
    is computed using the precise equation given by Eq. 5.15 in [5].

    The ERA is the angle between the Celestial Intermediate Origin, CIO, and 
    Terrestrial Intermediate Origin, TIO (a reference meridian 100m offset 
    from Greenwich meridian).
    
    r_CIRS = R * r_TIRS
   
    Parameters
    ------
    UT1_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC UT1
        
    Returns
    ------
    R : 3x3 numpy array
        matrix to compute frame rotation
    
    '''
    
    # Compute ERA based on Eq. 5.15 of [5]
    d,i = math.modf(UT1_JD)
    ERA = 2.*math.pi*(d + 0.7790572732640 + 0.00273781191135448*(UT1_JD - 2451545.))
    
    # Compute ERA between [0, 2*pi]
    ERA = ERA % (2.*math.pi)
    if ERA < 0.:
        ERA += 2*math.pi
        
#    print(ERA)
    
    # Construct rotation matrix
    # R = ROT3(-ERA)
    ct = math.cos(ERA)
    st = math.sin(ERA)
    R = np.array([[ct, -st, 0.],
                  [st,  ct, 0.],
                  [0.,  0., 1.]])

    return ERA, R


###############################################################################
# Time Systems
###############################################################################


def utcdt2ttjd(UTC, TAI_UTC):
    '''
    This function converts a UTC time to Terrestrial Time (TT) in Julian Date
    (JD) format.
    
    UTC = TAI - TAI_UTC
    TT = TAI + 32.184
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    TAI_UTC : float
        EOP parameter, time offset between atomic time (TAI) and UTC 
        (10 + leap seconds)        
    
    Returns
    ------
    TT_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC TT
    
    '''
    
    UTC_JD = dt2jd(UTC)
    TT_JD = UTC_JD + (TAI_UTC + 32.184)/86400.
    
    return TT_JD


def dt2jd(dt):
    '''
    This function converts a calendar time to Julian Date (JD) fractional days
    since 12:00:00 Jan 1 4713 BC.  No conversion between time systems is 
    performed.
    
    Parameters
    ------
    dt : datetime object
        time in calendar format
    
    Returns
    ------
    JD : float
        fractional days since 12:00:00 Jan 1 4713 BC
    
    '''
    
    MJD = dt2mjd(dt)
    JD = MJD + 2400000.5
    
    return JD


def dt2mjd(dt):
    '''
    This function converts a calendar time to Modified Julian Date (MJD)
    fractional days since 1858-11-17 00:00:00.  No conversion between time
    systems is performed.
    
    Parameters
    ------
    dt : datetime object
        time in calendar format
    
    Returns
    ------
    MJD : float
        fractional days since 1858-11-17 00:00:00
    '''
    
    MJD_datetime = datetime(1858, 11, 17, 0, 0, 0)
    delta = dt - MJD_datetime
    MJD = delta.total_seconds()/timedelta(days=1).total_seconds()
    
    return MJD


def jd2cent(JD):
    '''
    This function computes Julian centuries since J2000. No conversion between
    time systems is performed.
    
    Parameters
    ------
    JD : float
        fractional days since 12:00:00 Jan 1 4713 BC
    
    Returns
    ------
    cent : float
        fractional centuries since 12:00:00 Jan 1 2000
    '''
    
    cent = (JD - 2451545.)/36525.
    
    return cent


def utcdt2ut1jd(UTC, UT1_UTC):
    '''
    This function converts a UTC time to UT1 in Julian Date (JD) format.
    
    UT1_UTC = UT1 - UTC
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    UT1_UTC : float
        EOP parameter, time offset between UT1 and UTC 
    
    Returns
    ------
    UT1_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC UT1
    
    '''
    
    UTC_JD = dt2jd(UTC)
    UT1_JD = UTC_JD + (UT1_UTC/86400.)
    
    return UT1_JD


###############################################################################
# Ephemeris
###############################################################################


def compute_sun_coords(TT_cent):
    '''
    This function computes sun coordinates using the simplified model in
    Meeus Ch 25.  The results here follow the "low accuracy" model and are
    expected to have an accuracy within 0.01 deg.
    
    Parameters
    ------
    TT_cent : float
        Julian centuries since J2000 TT
        
    Returns
    ------
    sun_eci_geom : 3x1 numpy array
        geometric position vector of sun in ECI [m]
    sun_eci_app : 3x1 numpy array
        apparent position vector of sun in ECI [m]
        
    Reference
    ------
    [1] Meeus, J., "Astronomical Algorithms," 2nd ed., 1998, Ch 25.
    
    Note that Meeus Ch 7 and Ch 10 describe time systems TDT and TDB as 
    essentially the same for the purpose of these calculations (they will
    be within 0.0017 seconds of one another).  The time system TT = TDT is 
    chosen for consistency with the IAU Nutation calculations which are
    explicitly given in terms of TT.
    
    '''
    
    # Conversion
    deg2rad = math.pi/180.
    
    # Geometric Mean Longitude of the Sun (Mean Equinox of Date)
    Lo = 280.46646 + (36000.76983 + 0.0003032*TT_cent)*TT_cent   # deg
    Lo = Lo % 360.
    
    # Mean Anomaly of the Sun
    M = 357.52911 + (35999.05028 - 0.0001537*TT_cent)*TT_cent    # deg
    M = M % 360.
    Mrad = M*deg2rad                                             # rad
    
    # Eccentricity of Earth's orbit
    ecc = 0.016708634 + (-0.000042037 - 0.0000001267*TT_cent)*TT_cent
    
    # Sun's Equation of Center
    C = (1.914602 - 0.004817*TT_cent - 0.000014*TT_cent*TT_cent)*math.sin(Mrad) + \
        (0.019993 - 0.000101*TT_cent)*math.sin(2.*Mrad) + 0.000289*math.sin(3.*Mrad)  # deg
        
    # Sun True Longitude and True Anomaly
    true_long = Lo + C  # deg
    true_anom = M + C   # deg
    true_long_rad = true_long*deg2rad
    true_anom_rad = true_anom*deg2rad
    
    # Sun radius (distance from Earth)
    R_AU = 1.000001018*(1. - ecc**2)/(1 + ecc*math.cos(true_anom_rad))       # AU
    R_m = R_AU*AU_m                                                   # m
    
    # Compute Sun Apparent Longitude
    Omega = 125.04 - 1934.136*TT_cent                                   # deg
    Omega_rad = Omega*deg2rad                                           # rad
    apparent_long = true_long - 0.00569 - 0.00478*math.sin(Omega_rad)        # deg
    apparent_long_rad = apparent_long*deg2rad                           # rad
    
    # Obliquity of the Ecliptic (Eq 22.2)
    Eps0 = (((0.001813*TT_cent - 0.00059)*TT_cent - 46.8150)*TT_cent 
              + 84381.448)/3600.                                        # deg
    Eps0_rad = Eps0*deg2rad                                             # rad
    cEps0 = math.cos(Eps0_rad)
    sEps0 = math.sin(Eps0_rad)
    
    # Geometric Coordinates
    sun_ecliptic_geom = R_m*np.array([[math.cos(true_long_rad)],
                                      [math.sin(true_long_rad)],
                                      [                     0.]])

    # r_Equator = R1(-Eps) * r_Ecliptic
    R1 = np.array([[1.,       0.,       0.],
                   [0.,    cEps0,   -sEps0],
                   [0.,    sEps0,    cEps0]])
    
    sun_eci_geom = np.dot(R1, sun_ecliptic_geom)
    
    # Apparent Coordinates
    Eps_true = Eps0 + 0.00256*math.cos(Omega_rad)    # deg
    Eps_true_rad = Eps_true*deg2rad 
    cEpsA = math.cos(Eps_true_rad)
    sEpsA = math.sin(Eps_true_rad)
    
    sun_ecliptic_app = R_m*np.array([[math.cos(apparent_long_rad)],
                                     [math.sin(apparent_long_rad)],
                                     [                    0.]])
    
    # r_Equator = R1(-Eps) * r_Ecliptic 
    R1 = np.array([[1.,       0.,       0.],
                   [0.,    cEpsA,   -sEpsA],
                   [0.,    sEpsA,    cEpsA]])
    
    sun_eci_app = np.dot(R1, sun_ecliptic_app)


    
    return sun_eci_geom, sun_eci_app



def ode_twobody(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element array
      cartesian state vector (Inertial Frame)
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 6 element array array
      state derivative vector
    
    '''
    
    # Additional arguments
    GM = params['GM']

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Derivative vector
    dX = np.zeros(6,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3
    
    return dX