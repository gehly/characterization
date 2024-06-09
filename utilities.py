import numpy as np
import math


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
Re = 6378137.0   # km
rec_f = 298.257223563

# Light and Flux Parameters
c_light = 299792458.  # m/s
AU_m = 149597870700.  # m
SF = 1367.  # W/m^2 = kg/s^3
C_sunvis = 455.   # W/m^2 = kg/s^3



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