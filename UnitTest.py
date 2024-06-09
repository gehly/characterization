import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta

import utilities as util


# LEO Test Case
def setup_leo_sphere_case():
    
    UTC0 = datetime(2024, 6, 6, 12, 0, 0)
    
    # Initial state - approximately sun-synchronous orbit
    a = (6378+700)*1000.
    e = 0.001
    i = 98.
    RAAN = 90.
    AOP = 10.
    TA = 20.
    
    Xo = util.kep2cart([a, e, i, RAAN, AOP, TA])
    
    # Propagate orbit for 24 hours in 10 second increment
    intfcn = util.ode_twobody    
    tvec = np.arange(0., 86401., 10.)
    tin = (tvec[0], tvec[-1])
    method = 'DOP853'
    rtol = 1e-12
    atol = 1e-12
    params = {}
    params['GM'] = util.GME
    
    output = solve_ivp(intfcn,tin,Xo.flatten(),method=method,args=(params,),rtol=rtol,atol=atol,t_eval=tvec)    
    Xout = output['y'].T
    
    
    # Setup sensor - Boulder CO
    lat = 40.01499      # deg
    lon = -105.27055    # deg
    ht = 1655.  # m
    
    sensor_ecef = util.latlonht2ecef(lat, lon, ht)
    
    
    # Loop over times and compute measurements
    # If visible, store and plot output
    
    # Compute initial time in centuries from J2000 for sun ephemeris
    TAI_UTC = 37.                           # offset seconds (for June 6 2024)
    TT_JD0 = util.utcdt2ttjd(UTC0, TAI_UTC)
    TT_cent0 = util.jd2cent(TT_JD0)
    
    # Compute initial Earth Rotation Angle
    UT1_UTC = 0.037525                      # offset seconds (for June 6 2024)
    UT1_JD = util.utcdt2ut1jd(UTC0, UT1_UTC)
    ERA0, dum = util.compute_ERA(UT1_JD)
    
    for ti in tvec:
        
        # Compute current time, rotation, ephemeris data
        TT_cent = TT_cent0 + (ti - tvec[0])/(36525.*86400.)
        ERA = ERA0 + (ti - tvec[0])*util.wE
        sun_eci_geom, sun_eci_app = util.compute_sun_coords(TT_cent)
        
        
    
    
    
    return


if __name__ == '__main__':
    
    
    setup_leo_sphere_case()
    