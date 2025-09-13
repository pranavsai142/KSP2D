import VectorMath as vecm
import datetime
from skyfield.api import load
import math

# Constants
G = 6.674e-11
EARTH_AVG_RADIUS = 6371 * 1000
EARTH_A_RADIUS = 6378.135 * 1000
EARTH_FLATTEN_CONST = 1/298.257
EARTH_MASS = 5.972e24
EARTH_STANDARD_GRAV = G * EARTH_MASS
EARTH_GRAV_SEA = EARTH_STANDARD_GRAV / (EARTH_AVG_RADIUS**2)
EARTH_GRAV_AVG = 9.80665

MOON_AVG_RADIUS = 1737.4 * 1000
MOON_A_RADIUS = 1738.1 * 1000
MOON_FLATTEN_CONST = 0.0012
MOON_MASS = .07346e24
MOON_STANDARD_GRAV = G * MOON_MASS
MOON_GRAV_AVG = 1.625

VON_KARMAN_LINE = 100 * 1000

MOON_DENSITY = 1 * 10**(-15)
MARS_DENSITY = 0.020

delta_t = datetime.timedelta(microseconds=10000)
hstep = datetime.timedelta(seconds=10)
hstep_total_seconds = hstep.total_seconds()
simulationduration = 10000

	
ECI_X = vecm.vector(1,0,0)
ECI_Y = vecm.vector(0,1,0)
ECI_Z = vecm.vector(0,0,1)

# kg/m^3
TITANIUM_DENSITY = 4506
LIQUID_OXYGEN_DENSITY = 1141
LIQUID_METHANE_DENSITY = 422.8

planets = load('de421.bsp')
earth = planets['earth']
moon = planets['moon']
sun = planets['sun']
ts = load.timescale()

RCS_TIME_BUFFER = datetime.timedelta(seconds = 120)

# Return the air density, given the altitude in meters
def getDensity(altitude):
    """
    Return the air density (kg/m^3) given the altitude in meters.
    
    Args:
        altitude (float): Altitude above sea level in meters.
        
    Returns:
        float: Air density in kg/m^3.
    """
    # Constants for the troposphere
    RHO_0 = 1.225  # Sea-level air density (kg/m^3)
    H_TROPOSPHERE = 8400  # Scale height for troposphere (m)
    TROPOPAUSE_ALT = 11000  # Troposphere-stratosphere boundary (m)
    H_STRATOSPHERE = 10000  # Scale height for stratosphere (approximate, m)
    
    # Handle negative altitudes (assume sea-level density)
    if altitude < 0:
        return RHO_0
    
    # Troposphere (0 to 11,000 m)
    if altitude <= TROPOPAUSE_ALT:
        return RHO_0 * math.exp(-altitude / H_TROPOSPHERE)
    
    # Stratosphere and above (>11,000 m)
    # Continue exponential decay with a different scale height
    # Use density at tropopause as reference
    rho_tropopause = RHO_0 * math.exp(-TROPOPAUSE_ALT / H_TROPOSPHERE)
    return rho_tropopause * math.exp(-(altitude - TROPOPAUSE_ALT) / H_STRATOSPHERE)
    