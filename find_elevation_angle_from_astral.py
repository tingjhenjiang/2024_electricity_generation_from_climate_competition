# https://github.com/sffjunkie/astral/tree/master/src/astral
import datetime
import re
from typing import Optional, Tuple, Union
from enum import Enum
import datetime
from math import asin, atan2, cos, acos, tan, degrees, fabs, pi, radians, sin, sqrt


TimePeriod = Tuple[datetime.datetime, datetime.datetime]
Elevation = Union[float, Tuple[float, float]]
Degrees = float
Radians = float
Minutes = float

def refraction_at_zenith(zenith: float) -> float:
    """Calculate the degrees of refraction of the sun due to the sun's elevation."""

    elevation = 90 - zenith
    if elevation >= 85.0:
        return 0

    refraction_correction = 0.0
    te = tan(radians(elevation))
    if elevation > 5.0:
        refraction_correction = (
            58.1 / te - 0.07 / (te * te * te) + 0.000086 / (te * te * te * te * te)
        )
    elif elevation > -0.575:
        step1 = -12.79 + elevation * 0.711
        step2 = 103.4 + elevation * step1
        step3 = -518.2 + elevation * step2
        refraction_correction = 1735.0 + elevation * step3
    else:
        refraction_correction = -20.774 / te

    refraction_correction = refraction_correction / 3600.0

    return refraction_correction


def dms_to_float(
    dms: Union[str, float, Elevation], limit: Optional[float] = None
) -> float:
    """Converts as string of the form `degrees°minutes'seconds"[N|S|E|W]`,
    or a float encoded as a string, to a float

    N and E return positive values
    S and W return negative values

    Args:
        dms: string to convert
        limit: Limit the value between ± `limit`

    Returns:
        The number of degrees as a float
    """

    try:
        res = float(dms)  # type: ignore
    except (ValueError, TypeError) as exc:
        _dms_re = r"(?P<deg>\d{1,3})[°]((?P<min>\d{1,2})[′'])?((?P<sec>\d{1,2})[″\"])?(?P<dir>[NSEW])?"  # noqa
        dms_match = re.match(_dms_re, str(dms), flags=re.IGNORECASE)
        if dms_match:
            deg = dms_match.group("deg") or 0.0
            min_ = dms_match.group("min") or 0.0
            sec = dms_match.group("sec") or 0.0
            dir_ = dms_match.group("dir") or "E"

            res = float(deg)
            if min_:
                res += float(min_) / 60
            if sec:
                res += float(sec) / 3600

            if dir_.upper() in ["S", "W"]:
                res = -res
        else:
            raise ValueError(
                "Unable to convert degrees/minutes/seconds to float"
            ) from exc

    if limit is not None:
        if res > limit:
            res = limit
        elif res < -limit:
            res = -limit

    return res

class Observer:
    """Defines the location of an observer on Earth.

    Latitude and longitude can be set either as a float or as a string.
    For strings they must be of the form

        degrees°minutes'seconds"[N|S|E|W] e.g. 51°31'N

    `minutes’` & `seconds”` are optional.

    Elevations are either

    * A float that is the elevation in metres above a location, if the nearest
      obscuring feature is the horizon
    * or a tuple of the elevation in metres and the distance in metres to the
      nearest obscuring feature.

    Args:
        latitude:   Latitude - Northern latitudes should be positive
        longitude:  Longitude - Eastern longitudes should be positive
        elevation:  Elevation and/or distance to nearest obscuring feature
                    in metres above/below the location.
    """

    latitude: Degrees = 51.4733
    longitude: Degrees = -0.0008333
    elevation: Elevation = 0.0

    def __setattr__(self, name: str, value: Union[str, float, Elevation]):
        if name == "latitude":
            value = dms_to_float(value, 90.0)
        elif name == "longitude":
            value = dms_to_float(value, 180.0)
        elif name == "elevation":
            if isinstance(value, tuple):
                value = (float(value[0]), float(value[1]))
            else:
                value = float(value)
        super().__setattr__(name, value)

class Calendar(Enum):
    GREGORIAN = 1
    JULIAN = 2

def julianday(
    at: Union[datetime.datetime, datetime.date], calendar: Calendar = Calendar.GREGORIAN
) -> float:
    """Calculate the Julian Day (number) for the specified date/time

    julian day numbers for dates are calculated for the start of the day
    """

    def _time_to_seconds(t: datetime.time) -> int:
        return int(t.hour * 3600 + t.minute * 60 + t.second)

    year = at.year
    month = at.month
    day = at.day
    day_fraction = 0.0
    if isinstance(at, datetime.datetime):
        t = _time_to_seconds(at.time())
        day_fraction = t / (24 * 60 * 60)
    else:
        day_fraction = 0.0

    if month <= 2:
        year -= 1
        month += 12

    a = int(year / 100)
    if calendar == Calendar.GREGORIAN:
        b = 2 - a + int(a / 4)
    else:
        b = 0
    jd = (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + day_fraction
        + b
        - 1524.5
    )

    return jd

def julianday_to_juliancentury(julianday: float) -> float:
    """Convert a Julian Day number to a Julian Century"""
    return (julianday - 2451545.0) / 36525.0

def sun_eq_of_center(juliancentury: float) -> float:
    """Calculate the equation of the center of the sun"""
    m = geom_mean_anomaly_sun(juliancentury)

    mrad = radians(m)
    sinm = sin(mrad)
    sin2m = sin(mrad + mrad)
    sin3m = sin(mrad + mrad + mrad)

    c = (
        sinm * (1.914602 - juliancentury * (0.004817 + 0.000014 * juliancentury))
        + sin2m * (0.019993 - 0.000101 * juliancentury)
        + sin3m * 0.000289
    )

    return c

def sun_true_long(juliancentury: float) -> float:
    """Calculate the sun's true longitude"""
    l0 = geom_mean_long_sun(juliancentury)
    c = sun_eq_of_center(juliancentury)

    return l0 + c

def sun_apparent_long(juliancentury: float) -> float:
    true_long = sun_true_long(juliancentury)

    omega = 125.04 - 1934.136 * juliancentury
    return true_long - 0.00569 - 0.00478 * sin(radians(omega))

def sun_declination(juliancentury: float) -> float:
    """Calculate the sun's declination"""
    e = obliquity_correction(juliancentury)
    lambd = sun_apparent_long(juliancentury)

    sint = sin(radians(e)) * sin(radians(lambd))
    return degrees(asin(sint))

def geom_mean_long_sun(juliancentury: float) -> float:
    """Calculate the geometric mean longitude of the sun"""
    l0 = 280.46646 + juliancentury * (36000.76983 + 0.0003032 * juliancentury)
    return l0 % 360.0

def eccentric_location_earth_orbit(juliancentury: float) -> float:
    """Calculate the eccentricity of Earth's orbit"""
    return 0.016708634 - juliancentury * (0.000042037 + 0.0000001267 * juliancentury)

def geom_mean_anomaly_sun(juliancentury: float) -> float:
    """Calculate the geometric mean anomaly of the sun"""
    return 357.52911 + juliancentury * (35999.05029 - 0.0001537 * juliancentury)

def mean_obliquity_of_ecliptic(juliancentury: float) -> float:
    seconds = 21.448 - juliancentury * (
        46.815 + juliancentury * (0.00059 - juliancentury * (0.001813))
    )
    return 23.0 + (26.0 + (seconds / 60.0)) / 60.0

def obliquity_correction(juliancentury: float) -> float:
    e0 = mean_obliquity_of_ecliptic(juliancentury)

    omega = 125.04 - 1934.136 * juliancentury
    return e0 + 0.00256 * cos(radians(omega))

def var_y(juliancentury: float) -> float:
    epsilon = obliquity_correction(juliancentury)
    y = tan(radians(epsilon) / 2.0)
    return y * y

def eq_of_time(juliancentury: float) -> Minutes:
    l0 = geom_mean_long_sun(juliancentury)
    e = eccentric_location_earth_orbit(juliancentury)
    m = geom_mean_anomaly_sun(juliancentury)

    y = var_y(juliancentury)

    sin2l0 = sin(2.0 * radians(l0))
    sinm = sin(radians(m))
    cos2l0 = cos(2.0 * radians(l0))
    sin4l0 = sin(4.0 * radians(l0))
    sin2m = sin(2.0 * radians(m))

    Etime = (
        y * sin2l0
        - 2.0 * e * sinm
        + 4.0 * e * y * sinm * cos2l0
        - 0.5 * y * y * sin4l0
        - 1.25 * e * e * sin2m
    )

    return degrees(Etime) * 4.0

def zenith_and_azimuth(
    observer: Observer,
    dateandtime: datetime.datetime,
    with_refraction: bool = True,
) -> Tuple[float, float]:
    if observer.latitude > 89.8:
        latitude = 89.8
    elif observer.latitude < -89.8:
        latitude = -89.8
    else:
        latitude = observer.latitude

    longitude = observer.longitude

    if dateandtime.tzinfo is None:
        zone = 0.0
        utc_datetime = dateandtime
    else:
        zone = -dateandtime.utcoffset().total_seconds() / 3600.0  # type: ignore
        utc_datetime = dateandtime.astimezone(datetime.timezone.utc)

    jd = julianday(utc_datetime)
    t = julianday_to_juliancentury(jd)
    declination = sun_declination(t)
    eqtime = eq_of_time(t)

    # 360deg * 4 == 1440 minutes, 60*24 = 1440 minutes == 1 rotation
    solarTimeFix = eqtime + (4.0 * longitude) + (60 * zone)
    trueSolarTime = (
        dateandtime.hour * 60.0
        + dateandtime.minute
        + dateandtime.second / 60.0
        + solarTimeFix
    )
    #    in minutes as a float, fractional part is seconds

    while trueSolarTime > 1440:
        trueSolarTime = trueSolarTime - 1440

    hourangle = trueSolarTime / 4.0 - 180.0
    #    Thanks to Louis Schwarzmayr for the next line:
    if hourangle < -180:
        hourangle = hourangle + 360.0

    ch = cos(radians(hourangle))
    # sh = sin(radians(hourangle))
    cl = cos(radians(latitude))
    sl = sin(radians(latitude))
    sd = sin(radians(declination))
    cd = cos(radians(declination))

    csz = cl * cd * ch + sl * sd

    if csz > 1.0:
        csz = 1.0
    elif csz < -1.0:
        csz = -1.0

    zenith = degrees(acos(csz))

    azDenom = cl * sin(radians(zenith))

    if abs(azDenom) > 0.001:
        azRad = ((sl * cos(radians(zenith))) - sd) / azDenom

        if abs(azRad) > 1.0:
            if azRad < 0:
                azRad = -1.0
            else:
                azRad = 1.0

        azimuth = 180.0 - degrees(acos(azRad))

        if hourangle > 0.0:
            azimuth = -azimuth
    else:
        if latitude > 0.0:
            azimuth = 180.0
        else:
            azimuth = 0.0

    if azimuth < 0.0:
        azimuth = azimuth + 360.0

    if with_refraction:
        zenith -= refraction_at_zenith(zenith)
        # elevation = 90 - zenith

    return zenith, azimuth

def now(tz: Optional[datetime.tzinfo] = None) -> datetime.datetime:
    """Returns the current time in the specified time zone"""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if tz is None:
        return now_utc

    return now_utc.astimezone(tz)

def zenith(
    observer: Observer,
    dateandtime: Optional[datetime.datetime] = None,
    with_refraction: bool = True,
) -> float:
    """Calculate the zenith angle of the sun.

    Args:
        observer:    Observer to calculate the solar zenith for
        dateandtime: The date and time for which to calculate the angle.
                     If `dateandtime` is None or is a naive Python datetime
                     then it is assumed to be in the UTC timezone.
        with_refraction: If True adjust zenith to take refraction into account

    Returns:
        The zenith angle in degrees.
    """

    if dateandtime is None:
        dateandtime = now(datetime.timezone.utc)

    return zenith_and_azimuth(observer, dateandtime, with_refraction)[0]

def elevation(
    observer: Observer,
    dateandtime: Optional[datetime.datetime] = None,
    with_refraction: bool = True,
) -> float:
    """Calculate the sun's angle of elevation.

    Args:
        observer:    Observer to calculate the solar elevation for
        dateandtime: The date and time for which to calculate the angle.
                     If `dateandtime` is None or is a naive Python datetime
                     then it is assumed to be in the UTC timezone.
        with_refraction: If True adjust elevation to take refraction into account

    Returns:
        The elevation angle in degrees above the horizon.
    """

    if dateandtime is None:
        dateandtime = now(datetime.timezone.utc)

    return 90.0 - zenith(observer, dateandtime, with_refraction)