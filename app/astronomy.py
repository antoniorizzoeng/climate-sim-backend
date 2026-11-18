"""UTC astronomical twilight and airless topocentric lunar positions."""

from datetime import UTC, timedelta

import astronomy as astro


def ephemeris_time(at):
    return astro.Time.Make(
        at.year, at.month, at.day, at.hour, at.minute, at.second + at.microsecond / 1e6
    )


def altitude(body, observer, time):
    eq = astro.Equator(body, time, observer, ofdate=True, aberration=True)
    return astro.Horizon(
        time, observer, eq.ra, eq.dec, astro.Refraction.Airless
    ).altitude


def event_minute(event):
    # Publish events to the nearest UTC minute. This also lets a fixed-step
    # transport solve terminate at dawn without sub-second integration steps.
    return (event.Utc().replace(tzinfo=UTC) + timedelta(seconds=30)).replace(
        second=0, microsecond=0
    )


def night_at(start, lat, lon, hours):
    observer = astro.Observer(lat, lon, 0)
    time = ephemeris_time(start)
    dark = altitude(astro.Body.Sun, observer, time) < -18
    # At polar sites, the bounding events can be months apart.
    dusk = astro.SearchAltitude(
        astro.Body.Sun,
        observer,
        astro.Direction.Set,
        time,
        -370 if dark else hours / 24,
        -18,
    )
    if dusk is None:
        return None
    dawn = astro.SearchAltitude(
        astro.Body.Sun, observer, astro.Direction.Rise, time if dark else dusk, 370, -18
    )
    begin, end = event_minute(dusk), event_minute(dawn)
    if dark and end <= start:
        # Keep the remaining seconds of a night that ends in this minute.
        end = dawn.Utc().replace(tzinfo=UTC, microsecond=0) + timedelta(seconds=1)
    moon = []
    at = max(start, begin)
    while at <= end:
        time = ephemeris_time(at)
        moon.append(
            dict(
                time=at.isoformat(),
                altitude_deg=altitude(astro.Body.Moon, observer, time),
                illumination=astro.Illumination(astro.Body.Moon, time).phase_fraction,
            )
        )
        at += timedelta(minutes=10)
    return dict(dusk=begin.isoformat(), dawn=end.isoformat(), moon=moon)
