#!/usr/bin/env python
# -*- coding: utf-8 -*-

sbas_ident = {
    "S31": {
        "system": "WAAS",
        "Azimuth": 216.5-360,  # Adjusted for azimuth range [-180, 180]
        "Elevation": 31.1,
        "PRN": "131",
    },
    "S33": {
        "system": "WAAS",
        "Azimuth": 230.3-360,
        "Elevation": 38.3,
        "PRN": "133",
    },
    "S35": {
        "system": "WAAS",
        "Azimuth": 225.8-360,
        "Elevation": 33.7,
        "PRN": "135",
    },
    "S48": {
        "system": "WAAS",
        "Azimuth": 104.6,
        "Elevation": 8.9,  # too low!
        "PRN": "148",
    },
}

constellation_ident = {
    "G??": "GPS",
    "R??": "GLONASS",
    "E??": "Galileo",
    "C??": "BeiDou",
    "S??": "SBAS",
}