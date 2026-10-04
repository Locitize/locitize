# LOCITIZE platform package marker.
#
# This file exists so the flat milestone-1 modules (launcher, config, health,
# ...) can later be promoted to an installable package (locitize_platform)
# without moving files. In milestone 1 the modules are imported as flat
# top-level modules with the working directory set to this folder
# (see Architecture section 1). No package-level code runs at import time.

__all__ = []
