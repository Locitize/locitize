"""Session Portal v2 - restructured, testable parallel build.

v2 lives entirely under ``Codebase/v2/`` and does not touch any v1 file.
Same providers, same look, same resume commands - but providers are plugins
behind a :class:`Provider` protocol and the UI never parses session files.
"""

__version__ = "1.0.0"
