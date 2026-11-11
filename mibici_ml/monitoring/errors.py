"""Errors raised by the offline monitoring pipeline."""


class MonitoringError(ValueError):
    """Input, window, Evidently output, or output-path contract violation."""
