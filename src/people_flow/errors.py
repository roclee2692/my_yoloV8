"""Project-specific exceptions."""


class PeopleFlowError(Exception):
    """Base class for recoverable People Flow application errors."""


class ConfigurationError(PeopleFlowError):
    """Raised when a configuration file is missing or invalid."""


class FeatureNotAvailableError(PeopleFlowError):
    """Raised when a command belongs to a later implementation phase."""
