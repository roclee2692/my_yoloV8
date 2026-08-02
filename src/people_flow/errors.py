"""Project-specific exceptions."""


class PeopleFlowError(Exception):
    """Base class for recoverable People Flow application errors."""


class ConfigurationError(PeopleFlowError):
    """Raised when a configuration file is missing or invalid."""


class FeatureNotAvailableError(PeopleFlowError):
    """Raised when a command belongs to a later implementation phase."""


class AssetDownloadError(PeopleFlowError):
    """Raised when an official model or sample asset cannot be verified."""


class VideoSourceError(PeopleFlowError):
    """Raised when a video cannot be opened or has invalid metadata."""


class MotFormatError(PeopleFlowError):
    """Raised when a MOT sequence is missing data or violates the MOT format."""


class MotConversionError(PeopleFlowError):
    """Raised when a MOT image sequence cannot be converted without frame loss."""


class CountingError(PeopleFlowError):
    """Raised when counting geometry, state, or event evidence is invalid."""


class ModelInitializationError(PeopleFlowError):
    """Raised when the requested model or compute device cannot initialize."""


class PipelineError(PeopleFlowError):
    """Raised when a pipeline run cannot safely continue."""


class OutputError(PeopleFlowError):
    """Raised when an output artifact cannot be created."""
