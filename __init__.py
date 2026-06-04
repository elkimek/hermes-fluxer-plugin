def register(ctx):
    """Register the Fluxer platform with Hermes Agent."""
    try:
        from .adapter import register as _register
    except ImportError:
        from adapter import register as _register

    return _register(ctx)


__all__ = ["register"]
