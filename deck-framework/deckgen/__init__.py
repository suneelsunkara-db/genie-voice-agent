"""deckgen - a small, dependency-light framework for generating branded PPTX
decks from a declarative content spec + a brand theme, layered on top of an
existing .pptx template (its masters / layouts / theme / fonts are reused).

Public surface:
    from deckgen.engine import Theme, Engine, load_theme, package, validate
    from deckgen.slides import render_slides, SLIDE_TYPES
"""
__all__ = ["engine", "slides", "build"]
__version__ = "0.1.0"
