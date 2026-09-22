"""Post-production modules: Darshan log simplification and metric computation.

These are the production copies of the project's manual post-processing
scripts (previous/), ported unchanged apart from line endings. Each module
keeps its standalone command-line interface and owns its metric algorithms,
CSV columns and graphs. analysis.py orchestrates them as subprocesses:

    python -m darshanflow.post.<module> ...
"""
