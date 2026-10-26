import logging

try:
    from fibsem.applications.salami.ui.SalamiUI import SalamiUI
except ImportError as e:
    logging.info(f"Error importing salami.ui: {e}, using dummy instead.")

    SalamiUI = None
