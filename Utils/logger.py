import logging

# Define the common logging format
LOGGING_FORMAT = "%(asctime)s %(levelname)s (%(module)s:%(lineno)d): %(message)s"
# SHORT_LOGGING_FORMAT = "%(levelname)s: %(message)s"
TIME_DATE_FORMAT = "%H:%M:%S"

# Set up the logger
logger = logging.getLogger()

# Create logging formatter
_formatter = logging.Formatter(LOGGING_FORMAT, TIME_DATE_FORMAT)

# Add standard output logging
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)

# Set logging level
logger.setLevel(logging.INFO)
