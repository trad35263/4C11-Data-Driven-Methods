# import modules
import numpy as np
import matplotlib.pyplot as plt

# create Colour class
class Colours:
    """Stores several ASCII colour codes for terminal outputs."""
    # store ASCII codes for selected colours as class attributes
    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    PINK = '\033[38;5;212m'
    GREY = '\033[90m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    END = '\033[0m'
