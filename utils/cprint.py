def red(text, bold=False):
    if bold:
        return "\033[1;31m" + str(text) + "\033[0m"
    else:
        return "\033[31m" + str(text) + "\033[0m"


def green(text, bold=False):
    if bold:
        return "\033[1;32m" + str(text) + "\033[0m"
    else:
        return "\033[32m" + str(text) + "\033[0m"


def yellow(text, bold=False):
    if bold:
        return "\033[1;33m" + str(text) + "\033[0m"
    else:
        return "\033[33m" + str(text) + "\033[0m"


def blue(text, bold=False):
    if bold:
        return "\033[1;34m" + str(text) + "\033[0m"
    else:
        return "\033[34m" + str(text) + "\033[0m"


def magenta(text, bold=False):
    if bold:
        return "\033[1;35m" + str(text) + "\033[0m"
    else:
        return "\033[35m" + str(text) + "\033[0m"


def cyan(text, bold=False):
    if bold:
        return "\033[1;36m" + str(text) + "\033[0m"
    else:
        return "\033[36m" + str(text) + "\033[0m"


def white(text, bold=False):
    if bold:
        return "\033[1;37m" + str(text) + "\033[0m"
    else:
        return "\033[37m" + str(text) + "\033[0m"


def gray(text, bold=False):
    if bold:
        return "\033[1;30m" + str(text) + "\033[0m"
    else:
        return "\033[30m" + str(text) + "\033[0m"


def bold(text):
    return "\033[1m" + str(text) + "\033[0m"


def underline(text):
    return "\033[4m" + str(text) + "\033[0m"


def blink(text):
    return "\033[5m" + str(text) + "\033[0m"


def reverse(text):
    return "\033[7m" + str(text) + "\033[0m"


def conceal(text):
    return "\033[8m" + str(text) + "\033[0m"


def strikethrough(text):
    return "\033[9m" + str(text) + "\033[0m"


def black_background(text):
    return "\033[40m" + str(text) + "\033[0m"


def red_background(text):
    return "\033[41m" + str(text) + "\033[0m"


def green_background(text):
    return "\033[42m" + str(text) + "\033[0m"


def yellow_background(text):
    return "\033[43m" + str(text) + "\033[0m"


def blue_background(text):
    return "\033[44m" + str(text) + "\033[0m"


def magenta_background(text):
    return "\033[45m" + str(text) + "\033[0m"


def cyan_background(text):
    return "\033[46m" + str(text) + "\033[0m"


def white_background(text):
    return "\033[47m" + str(text) + "\033[0m"


def default_color(text):
    return "\033[39m" + str(text) + "\033[0m"


def default_background(text):
    return "\033[49m" + str(text) + "\033[0m"


def light_red(text):
    return "\033[91m" + str(text) + "\033[0m"


def light_green(text):
    return "\033[92m" + str(text) + "\033[0m"


def light_yellow(text):
    return "\033[93m" + str(text) + "\033[0m"


def light_blue(text):
    return "\033[94m" + str(text) + "\033[0m"


def light_magenta(text):
    return "\033[95m" + str(text) + "\033[0m"


def light_cyan(text):
    return "\033[96m" + str(text) + "\033[0m"


def clear():
    return "\033[2J"



if __name__ == "__main__":
    print(red("red"))