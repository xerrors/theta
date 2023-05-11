def safe_list_get(l, idx, default=None):
    """https://stackoverflow.com/a/5125636/18655723"""
    try:
        return l[idx]
    except IndexError:
        return default

import math
def cosine_ease_in_out_minmax(cur, max_step):
    x = min(cur / (max_step + 1e-8), 1)
    return 0.5 * (1 - math.cos(x * math.pi))