def safe_list_get(l, idx, default=None):
    """https://stackoverflow.com/a/5125636/18655723"""
    try:
        return l[idx]
    except IndexError:
        return default
