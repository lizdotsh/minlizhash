def progress_bar(i, mx, bar_length=20) -> str:
    """Generates an ASCII progress bar."""
    percent = float(i) / mx
    hashes = "#" * int(round(percent * bar_length))
    spaces = " " * (bar_length - len(hashes))
    return f"[{hashes}{spaces}] {i} of {mx}"
