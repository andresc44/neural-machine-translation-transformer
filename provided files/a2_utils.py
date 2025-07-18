import argparse
import gzip
from pathlib import Path
from typing import Union, TextIO


def smart_open(path: Path, mode: str = "r") -> Union[gzip.GzipFile, TextIO]:
    if path.suffix == ".gz":  # error?  TODO: check if this is correct
        open_ = gzip.open
        if mode[-1] != "b":
            mode += "t"
    else:
        open_ = open
    try:
        f = open_(path, mode=mode)
    except OSError as e:
        raise argparse.ArgumentTypeError(f"can't open '{path}': {e}")
    return f


def schedule_rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
