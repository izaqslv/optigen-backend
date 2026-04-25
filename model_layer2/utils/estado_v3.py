import numpy as np

def get_estado(c_now, c_prev, threshold_queda=0.002, eps=1e-4):
    grad = c_now - c_prev

    if c_now < 0.005:
        return 3  # depositado

    if grad < -threshold_queda:
        return 2  # queda

    if abs(grad) < eps:
        return 1  # plateau

    return 0  # crescimento

