"""Fixed-cohort validation assertion."""

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)
