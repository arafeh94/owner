def train_ok(**args):
    return {
        "status": "ok",
        **args
    }


def train_failed(error, **args):
    return {
        "status": "failed",
        "error": error,
        **args
    }
