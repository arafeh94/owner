def train_ok(model_path: str, **args):
    return {
        "status": "ok",
        "model_path": model_path,
        **args
    }


def train_failed(error, **args):
    return {
        "status": "failed",
        "error": error,
        **args
    }
