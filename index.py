import json

from common.ml_args import MLArgs
from common.script_tools import run_script, try_parse_json
from module import ROOT


def main():
    ml_args = MLArgs()

    result = run_script(
        root=ROOT,
        script_name="train.py" if ml_args.run == "train" else "detect.py",
        ml_args=ml_args,
        console_display=ml_args.run == "train"
    )

    result = try_parse_json(result)

    if ml_args.run == "train":
        print(json.dumps({
            "status": "trained",
            "result": result
        }))

    elif ml_args.run == "detect":
        if not isinstance(result, list):
            raise ValueError("detect.py must return a JSON array")

        print(json.dumps(result))

    else:
        raise ValueError(f"Unsupported run mode: {ml_args.run}")


if __name__ == "__main__":
    main()
