import argparse
import os
import platform
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect and probe the local Torch GPU runtime.")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run a small tensor allocation and conv forward/backward pass on the selected device.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to probe. Defaults to cuda.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"platform={platform.platform()}")
    print(f"python={sys.version.splitlines()[0]}")
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")
    print(f"POKEMON_TRAIN_DEVICE={os.environ.get('POKEMON_TRAIN_DEVICE')}")

    import torch

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"device[{i}]={torch.cuda.get_device_name(i)}")

    if not args.probe:
        return

    print(f"probe_device={args.device}")
    start = time.time()
    x = torch.randn(2, 4, 84, 84, device=args.device, requires_grad=True)
    print(f"alloc_ok elapsed={time.time() - start:.2f}s")

    start = time.time()
    model = torch.nn.Conv2d(4, 8, 3).to(args.device)
    y = model(x).sum()
    print(f"forward_ok elapsed={time.time() - start:.2f}s")

    start = time.time()
    y.backward()
    print(f"backward_ok elapsed={time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
