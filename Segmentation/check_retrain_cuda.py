"""Small CUDA check inside the allocated job, before checkpoint loading."""
import json
import os
import socket
import subprocess
import sys

import torch


def main():
    print(json.dumps(dict(host=socket.gethostname(), python=sys.executable,
                          torch_version=torch.__version__, torch_path=torch.__file__,
                          cuda_build=torch.version.cuda,
                          cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), indent=2), flush=True)
    try:
        subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                       check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        print('nvidia-smi unavailable:', error, flush=True)
    if torch.__version__ != '2.3.1+cu118' or torch.version.cuda != '11.8':
        raise RuntimeError('Unexpected PyTorch build. Run bash setup_retrain_env.sh and use its venv.')
    try:
        # Check actual kernel execution, not just device enumeration. This is not training.
        torch.cuda.init()
        conv = torch.nn.Conv2d(3, 4, 3, padding=1).cuda()
        x = torch.randn(2, 3, 16, 16, device='cuda', requires_grad=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y = conv(x)
        y.float().square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(y).all() or not torch.isfinite(x.grad).all():
            raise RuntimeError('Non-finite output/gradient in the CUDA check')
    except Exception as error:
        raise RuntimeError('CUDA startup failed on {} before loading or modifying any checkpoint. '
                           'Check the driver/runtime with the cluster administrator. Cause: {}'.format(
                               socket.gethostname(), error)) from error
    print('CUDA check passed on {}. Training can now load its checkpoint.'.format(
        torch.cuda.get_device_name()), flush=True)


if __name__ == '__main__':
    main()
