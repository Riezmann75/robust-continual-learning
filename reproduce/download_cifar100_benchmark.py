"""Download verified official CIFAR-100 and CIFAR-100-C (stream tar to save disk)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

import numpy as np
from torchvision.datasets import CIFAR100

CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def verify_arrays(target, clean_targets):
    labels = np.load(target/'labels.npy', allow_pickle=False)
    if not np.array_equal(labels, np.tile(clean_targets, 5)):
        raise RuntimeError('Corruption labels do not match clean test ordering')
    for name in CORRUPTIONS:
        array = np.load(target/(name+'.npy'), mmap_mode='r', allow_pickle=False)
        if array.shape != (50000, 32, 32, 3) or array.dtype != np.uint8:
            raise RuntimeError(f'Unexpected array format: {name}')


class HashReader:
    def __init__(self, raw):
        self.raw, self.digest, self.count, self.report = raw, hashlib.md5(), 0, 0

    def read(self, size=-1):
        data = self.raw.read(size)
        self.digest.update(data)
        self.count += len(data)
        if self.count-self.report >= 100_000_000:
            print(f'Downloaded {self.count/1e9:.2f} GB', flush=True)
            self.report = self.count
        return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('experiments/metamix_cifar100/data'))
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    # Official Toronto archive; torchvision checks its published MD5.
    CIFAR100(args.root, train=True, download=True)
    CIFAR100(args.root, train=False, download=True)
    target = args.root/'CIFAR-100-C'
    target.mkdir(exist_ok=True)
    marker = target/'verified.json'
    if marker.exists():
        verify_arrays(target, CIFAR100(args.root, train=False).targets)
        print('Official CIFAR-100-C revalidated', flush=True)
        return
    url = 'https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1'
    expected = '11f0ed0f1191edbf9fa23466ae6021d3'
    names = {name+'.npy' for name in CORRUPTIONS} | {'labels.npy'}
    found = set()
    with urllib.request.urlopen(url, timeout=120) as response:
        reader = HashReader(response)
        with tarfile.open(fileobj=reader, mode='r|*') as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or name not in names:
                    continue
                # Only explicitly allowed basenames are written; no archive paths used.
                with archive.extractfile(member) as src, (target/(name+'.partial')).open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
                found.add(name)
                print('Extracted', name, flush=True)
        while reader.read(1024*1024):
            pass
    if reader.digest.hexdigest()!=expected or found!=names:
        raise RuntimeError(f'Integrity failure: {reader.digest.hexdigest()}, missing {names-found}')
    for name in found:
        (target/(name+'.partial')).replace(target/name)
    clean = CIFAR100(args.root, train=False)
    labels = np.load(target/'labels.npy')
    if not np.array_equal(labels, np.tile(clean.targets,5)):
        raise RuntimeError('Corruption labels do not match clean test ordering')
    for name in CORRUPTIONS:
        array = np.load(target/(name+'.npy'), mmap_mode='r')
        if array.shape!=(50000,32,32,3) or array.dtype!=np.uint8:
            raise RuntimeError(f'Unexpected array format: {name}')
    marker.write_text(json.dumps(dict(url=url, md5=expected, bytes=reader.count,
                                      corruptions=CORRUPTIONS, severities=5),indent=2))
    print('Verified CIFAR-100 and all 15 CIFAR-100-C corruptions / 5 severities', flush=True)


if __name__=='__main__':
    main()
