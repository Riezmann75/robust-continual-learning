"""Download CIFAR-10 and MD5-verified CIFAR-10-C with safe basename extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from torchvision.datasets import CIFAR10
from download_cifar100_benchmark import CORRUPTIONS, HashReader, verify_arrays

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('experiments/metamix_cifar10/data'))
    p.add_argument('--corruption-url', default='https://huggingface.co/datasets/torch-uncertainty/CIFAR-C/resolve/main/CIFAR-10-C.tar')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    CIFAR10.url = 'https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/main/cifar-10-python.tar.gz'
    CIFAR10(a.root, train=True, download=True)
    clean = CIFAR10(a.root, train=False, download=True)
    target = a.root/'CIFAR-10-C'
    target.mkdir(exist_ok=True)
    marker = target/'verified.json'
    expected = '56bf5dcef84df0e2308c6dcbcbbd8499'
    if marker.exists():
        record = json.loads(marker.read_text())
        assert record['md5'] == expected
        verify_arrays(target, clean.targets)
        for name, digest in record['sha256'].items():
            with (target/name).open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == digest, name
        print('Existing CIFAR-10-C reverified', flush=True)
        return
    names = {c+'.npy' for c in CORRUPTIONS} | {'labels.npy'}
    found = set()
    with urllib.request.urlopen(a.corruption_url, timeout=120) as response:
        reader = HashReader(response)
        with tarfile.open(fileobj=reader, mode='r|*') as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or name not in names:
                    continue
                with archive.extractfile(member) as src, (target/(name+'.partial')).open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024*1024)
                found.add(name)
                print('Extracted', name, flush=True)
        while reader.read(1024*1024):
            pass
    assert reader.digest.hexdigest() == expected, reader.digest.hexdigest()
    assert found == names, names-found
    for name in sorted(names):
        (target/(name+'.partial')).replace(target/name)
    verify_arrays(target, clean.targets)
    hashes = {}
    for name in sorted(names):
        with (target/name).open('rb') as f:
            hashes[name] = hashlib.file_digest(f, 'sha256').hexdigest()
    marker.write_text(json.dumps(dict(url=a.corruption_url, md5=expected,
        bytes=reader.count, sha256=hashes, corruptions=CORRUPTIONS, severities=5), indent=2))
    print('Verified clean CIFAR-10 and all 15 corruptions x 5 severities', flush=True)

if __name__ == '__main__':
    main()


