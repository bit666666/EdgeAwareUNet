"""Download and verify the ten public historical state dictionaries."""
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'docs' / 'WEIGHTS.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        target = root / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']:
            print('Verified existing ' + item['path'])
            continue
        temporary = target.with_suffix('.download')
        digest = hashlib.sha256()
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'EdgeAwareUNet-weight-download'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != item['sha256']:
                raise RuntimeError('Checksum mismatch for ' + item['path'])
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        print('Downloaded and verified ' + item['path'])


if __name__ == '__main__':
    main()
