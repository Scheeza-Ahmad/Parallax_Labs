"""Download an immutable Wikipedia snapshot and select whole source articles."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

REVISION = 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa'
URL = f'https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{REVISION}/20231101.simple/train-00000-of-00001.parquet'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=7000)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    import pyarrow.parquet as pq
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    archive = args.raw_dir / 'wikipedia-20231101-simple.parquet'
    target = args.raw_dir / f'wikipedia-{args.limit}.jsonl'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    if not archive.exists():
        partial = archive.with_suffix('.download')
        print('Downloading immutable snapshot...', flush=True)
        request = urllib.request.Request(URL, headers={'User-Agent': 'ParallaxLabsWeek1/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('xb') as output:
            shutil.copyfileobj(response, output)
        partial.rename(archive)
    parquet = pq.ParquetFile(archive)
    if parquet.metadata.num_rows < args.limit:
        raise ValueError('Snapshot has fewer records than requested')
    count = 0
    selection_partial = target.with_suffix('.partial')
    with selection_partial.open('x', encoding='utf-8') as output:
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if count >= args.limit:
                    break
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
            if count >= args.limit:
                break
    selection_partial.rename(target)
    digest = hashlib.sha256()
    with archive.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    manifest = {'dataset': 'wikimedia/wikipedia', 'config': '20231101.simple',
                'revision': REVISION, 'download_url': URL,
                'acquired_at_utc': datetime.now(timezone.utc).isoformat(),
                'license': 'CC-BY-SA-3.0 and GFDL (dataset card)',
                'attribution': 'Wikipedia contributors; individual article URLs retained',
                'selection': 'First N whole articles in immutable parquet row order; no random sampling',
                'selected_records': count, 'snapshot_records': parquet.metadata.num_rows,
                'archive_sha256': digest.hexdigest(), 'raw_jsonl': target.as_posix()}
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    Path('reports').mkdir(exist_ok=True)
    Path('reports/acquisition.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
