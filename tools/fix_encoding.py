#!/usr/bin/env python3
"""Re-encode UTF-16 and mixed-encoding text files as UTF-8 with LF endings.

    python3 tools/fix_encoding.py --check     # report, change nothing
    python3 tools/fix_encoding.py             # rewrite in place

PowerShell's `>` redirect writes UTF-16LE, so files created that way reach git as
binary blobs: no diff, no merge, no blame, and Python refuses to import them
("source code string cannot contain null bytes"). Some files are worse than
UTF-16 -- a UTF-8 body with one UTF-16 line appended, which no single decoder reads.
"""
import argparse
import pathlib
import sys

SKIP_SUFFIX = {'.png', '.jpg', '.jpeg', '.pdf', '.pyc', '.npz', '.pgm'}
SKIP_DIR = {'.git', '.venv', 'node_modules', '__pycache__'}


def decode(raw):
    """Return (text, how) or raise ValueError if nothing reads it cleanly."""
    if raw[:2] == b'\xff\xfe':
        return raw[2:].decode('utf-16-le'), 'utf-16-le'
    if raw[:3] == b'\xef\xbb\xbf':
        return raw[3:].decode('utf-8'), 'utf-8-bom'
    i = raw.find(b'\x00')
    if i < 0:
        return raw.decode('utf-8'), 'utf-8'
    # Mixed. The UTF-16 run starts one byte before the first NUL: that byte is the
    # low half of its first character.
    head, tail = raw[:i - 1], raw[i - 1:]
    return head.decode('utf-8') + tail.decode('utf-16-le'), 'mixed utf-8 + utf-16-le'


def walk(root):
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() in SKIP_SUFFIX:
            continue
        if SKIP_DIR & set(p.parts):
            continue
        yield p


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('root', nargs='?', default='.')
    o = ap.parse_args(argv)

    changed, failed = [], []
    for p in walk(pathlib.Path(o.root)):
        raw = p.read_bytes()
        try:
            text, how = decode(raw)
        except (UnicodeDecodeError, ValueError) as e:
            failed.append((p, e))
            continue
        if how == 'utf-8' and b'\r' not in raw:
            continue
        out = text.replace('\r\n', '\n').replace('\r', '\n').encode('utf-8')
        if out == raw:
            continue
        changed.append((p, how, len(raw), len(out)))
        if not o.check:
            p.write_bytes(out)

    for p, how, a, b in changed:
        print(f'  {"would fix" if o.check else "fixed"}  {p}  [{how}]  {a} -> {b} B')
    for p, e in failed:
        print(f'  UNREADABLE  {p}: {e}', file=sys.stderr)
    print(f'{len(changed)} file(s) {"need fixing" if o.check else "rewritten"}; '
          f'{len(failed)} unreadable')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
