#!/usr/bin/env python3
"""Make backup-sized copies of the upstream result*.root files.

The raw productions are ~45 MB each and ~97% of that is QA/PID/correlation
histograms this pipeline never opens.  This script copies out only the objects
the Snakemake rules actually read and rewrites them with a stronger compression
setting, which takes data/ from ~1.4 GB to ~0.4 GB (~0.15 GB with --no-lambda)
with zero loss on the histograms that are kept.

    python3 scripts/slim_result_files.py --out backup/data
    python3 scripts/slim_result_files.py --out backup/data --check   # verify

The slim tree is a drop-in replacement for data/ for every rule except
`fit_lambda` when --no-lambda is used.
"""
import argparse
import os
import re
import sys
import time
from glob import glob

import numpy as np
import uproot

SPECIES = r"(piplus|piminus|proton|antiproton|kplus|kminus)"

# Objects the pipeline opens.  Anything not matched here is dropped.
KEEP_CORE = [
    # v2 profiles: integrated, per-pT (plus the _1st/_2nd plane variants), per-y.
    # The _y_pt_ form is the optional 2D histogram coal_preprocess.cpp probes for.
    rf"^h{SPECIES}_(TPC|EPD)_v2(_pt_\d+(_1st|_2nd)?|_y_(pt_)?\d+|)$",
    r"^h(TPCEP_ew_cos|EPDEP_ew_cos_\d+)$",   # event-plane resolution
    r"^hg[A-Za-z]*(_TOF)?_\d+$",             # hgpTeta_/hgpT_/hgp_ (+_TOF): efficiency inputs
    r"^hpT(_TOF)?_\d+$",
    r"^hRefMultCorr_cent_\d+$",              # centrality QA
]
# rule fit_lambda (scripts/fit_v2.py); the strange-baryon names are kept too so the
# same slim tree works if that rule is ever pointed at Xi/Omega.
STRANGE = r"(Lambda|Lambdabar|Xi|Xibar|Omega|Omegabar)"
KEEP_LAMBDA = [
    rf"^h{STRANGE}M_pt_\d+_cen_\d+$",
    rf"^h{STRANGE}_(TPC|EPD)_v2_pt_\d+_cen_\d+$",
]

COMPRESSION = {
    "lzma": lambda: uproot.LZMA(5),   # smallest, slowest to write
    "zlib": lambda: uproot.ZLIB(6),   # what ROOT writes by default
    "lz4": lambda: uproot.LZ4(4),     # fastest to read back
}


def keep_re(with_lambda=True):
    pats = KEEP_CORE + (KEEP_LAMBDA if with_lambda else [])
    return re.compile("|".join(pats))


def slim(src, dst, pattern, compression):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    n = 0
    with uproot.open(src) as fin, uproot.recreate(dst, compression=compression) as fout:
        for key in fin.keys(cycle=False, recursive=False):
            if pattern.match(key):
                fout[key] = fin[key]
                n += 1
    return n


def check(src, dst):
    """Every object in dst must be byte-identical in content to the one in src."""
    bad = []
    with uproot.open(src) as a, uproot.open(dst) as b:
        keys = b.keys(cycle=False, recursive=False)
        for k in keys:
            x, y = a[k], b[k]
            if x.classname != y.classname:
                bad.append((k, "classname"))
                continue
            for m in ("values", "errors"):
                va = np.nan_to_num(getattr(x, m)())
                vb = np.nan_to_num(getattr(y, m)())
                if not np.array_equal(va, vb):
                    bad.append((k, m))
            for member in ("fBinEntries", "fSumw2", "fEntries"):
                try:
                    va, vb = x.member(member), y.member(member)
                except Exception:
                    continue
                if not np.array_equal(np.asarray(va), np.asarray(vb)):
                    bad.append((k, member))
    return len(keys), bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data", help="source tree (default: data)")
    ap.add_argument("--out", required=True, help="destination tree for the slim copies")
    ap.add_argument("--compression", default="lzma", choices=sorted(COMPRESSION))
    ap.add_argument("--no-lambda", action="store_true",
                    help="also drop the Lambda mass/v2 histograms (breaks rule fit_lambda, "
                         "but halves the result again)")
    ap.add_argument("--check", action="store_true",
                    help="verify existing slim copies against the source instead of writing")
    args = ap.parse_args()

    pattern = keep_re(with_lambda=not args.no_lambda)
    files = sorted(glob(os.path.join(args.src, "**", "result*.root"), recursive=True))
    if not files:
        sys.exit(f"no result*.root under {args.src}/")

    tot_src = tot_dst = 0
    failures = 0
    for src in files:
        dst = os.path.join(args.out, os.path.relpath(src, args.src))
        if args.check:
            if not os.path.exists(dst):
                print(f"MISSING  {dst}")
                failures += 1
                continue
            n, bad = check(src, dst)
            print(f"{'OK ' if not bad else 'FAIL'}  {n:5d} objs  {dst}"
                  + (f"   {bad[:3]}" if bad else ""))
            failures += bool(bad)
            continue

        t0 = time.time()
        n = slim(src, dst, pattern, COMPRESSION[args.compression]())
        s_in, s_out = os.path.getsize(src), os.path.getsize(dst)
        tot_src += s_in
        tot_dst += s_out
        flag = "  <-- nothing matched, stale production?" if n == 0 else ""
        print(f"{s_in/1e6:7.1f} -> {s_out/1e6:6.1f} MB  {n:5d} objs  {time.time()-t0:4.0f}s  {dst}{flag}")

    if args.check:
        print(f"\n{len(files)} files checked, {failures} problem(s)")
        sys.exit(1 if failures else 0)
    print(f"\n{len(files)} files: {tot_src/1e9:.2f} GB -> {tot_dst/1e9:.2f} GB "
          f"({tot_dst/tot_src*100:.0f}%)")


if __name__ == "__main__":
    main()
