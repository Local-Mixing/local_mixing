"""Keyed integrity tags for a TDP run (docs/STAGE4_RUST_DESIGN.md §2.3, §8).

usage: python3 -I run_mac.py SEEDS NAME=PATH ...      -> "mac.NAME=<hex>" per argument
       python3 -I run_mac.py SEEDS --seeds            -> "seeds_mac=<hex>"

HMAC-SHA256 keyed by the run's private 256-bit run_key (read from the private
SEEDS file, never from argv). No unkeyed hash of C, of a circuit or of a seed
is ever written; these tags detect a swapped or edited input on resume
without publishing a brute-forceable fingerprint. Nothing secret is printed.
"""
import hashlib
import hmac
import sys


def run_key(path):
    keys = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("=")
            keys[k] = v
    key = keys.get("run_key", "")
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        sys.exit("run_mac: malformed run_key in SEEDS")
    return bytes.fromhex(key), keys


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    key, keys = run_key(sys.argv[1])
    if sys.argv[2] == "--seeds":
        msg = "gen_seed=%s\nmix_seed=%s\n" % (keys["gen_seed"], keys["mix_seed"])
        print("seeds_mac=" + hmac.new(key, msg.encode(), hashlib.sha256).hexdigest())
        return
    for arg in sys.argv[2:]:
        name, _, path = arg.partition("=")
        h = hmac.new(key, b"tdp-file\n", hashlib.sha256)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        print("mac.%s=%s" % (name, h.hexdigest()))


if __name__ == "__main__":
    main()
