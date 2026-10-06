"""Check a generated episode against a released Bench2Dex episode of the same stage.

Every dataset path of the reference must exist in the candidate with the same dtype and the same
shape apart from the frame axis (and the joint axis for robot/action arrays, which depends on the
hand). Object, label and camera sub-trees are compared by structure, not by object id. Exits 1 on
any mismatch.

  python scripts/verify_format.py <candidate.hdf5> <released.hdf5>
"""

import re
import sys

import h5py


def norm(path):
    path = re.sub(r"^metrics/timeseries/(.*?)_(obj|distractor)_.*$", r"metrics/timeseries/\1_<obj>", path)
    path = re.sub(r"^(objects|labels/box3d)/[^/]+", r"\1/<obj>", path)
    path = re.sub(r"^labels/box2d/([^/]+)/[^/]+", r"labels/box2d/\1/<obj>", path)
    return path


def tree(p):
    out = {}
    with h5py.File(p, "r") as h:
        n = len(h["time/sim_step"])

        def v(name, o):
            if isinstance(o, h5py.Dataset):
                shape = tuple("T" if (i == 0 and s == n) else s for i, s in enumerate(o.shape))
                if name.startswith(("robot/q", "action/commanded", "action/action_names", "robot/joint_names")):
                    shape = tuple(x if x == "T" else "J" for x in shape)
                out.setdefault(norm(name), (shape, str(o.dtype)))
            else:
                out.setdefault(norm(name), "group")
        h.visititems(v)
    return out


def main():
    cand, ref = tree(sys.argv[1]), tree(sys.argv[2])
    missing = sorted(k for k in ref if k not in cand and not k.startswith("meta/"))
    extra = sorted(k for k in cand if k not in ref and not k.startswith("meta/"))
    mism = sorted(k for k in ref if k in cand and ref[k] != cand[k] and not k.startswith("meta/"))
    meta_missing = sorted(k for k in ref if k.startswith("meta/") and k not in cand)
    for title, keys in (("missing", missing), ("extra", extra), ("shape/dtype differs", mism), ("meta missing", meta_missing)):
        print(f"{title}: {len(keys)}")
        for k in keys[:40]:
            print(f"   {k}: released {ref.get(k)} | candidate {cand.get(k)}")
    ok = not (missing or mism or meta_missing)
    print("FORMAT OK" if ok else "FORMAT MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
