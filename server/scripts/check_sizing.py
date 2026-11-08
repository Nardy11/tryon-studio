"""Unit checks for the size rule (no GPU, no photos):  cd server; .venv\\Scripts\\python scripts\\check_sizing.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sizing import CHEST, WAIST, label, size_of, verdict  # noqa: E402

XS, S, M, L, XL = range(5)
fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, "->", got, "" if ok else f"(want {want})")


# Size chart boundaries (chest, cm). Within 2 cm below a boundary rounds up ("between").
check("chest 85 = XS", size_of(85, CHEST), (XS, False))
check("chest 87 = between XS/S -> S", size_of(87, CHEST), (S, True))
check("chest 100 = M", size_of(100, CHEST), (M, False))
check("chest 111 = between L/XL -> XL", size_of(111, CHEST), (XL, True))
check("chest 117 = XL", size_of(117, CHEST)[0], XL)
check("chest 125 = XL + 1", label(size_of(125, CHEST)[0]), "XL + 1")
check("chest 132 = XL + 2", label(size_of(132, CHEST)[0]), "XL + 2")
check("chest 135 = between XL + 2 / XL + 3 -> XL + 3", label(size_of(135, CHEST)[0]), "XL + 3")
check("waist 79 = M", size_of(79, WAIST)[0], M)

# The agreed verdict table.
check("bigger than XL (+2), stock XS-L -> too small, shown in L", verdict(XL + 2, [XS, S, M, L]),
      {"verdict": "too_small", "needed": "XL + 2", "worn": "L", "off": -3})
check("S, stock XL only -> too big, shown in XL", verdict(S, [XL]),
      {"verdict": "too_big", "needed": "S", "worn": "XL", "off": 3})
check("M, stock S-XL -> fits", verdict(M, [S, M, L, XL]), {"verdict": "fits", "needed": "M", "worn": "M", "off": 0})
check("M, stock S and L -> close, shown in L", verdict(M, [S, L]), {"verdict": "close", "needed": "M", "worn": "L", "off": 1})
check("XL, stock up to L -> snug", verdict(XL, [XS, S, M, L]), {"verdict": "snug", "needed": "XL", "worn": "L", "off": -1})
check("no size known -> unknown", verdict(None, [M])["verdict"], "unknown")
check("no stock -> unknown", verdict(M, [])["verdict"], "unknown")
check("XS, stock M only -> too big by 2", verdict(XS, [M]), {"verdict": "too_big", "needed": "XS", "worn": "M", "off": 2})
check("clamped at -3", verdict(XL + 5, [XS])["off"], -3)

print("\nall good" if not fails else f"\n{fails} failed")
sys.exit(1 if fails else 0)
