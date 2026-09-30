"""Tools the agent can call instead of guessing."""
import math
import re

import requests

OPS_PER_SECOND = 1e8  # common contest rule of thumb for simple operations


def codeforces_rating(handle: str) -> str:
    """Live lookup of a Codeforces user via the public API (no key needed)."""
    try:
        r = requests.get("https://codeforces.com/api/user.info",
                         params={"handles": handle}, timeout=8)
        data = r.json()
        if data.get("status") != "OK":
            return f"Codeforces could not find the handle '{handle}'."
        u = data["result"][0]
        return (f"Handle {u['handle']}: rating {u.get('rating', 'unrated')}, "
                f"max rating {u.get('maxRating', 'n/a')}, rank {u.get('rank', 'unrated')}, "
                f"max rank {u.get('maxRank', 'n/a')}.")
    except Exception:
        return "The Codeforces API is not reachable right now. Please try again later."


_FUNCS = {
    "1": lambda n: 1,
    "logn": lambda n: math.log2(n),
    "sqrtn": lambda n: math.sqrt(n),
    "n": lambda n: n,
    "nlogn": lambda n: n * math.log2(n),
    "nsqrtn": lambda n: n * math.sqrt(n),
    "n2": lambda n: n ** 2,
    "n3": lambda n: n ** 3,
    "2n": lambda n: 2 ** n if n <= 1000 else float("inf"),
}


def _normalise(c: str) -> str:
    c = c.lower().strip().replace("o(", "")
    return re.sub(r"[\s*^()]", "", c)


def complexity_check(complexity: str, n: int) -> str:
    """Estimate operations for a time complexity and input size n."""
    key = _normalise(complexity)
    if key not in _FUNCS or n < 1:
        return f"Unsupported complexity '{complexity}' or invalid n={n}."
    complexity = re.sub(r"^[oO]\(|\)$", "", complexity.strip())
    ops = _FUNCS[key](n)
    seconds = ops / OPS_PER_SECOND
    if ops == float("inf"):
        verdict = "far too slow"
    elif seconds <= 1:
        verdict = "should fit in a 1 second limit"
    elif seconds <= 3:
        verdict = "risky for a 1 second limit (may fit in 2-3 seconds)"
    else:
        verdict = "too slow for typical contest limits"
    ops_txt = "infinite" if ops == float("inf") else f"{ops:.2e}"
    return (f"O({complexity}) with n = {n:,} is about {ops_txt} operations "
            f"(~{seconds:.2f}s at 1e8 ops/s): {verdict}.")


def current_datetime() -> str:
    from datetime import datetime
    return datetime.now().strftime("Today is %A, %d %B %Y, %H:%M local time.")
