import datetime as dt
import zipfile
from pathlib import Path

R = Path(r"F:\Projects\trading_stack_py")
REP = R / "reports"
REP.mkdir(exist_ok=True, parents=True)
ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out = REP / f"W24_review_{ts}.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in [REP / "wk24_black_litterman_compare.csv", REP / "w24_bl_summary.json"]:
        if p.exists():
            z.write(p, p.name)
print({"created": str(out)})
