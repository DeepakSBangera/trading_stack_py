import pandas as pd
import os
os.makedirs("reports/artifacts", exist_ok=True)
df = pd.DataFrame({"id":[1,2,3], "price":[100, None, 120]})
checks = {
    "null_price": df["price"].isna().sum(),
    "dup_ids": df["id"].duplicated().sum(),
}
df.to_csv("reports/artifacts/data_qc_sample.csv", index=False)
with open("reports/artifacts/data_qc_report.csv","w") as f:
    f.write("check,count\n")
    for k,v in checks.items():
        f.write(f"{k},{v}\n")
print("QC written to reports/artifacts/")
