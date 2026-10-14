import sys; sys.path.insert(0,'pylib')
import openpyxl, csv
APTS="EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM".split()
wb=openpyxl.load_workbook('txo.xlsx',read_only=True)
ws=wb['DATA']; rows=list(ws.iter_rows(values_only=True)); hdr=rows[0]
out=[r for r in rows[1:] if str(r[0]) in ("2024","2025") and r[3] in APTS]
with open('pru_published_2024_2025.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(hdr[:12]); [w.writerow(r[:12]) for r in out]
print(len(out))
for r in out:
    if str(r[0])=="2025" and str(r[1])=="6": print(r[:12], round(r[11]/r[9],2), round(r[10]/r[9],2))
