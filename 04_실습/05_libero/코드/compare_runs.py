"""평가 결과 비교 (10/17): python compare_runs.py piper_als2h piper_als2place ... → 공통 항목마다 (3cm, 판 안, 돌아오기 3cm, 돌아오기 판 안, 전환 수)"""
import json,glob,os,sys
def load(o):
    r={}
    for f in glob.glob(f"outputs/{o}_forget/*.json")+glob.glob(f"outputs/{o}_switch/*.json"):
        s=json.load(open(f)); m=s["summary"]; n=m["n"]; k=os.path.basename(f).split("_grasp")[0]
        if "b_success_rate_all" in m:
            r[k]=(n,round(m["b_success_rate_all"]*n),round(m["b_success_loose_rate_all"]*n),round(m["a_resume_genuine_rate_all"]*n),round(m["a_resume_genuine_loose_rate_all"]*n),m.get("n_switched"))
        else:
            r[k]=(n,round(m["a_success_rate"]*n),round(m["a_success_loose_rate"]*n),None,None,m.get("n_switched"))
    return r
names=sys.argv[1:]
R={o:load(o) for o in names}
keys=sorted(set.intersection(*[set(v) for v in R.values()]))
print("항목", *names)
for k in keys:
    print(k, *[R[o][k][1:] for o in names])
