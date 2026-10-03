"""汇总若干 kitchen-eval 报告：k/n、Wilson 95% CI、逐任务率、按序前缀阶梯。"""
import glob, json, math, sys

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        eps = r["episodes"]; n = len(eps)
        k = sum(1 for e in eps if e.get("success"))
        tasks = r["tasks"]
        ladder = [sum(1 for e in eps if e.get("ordered_prefix_length", 0) >= i)
                  for i in range(1, len(tasks) + 1)]
        out.append((f, r, n, k, ladder))
    return out

def main(pattern):
    rows = load(pattern)
    if not rows:
        print(f"没有匹配的报告：{pattern}"); return
    print(f"{'报告':52s} {'k/n':>8s} {'成功率':>7s} {'95% CI':>15s}  按序前缀阶梯")
    for f, r, n, k, ladder in rows:
        lo, hi = wilson(k, n)
        lad = "  ".join(f"{i}:{c}/{n}" for i, c in enumerate(ladder, 1))
        name = f.split("outputs/eval/")[-1].split("/")[0] if "outputs/eval/" in f else f
        print(f"{name:52s} {k:>3d}/{n:<4d} {k/n:>7.3f} {f'[{lo:.3f}, {hi:.3f}]':>15s}  {lad}")
    if len(rows) > 1:
        best = max(rows, key=lambda x: x[3] / max(x[2], 1))
        print(f"\n最高：{best[0].split('outputs/eval/')[-1].split('/')[0]}  "
              f"{best[3]}/{best[2]} = {best[3]/best[2]:.3f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval/*/report_*.json")
