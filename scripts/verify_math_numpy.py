# numpy mirror of oracle + estimators: brute-force check and Pythagoras with an arbitrary non-local f
import itertools
import numpy as np
from scipy.special import logsumexp, softmax

rng = np.random.default_rng(0)
K, d = 3, 3
labels = np.repeat(np.arange(10), 2)
tiles = rng.uniform(-1, 1, (20, d))
res_all = labels % K
order = np.argsort(res_all, kind="stable")
X, res = tiles[order], res_all[order]
sizes = np.bincount(res, minlength=K)
offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
P = len(X)


def tile_post(z, a, s):  # z (d,)
    logw = (a * X @ z - 0.5 * a ** 2 * (X ** 2).sum(1)) / s ** 2
    lL = np.array([logsumexp(logw[res == c]) - np.log(sizes[c]) for c in range(K)])
    eta = softmax(lL)
    mu = np.stack([softmax(logw[res == c]) @ X[res == c] for c in range(K)])
    return eta, mu


def conv(p, q):
    return np.array([sum(p[a_] * q[(r - a_) % K] for a_ in range(K)) for r in range(K)])


def targets(z, a, s):  # z (4,d)
    E, MU = zip(*[tile_post(z[i], a, s) for i in range(4)])
    E, MU = np.array(E), np.array(MU)
    pi = np.zeros_like(E); Zs = []
    for i in range(4):
        o = [j for j in range(4) if j != i]
        S = conv(conv(E[o[0]], E[o[1]]), E[o[2]])
        un = E[i] * S[(-np.arange(K)) % K]
        Zs.append(un.sum()); pi[i] = un / un.sum()
    ml = (z - a * (E[..., None] * MU).sum(1)) / s
    mg = -(a / s) * ((pi - E)[..., None] * MU).sum(1)
    return ml + mg, ml, mg, E, np.array(Zs)


def brute(z, a, s):
    logl = -((z[:, None] - a * X[None]) ** 2).sum(-1) / (2 * s ** 2)
    l = np.exp(logl - logl.max(1, keepdims=True)) / sizes[res][None]
    num = np.zeros((4, d)); den = 0
    for idx in itertools.product(range(P), repeat=4):
        if sum(res[list(idx)]) % K: continue
        w = l[0, idx[0]] * l[1, idx[1]] * l[2, idx[2]] * l[3, idx[3]]
        den += w
        for i in range(4): num[i] += w * X[idx[i]]
    Ex = num / den
    Exl = (l / l.sum(1, keepdims=True)) @ X
    return (z - a * Ex) / s, (z - a * Exl) / s


def sample_pop(n):
    c = rng.integers(0, K, (n, 4)); c[:, 3] = (-c[:, :3].sum(1)) % K
    idx = offs[c] + (rng.random((n, 4)) * sizes[c]).astype(int)
    return X[idx]


# 1) brute force
for (a, s) in [(0.95, 0.3), (0.7, 0.7), (0.3, 0.95)]:
    x = sample_pop(3)
    for b in range(3):
        z = a * x[b] + s * rng.standard_normal((4, d))
        ms, ml, mg, _, Zs = targets(z, a, s)
        bs, bl = brute(z, a, s)
        print(f"a={a} b={b}: |m* err| {np.abs(ms-bs).max():.1e} |m_loc err| {np.abs(ml-bl).max():.1e} "
              f"Z spread {np.ptp(Zs)/Zs.mean():.1e}")

# 2) Pythagoras + estimators with an arbitrary NON-local f
W = rng.standard_normal((4 * d, 4 * d)) * 0.5
f = lambda z: np.tanh(z.reshape(-1) @ W).reshape(4, d) + 0.3 * z


def completions(z, i, E_i, a, s, M):
    out = []
    for _ in range(M):
        c = rng.integers(0, K, 4)
        c[i] = rng.choice(K, p=E_i)
        last = (i + 3) % 4
        c[last] = (-(c.sum() - c[last])) % K
        idx = offs[c] + (rng.random(4) * sizes[c]).astype(int)
        zz = a * X[idx] + s * rng.standard_normal((4, d))
        zz[i] = z[i]
        out.append(f(zz)[i])
    return np.array(out)


a, s, M, n = 0.8, 0.6, 16, 3000
x = sample_pop(n)
U, G, EXC, LOSS, C = [], [], [], [], []
for b in range(n):
    eps = rng.standard_normal((4, d)); z = a * x[b] + s * eps
    ms, ml, mg, E, _ = targets(z, a, s)
    fz = f(z)
    EXC.append(((fz - ms) ** 2).sum()); LOSS.append(((eps - fz) ** 2).sum()); C.append(((eps - ms) ** 2).sum())
    u = g = 0
    for i in range(4):
        v = completions(z, i, E[i], a, s, M) - ml[i]
        Sv = v.sum(0); Q = (v ** 2).sum()
        u += ((Sv ** 2).sum() - Q) / (M * (M - 1))
        vbar = Sv / M; S2 = (Q - M * (vbar ** 2).sum()) / (M - 1)
        g += ((fz[i] - mg[i] - ml[i] - vbar) ** 2).sum() - S2 / M
    U.append(u); G.append(g)
U, G, EXC = map(np.array, (U, G, EXC))
se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
print(f"\nR_local {U.mean():.4f}±{se(U):.4f}  R_global(direct) {G.mean():.4f}±{se(G):.4f}  "
      f"excess {EXC.mean():.4f}±{se(EXC):.4f}")
r = EXC - U - G
print(f"Pythagoras residual excess - R_loc - R_glob = {r.mean():+.4f} ± {se(r):.4f}")
print(f"loss - C - excess = {np.mean(LOSS)-np.mean(C)-EXC.mean():+.4f} (sd of loss {np.std(LOSS):.2f})")
