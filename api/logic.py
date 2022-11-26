import numpy as np
import pandas as pd
from scipy.linalg import hankel


def bilinear_transform(continuous_state_space: dict, T):
    A = continuous_state_space["A"]
    B = continuous_state_space["B"]
    C = continuous_state_space["C"]
    D = continuous_state_space["D"]

    # Dimension Identification
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    # Dimension Assertion
    assert A.shape == (n, n)
    assert B.shape == (n, m)
    assert C.shape == (p, n)
    half_T = T / 2

    identity = np.identity(n)

    inverse_matrix = np.linalg.inv(identity - half_T * A)

    Ad = np.matmul(inverse_matrix, (identity + half_T * A))
    Bd = np.matmul(inverse_matrix, (T / (1 - half_T)) * B)
    Cd = C

    out = {
        "Ad": Ad,
        "Bd": Bd,
        "Cd": Cd,
        "Dd": None,
    }

    if D is not None:
        assert D.shape == (p, m)
        Dd = D
        out["Dd"] = Dd

    # print(f"{out=}")
    return out


def model_response(state_space: dict, x, u):
    n = state_space["Ad"].shape[0]
    x = x.reshape(n, 1)

    x_plus = np.matmul(state_space["Ad"], x) + np.matmul(state_space["Bd"], u)

    y = np.matmul(state_space["Cd"], x)

    if state_space["Dd"] is not None:
        y += np.matmul(state_space["Dd"], u)

    return x_plus, y


def step_response(state_space: dict, x):

    n = state_space["Ad"].shape[0]
    x = x.reshape(n, 1)

    u = np.array([1]).reshape([1, 1])
    x_plus, y = model_response(state_space, x, u)
    x_plus = np.ravel(x_plus)

    return x_plus, y


def step_response_loop(state_space, k_max, T):

    n = state_space["Ad"].shape[0]

    x = np.zeros((n, k_max))
    y = np.zeros((1, k_max))
    k_array = np.arange(k_max)

    for k in range(k_max):
        if k + 1 >= k_max:
            break

        x[:, k + 1], y[0, k] = step_response(state_space, x[:, k])

    imp = np.diff(y, n=1, prepend=[[0]])

    df = pd.DataFrame(
        {
            "Time": k * T,
            "k": k_array[:-1],
            "Step Response": np.ravel(y)[:-1],
            "Impulse Response": np.ravel(imp)[:-1],
        }
    )
    return df


def ho_kalman(y, n, ns):
    # Ho Kalman Algorithm converted from Matlab to Python
    # Source:
    # - http://mocha-java.uccs.edu/ECE5710/ECE5710-Notes05.pdf(page 29-31)
    # - https://research.wmz.ninja/articles/2018/10/notes-on-migrating-doa-tools-from-matlab-to-python.html (Matlab to Python conversion)

    if n < ns:
        raise Exception(
            "N input needed to be bigger than Ho Kalman Estimation N data")

    y = y.to_numpy()
    y = np.ravel(y)

    big_hankel = hankel(y[1:])

    hankel_k = big_hankel[:n, :n]
    hankel_k_plus = big_hankel[1: n + 1, :n]

    u, s, v_t = np.linalg.svd(hankel_k)
    v = v_t.T
    s_matrix = np.zeros((hankel_k.shape[0], hankel_k.shape[1]))
    s_matrix[: hankel_k.shape[1], : hankel_k.shape[1]] = np.diag(s)

    us = u[:, :ns]
    ss = s_matrix[:ns, :ns]
    vs = v[:, :ns]

    ok = np.matmul(us, np.sqrt(ss))
    cl = np.matmul(np.sqrt(ss), vs.T)

    A_hat = mr_divide(np.array(ml_divide(ok, hankel_k_plus)), cl)
    B_hat = cl[:, 0]
    C_hat = ok[0, :]
    D_hat = y[0]

    df = {
        "Ad": A_hat,
        "Bd": B_hat.reshape(-1, 1),
        "Cd": C_hat.reshape(1, -1),
        "Dd": D_hat.reshape(-1, 1),
    }

    return df


def mr_divide(A, B):
    # Python implementation of MATLAB mr_divide
    return np.linalg.lstsq(B.T, A.T, rcond=None)[0].T


def ml_divide(A, B):
    # Python implementation of MATLAB ml_divide
    return np.linalg.lstsq(A, B, rcond=None)[0]
