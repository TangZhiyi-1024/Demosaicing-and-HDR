import numpy as np
import cv2

def icam06(rgb_lin, output_range=4.0, sigma_s=8, sigma_r=0.4, eps=1e-8):
    # rgb_lin: float, 线性RGB, HxWx3
    R, G, B = [rgb_lin[..., i] for i in range(3)]

    # 1) intensity + chroma
    I = (20*R + 40*G + 1*B) / 61.0
    r, g, b = R / I, G / I, B / I

    # 2) bilateral on log-intensity (cv2 需要 float32)
    logI = np.log(I).astype(np.float32)
    # sigma_s: 空间半径（像素）；sigma_r: 颜色/强度域，按 ln(I) 的动态设置
    log_base = cv2.bilateralFilter(logI, d=0, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    log_detail = logI - log_base

    # 3) base 压缩
    Lmin, Lmax = float(log_base.min()), float(log_base.max())
    c = np.log(output_range) / max(Lmax - Lmin, eps)
    o = -Lmax * c
    I_out = np.exp(c*log_base + o + log_detail)

    # 4) 复原 RGB
    Rout, Gout, Bout = r*I_out, g*I_out, b*I_out
    out = np.stack([Rout, Gout, Bout], axis=-1)
    return out

