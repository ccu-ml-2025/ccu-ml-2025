
import math

def FFT(xreal, ximag):    
    n = 2
    while n*2 <= len(xreal):
        n *= 2
    p = int(math.log(n, 2))
    for i in range(n):
        a, b = i, 0
        for j in range(p):
            b = int(b*2 + a % 2)
            a /= 2
        if b > i:
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]
    wreal, wimag = [1.0], [0.0]
    arg = -2 * math.pi / n
    treal, timag = math.cos(arg), math.sin(arg)
    for _ in range(1, n // 2):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-2] * timag + wimag[-1] * treal)
    m = 2
    while m <= n:
        for k in range(0, n, m):
            for j in range(m // 2):
                t = int(n * j / m)
                ureal, uimag = xreal[k + j], ximag[k + j]
                treal = wreal[t] * xreal[k + j + m // 2] - wimag[t] * ximag[k + j + m // 2]
                timag = wreal[t] * ximag[k + j + m // 2] + wimag[t] * xreal[k + j + m // 2]
                xreal[k + j] = ureal + treal
                ximag[k + j] = uimag + timag
                xreal[k + j + m // 2] = ureal - treal
                ximag[k + j + m // 2] = uimag - timag
        m *= 2
    return n, xreal, ximag

def FFT_data(input_data, swinging_times):   
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength
    for num in range(len(swinging_times)-1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num+1]):
            a.append(math.sqrt(sum(input_data[swing][i] for i in range(3))**2))
            g.append(math.sqrt(sum(input_data[swing][i+3] for i in range(3))**2))
        a_mean[num] = sum(a) / len(a)
        g_mean[num] = sum(g) / len(g)
    return a_mean, g_mean
