import numpy as np

def EBaperture(quarter):
    """Return an aperture mask for the EB for a given quarter."""
    if quarter == 1:
        c1 = 164
        c2 = 169
        r1 = 173
        r2 = 179
    elif quarter == 2:
        c1 = 168
        c2 = 174
        r1 = 174
        r2 = 180
    elif quarter == 3:
        c1 = 168
        c2 = 174
        r1 = 167
        r2 = 174
    elif quarter == 4:
        c1 = 164
        c2 = 170
        r1 = 167
        r2 = 173
    elif quarter == 5:
        c1 = 164
        c2 = 169
        r1 = 173
        r2 = 179
    elif quarter == 6:
        c1 = 168
        c2 = 174
        r1 = 174
        r2 = 180
    elif quarter == 7:
        c1 = 168
        c2 = 174
        r1 = 167
        r2 = 174
    elif quarter == 8:
        c1 = 164
        c2 = 169
        r1 = 167
        r2 = 173
    elif quarter == 9:
        c1 = 164
        c2 = 169
        r1 = 173
        r2 = 179
    elif quarter == 10:
        c1 = 168
        c2 = 174
        r1 = 174
        r2 = 180
    elif quarter == 11:
        c1 = 168
        c2 = 174
        r1 = 167
        r2 = 174
    elif quarter == 12:
        c1 = 164
        c2 = 169
        r1 = 167
        r2 = 173
    elif quarter == 13:
        c1 = 164
        c2 = 169
        r1 = 173
        r2 = 179
    elif quarter == 14:
        c1 = 168
        c2 = 174
        r1 = 174
        r2 = 180
    elif quarter == 15:
        c1 = 168
        c2 = 174
        r1 = 167
        r2 = 174
    elif quarter == 16:
        c1 = 164
        c2 = 169
        r1 = 167
        r2 = 173
    elif quarter == 17:
        c1 = 164
        c2 = 169
        r1 = 174
        r2 = 178
    else:
        raise ValueError("Invalid quarter.")

    ap = np.zeros((200, 200), dtype=bool)
    for col in np.arange(c1, c2):
        for row in np.arange(r1, r2):
            ap[row, col] = True

    return ap
