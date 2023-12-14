

def calc_theta_abs(p1, p2):
    angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    angle = abs(angle)
    angle = min(angle, abs(angle - 180))
    return float(angle)


def calc_distance(point1, point2):
    # 这两个 Mat 有点多余？
    m1 = np.array([[point1[0], point1[1]]], dtype=np.float32)
    m2 = np.array([[point2[0], point2[1]]], dtype=np.float32)
    line = m1 - m2

    x = line[0, 0]
    y = line[0, 1]
    z = x ** 2 + y ** 2
    return np.sqrt(z)