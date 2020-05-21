import math
import numpy as np
import matplotlib.pyplot as plt
import re
import csv


def distance(point1, point2):
    return math.sqrt(
        math.pow((point1[0] - point2[0]), 2)
        + math.pow((point1[1] - point2[1]), 2)
        + math.pow((point1[2] - point2[2]), 2)
    )


def quat_difference(q1, q2):
    return (q1[0] * -q2[0], q1[1] * -q2[1], q1[2] * -q2[2], q1[3] * -q2[3])


def quat_euler(qdat):
    [X, Y, Z, W] = qdat
    phi = math.atan2(2 * (W * X + Y * Z), 1 - 2 * (X ** 2 + Y ** 2))
    theta = math.asin(2 * (W * Y - X * Z))
    psi = math.atan2(2 * (W * Z + X * Y), 1 - 2 * (Y ** 2 + Z ** 2))
    return (phi, theta, psi)


def magnatude(coord):
    return math.sqrt(coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2)


def bbox(points):
    """
    [xmin xmax]
    [ymin ymax]
    [zmin zmax]
    """
    a = np.zeros((3, 2))
    a[:, 0] = np.min(points, axis=0)
    a[:, 1] = np.max(points, axis=0)
    return a


def bboxVolume(points):
    box = bbox(points)
    dim = box[:, 1] - box[:, 0]
    return dim[0] * dim[1] * dim[2]


class MoveState:
    def __init__(self, pos, vel, acc, force, dist):
        self.p = pos
        self.v = vel
        self.a = acc
        self.f = force
        self.d = dist

    def set_p(self, pos):
        self.p = pos

    def set_v(self, vel):
        self.v = vel

    def set_a(self, acc):
        self.a = acc

    def set_f(self, force):
        self.f = force

    def set_d(self, dist):
        self.d = dist

    def get_p(self):
        return self.p

    def get_v(self):
        return self.v

    def get_a(self):
        return self.a

    def get_f(self):
        return self.f

    def get_d(self):
        return self.d


def update_movement(row, elem, elem_id):
    if row[elem_id] == "":
        print(row)
    dt = 1 / 150
    if elem == 0:
        elem = MoveState(
            (float(row[elem_id]), float(row[elem_id + 1]), float(row[elem_id + 2])),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.0,
        )
    else:
        oldA = elem.get_a()
        oldV = elem.get_v()
        oldP = elem.get_p()
        oldD = elem.get_d()
        newP = (float(row[elem_id]), float(row[elem_id + 1]), float(row[elem_id + 2]))
        newV = (
            (oldP[0] - newP[0]) / dt,
            (oldP[1] - newP[1]) / dt,
            (oldP[2] - newP[2]) / dt,
        )
        newA = (
            (oldV[0] - newV[0]) / dt,
            (oldV[1] - newV[1]) / dt,
            (oldV[2] - newV[2]) / dt,
        )
        newF = (
            (oldA[0] - newA[0]) / dt,
            (oldA[1] - newA[1]) / dt,
            (oldA[2] - newA[2]) / dt,
        )
        newD = distance(oldP, newP)
        elem.set_f(newF)
        elem.set_a(newA)
        elem.set_v(newV)
        elem.set_p(newP)
        elem.set_d(newD)
    return elem


def laban_feature_pop(row, movement, FEATURES):
    # define feature space
    features = []

    ## feet to hip distance
    lf = row[113:116]
    rf = row[141:144]
    hp = row[1:4]
    features.append((distance(rf, hp) + distance(lf, hp)) / 2)

    ## hand to shoulder distance
    lh = row[64:67]
    rh = row[92:95]
    ls = row[43:46]
    rs = row[71:74]
    features.append((distance(lh, ls) + distance(rh, rs)) / 2)

    ## hand to hand distance -- "hands"
    features.append(distance(lh, rh))

    ## hand to head distance
    hd = (row[36], row[37], row[38])
    features.append((distance(lh, hd) + distance(rh, hd)) / 2)

    ## hand to hip distance
    features.append((distance(lh, hp) + distance(rh, hp)) / 2)

    ## hip ground distance -- "hip_ground"
    features.append(row[3])

    ## hip to ground distance minus feet hip distance
    diffl = row[3] - distance(lf, hp)
    diffr = row[3] - distance(rf, hp)
    features.append((diffl + diffr) / 2)

    ## gait size -- "gait"
    features.append(distance(lf, rf))

    ## Head orientation
    qhd_diff = quat_difference(row[39:43], row[4:8])
    euler = quat_euler(qhd_diff)
    # "head_orientation_phi"
    features.append(euler[0])
    # "head_orientation_theta"
    features.append(euler[1])
    # "head_orientation_psi"
    features.append(euler[2])

    ## Deceleration peaks
    root_a = movement["root"].get_a()
    for it in root_a:
        it = it if it < 0 else 0
    # "decel_peaks"
    features.append(magnatude(root_a))

    ## Hip velocity -- "hip_vel"
    hip_v = movement["root"].get_v()
    features.append(magnatude(hip_v))

    ## Hands velocity
    hl_v = movement["lhand"].get_v()
    hr_v = movement["rhand"].get_v()
    features.append((magnatude(hl_v) + magnatude(hr_v)) / 2)

    ## Foot velocity
    # Foot vel left -- "feet_vel_l"
    fl_v = movement["lfoot"].get_v()
    fr_v = movement["rfoot"].get_v()
    features.append((magnatude(fl_v) + magnatude(fr_v)) / 2)

    ## Hip acceleration
    # "hip_acc"
    hip_a = movement["root"].get_a()
    features.append(magnatude(hip_a))

    ## Hand acceleration
    # left -- "hand_acc_l"
    lh_a = movement["lhand"].get_a()
    rh_a = movement["rhand"].get_a()
    features.append((magnatude(lh_a) + magnatude(rh_a)) / 2)

    ## Foot acceleration
    # left -- "feet_acc_l"
    lf_a = movement["lfoot"].get_a()
    rf_a = movement["rfoot"].get_a()
    features.append((magnatude(lf_a) + magnatude(rf_a)) / 2)

    ## Jerk -- "jerk"
    root_f = movement["root"].get_f()
    features.append(magnatude(root_f))

    ## Volume
    # 5 joints -- "vol_5"
    # a = np.transpose([lf,rf,lh,rh,hd])
    features.append(bboxVolume([lf, rf, lh, rh, hd]))
    # All joints -- "vol_all"
    r = re.compile(".*_X")
    fullBody = list(filter(r.match, FEATURES))
    bodyArray = []
    for jointName in fullBody:
        jointID = FEATURES.index(jointName)
        bodyArray.append(row[jointID : jointID + 3])
    a = np.transpose(bodyArray)
    features.append(bboxVolume(bodyArray))

    # Volume upper -- "vol_upper"
    features.append(bboxVolume(bodyArray[0:14]))

    # Volume lower -- "vol_lower"
    lwr_arr = [bodyArray[0]] + bodyArray[15:-1]
    features.append(bboxVolume(lwr_arr))

    # Volume left -- "vol_l"
    r = re.compile("Left.*_X")
    leftBody = list(filter(r.match, FEATURES))
    leftArray = []
    for jointName in leftBody:
        jointID = FEATURES.index(jointName)
        leftArray.append(row[jointID : jointID + 3])
    a = np.transpose(leftArray)
    features.append(bboxVolume(leftArray))

    # Volume right -- "vol_r"
    r = re.compile("Right.*_X")
    rightBody = list(filter(r.match, FEATURES))
    rightArray = []
    for jointName in rightBody:
        jointID = FEATURES.index(jointName)
        rightArray.append(row[jointID : jointID + 3])
    a = np.transpose(rightArray)
    features.append(bboxVolume(rightArray))

    # Torso height -- "torso_height"
    features.append(distance(hd, hp))

    ## Hand Level
    hll = 0
    hlr = 0
    if lh[2] >= hd[2]:
        hll = 2
    elif lh[2] >= hp[2]:
        hll = 1

    if rh[2] >= hd[2]:
        hlr = 2
    elif rh[2] >= hp[2]:
        hlr = 1

    features.append((hll + hlr) / 2)

    # Total Distance -- "total_dist"
    rt_d = movement["root"].get_d()
    features.append(rt_d)

    # Total Area -- "total_area"
    features.append(rt_d * hp[2])

    return features


def featureFrames(labal_features, frame_size):
    lframe = []
    for index in range(len(labal_features) - frame_size):
        fframe = []
        rframe = np.array(labal_features[index : (index + frame_size)])
        for j in range(len(labal_features[index])):
            frame = rframe[:, j]
            if j in [11, 26, 27, 28]:
                fframe.append(frame.sum())
            elif j in [15, 16, 17, 18]:
                # MAX
                fframe.append(frame.max())
                # STD
                fframe.append(frame.std())

            elif j in [12, 13, 14]:
                # MAX
                fframe.append(frame.max())
                # MIN
                fframe.append(frame.min())
                # STD
                fframe.append(frame.std())
            else:
                # MAX
                fframe.append(frame.max())
                # MIN
                fframe.append(frame.min())
                # MEAN
                fframe.append(frame.mean())
                # STD
                fframe.append(frame.std())
        lframe.append(fframe)
    return lframe


def read_in_features(path_in):
    skeleton_labels = []
    with open(path_in, "r") as inp:
        for row in csv.reader(inp):
            index = 0
            prefix = ""
            for elem in row:
                if index == 0:
                    prefix = elem
                else:
                    skeleton_labels.append(prefix + "_" + elem)
                index = (index + 1) % 8
            return skeleton_labels


def read_in_dancer(path_in, FEATURES):
    index = 0
    dancer_data = []
    movement = {"root": 0, "lhand": 0, "rhand": 0, "lfoot": 0, "rfoot": 0}
    with open(path_in, "r") as inp:
        timestamp = 0
        for row in csv.reader(inp):
            if index > 300000:
                break
            elif index != 0:
                i = 0
                exprow = [timestamp]
                for elem in row:
                    if i != 0:
                        exprow.append(float(elem))
                    i = (i + 1) % 8
                # root_elem = (exprow[0], exprow[1], exprow[2])
                # i = 0
                # temp = []
                # for elem in exprow:
                # if i < 3:
                # temp.append(elem - root_elem[i])
                # else:
                # temp.append(elem)
                # i = (i + 1) % 7
                # exprow = temp

                elem_id = FEATURES.index("Hips_X") + 1
                movement["root"] = update_movement(exprow, movement["root"], elem_id)
                elem_id = FEATURES.index("LeftHand_X") + 1
                movement["lhand"] = update_movement(exprow, movement["lhand"], elem_id)
                elem_id = FEATURES.index("RightHand_X") + 1
                movement["rhand"] = update_movement(exprow, movement["rhand"], elem_id)
                elem_id = FEATURES.index("LeftFoot_X") + 1
                movement["lfoot"] = update_movement(exprow, movement["lfoot"], elem_id)
                elem_id = FEATURES.index("RightFoot_X") + 1
                movement["rfoot"] = update_movement(exprow, movement["rfoot"], elem_id)
                features = laban_feature_pop(exprow, movement, FEATURES)
                dancer_data.append(features)
            index = index + 1
    return dancer_data


def read_in_labels(labels_path, dance_len):
    timestamp = 0
    count = 0
    temp = []
    labels = []
    with open(labels_path, "r") as inp:
        labels.append(["timestamp", "label"])
        for row in csv.reader(inp):
            if count == 0:
                count = 1
            else:
                temp.append(row)
    total_time = int(temp[-1][2])
    print("total time: ", total_time)
    fps = int(dance_len / total_time)
    print("frames per second: ", fps)

    for row in temp:
        endT = (int(row[2]) * fps) - 1
        startT = int(row[1]) * fps
        for timestamp in range(startT, endT):
            ts = float(timestamp) / fps
            labels.append([ts, int(row[0])])
    return labels


def read_in_selected_features(features_path):
    with open(features_path, "r") as inp:
        for row in csv.reader(inp):
            return row[2:-1]


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    return plt


def multivariate_data(
    dataset,
    target,
    start_index,
    end_index,
    history_size,
    target_size,
    step,
    single_step=False,
):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i : i + target_size])

    return np.array(data), np.array(labels)


def plot_train_history(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.legend()

    plt.show()


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)
