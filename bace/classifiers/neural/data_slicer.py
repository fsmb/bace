
def slice_data(X_data, y_data = None, slice_length=100, overlap_percent=0):
    """

    Parameters
    ----------
    X_data : array_like

    y_data : array_like, optional
    slice_length : int, optional
    overlap_percent : float, optional

    Returns
    -------
    x_sliced : array_like
		X_sliced is a list of lists
		X_sliced = [slice1, slice2, slice3... sliceN]
		for all i (slice i is a list of symbols)
		for all i (slice i is drawn from a doc with class i)
		for all i (slice i has length at most slice_length)
	y_sliced: array_like
		y_sliced is a list of  labels, or none if y_data is none
		len(X_sliced) == len(y_sliced) == N
		y_sliced = [class1, class2, class3... classN]
    """

    if overlap_percent is not None and (overlap_percent >= 1 or overlap_percent < 0):
        raise Exception("Invalid overlap amount")

    if y_data is not None and len(X_data) != len(y_data):
        raise Exception("Unequal lengths of input vectors")

    X_sliced = []
    y_sliced = [] if y_data is not None else None

    step = max(1, slice_length if overlap_percent is None else int(slice_length * (1 - overlap_percent)))

    for i in range(len(X_data)):
        data = X_data[i]
        slices = [data[i:min(len(data), i + slice_length)] for i in range(0, len(data), step)]
        X_sliced.extend(slices)
        if y_data is not None:
            y_sliced.extend([y_data[i]] * len(slices))

    return X_sliced, y_sliced

def test():
    """Test function

    """
    classes = range(5)
    x = [[(x, i) for x in range(1, 5 + i)] for i in classes]
    y = [i for i in classes]

    x2, y2 = slice_data(x, y, slice_length=3)
    for i in range(len(x2)):
        print(x2[i], y2[i])
    return x2, y2
