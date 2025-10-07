import numpy as np
from sklearn.linear_model import LogisticRegression  # <-- CRITICAL LINE
from scipy.linalg import khatri_rao
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

MLPUF_N_BITS = 8
MLPUF_FEATURE_DIM = (MLPUF_N_BITS * (MLPUF_N_BITS - 1) // 2) * 2


################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

    feat_train = my_map(X_train)
    # Revised parameters: Prioritize accuracy (l1), optimize C and tol for speed
    model = LogisticRegression(
        C=1,              # Reduced C for speed, minimal expected accuracy impact
        penalty='l1',     # Keep l1 for best accuracy based on graphs
        solver='liblinear', # Keep solver compatible with l1
        tol=0.01,         # Increase tol for speed, no accuracy impact seen in graphs
        max_iter=1000,    # Keep max_iter (adjust if convergence issues arise)
        fit_intercept=True,
        random_state=42
    )

    model.fit(feat_train, y_train)
    w0 = model.coef_.flatten()
    b0 = model.intercept_[0]
    return w0, b0

    
################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
    num_samples, N = X.shape

    X_float = X.astype(np.float64)
    D = 1.0 - 2.0 * X_float
    P = np.fliplr(np.cumprod(np.fliplr(D), axis=1))

    feature_columns = []
    for i in range(N):
        for j in range(i + 1, N):
            feature_columns.append(D[:, i] * D[:, j])
    for i in range(N):
        for j in range(i + 1, N):
            feature_columns.append(P[:, i] * P[:, j])

    feat = np.stack(feature_columns, axis=1)
    return feat


ARBITER_N_BITS = 64
ARBITER_NUM_DELAYS = 4 * ARBITER_N_BITS
ARBITER_MODEL_DIM = ARBITER_N_BITS + 1

def construct_A(k=64):
    num_model_params = k + 1
    num_delays = 4 * k
    A = np.zeros((num_model_params, num_delays))

    A[0, 0] = 0.5
    A[0, 1] = -0.5
    A[0, 2] = 0.5
    A[0, 3] = -0.5

    for i in range(1, k):
        idx_i = 4 * i
        idx_prev = 4 * (i - 1)

        A[i, idx_i + 0] = 0.5
        A[i, idx_i + 1] = -0.5
        A[i, idx_i + 2] = 0.5
        A[i, idx_i + 3] = -0.5

        A[i, idx_prev + 0] += 0.5
        A[i, idx_prev + 1] += -0.5
        A[i, idx_prev + 2] += -0.5
        A[i, idx_prev + 3] += 0.5

    idx_last = 4 * (k - 1)
    A[k, idx_last + 0] = 0.5
    A[k, idx_last + 1] = -0.5
    A[k, idx_last + 2] = -0.5
    A[k, idx_last + 3] = 0.5

    return A

################################
# Non Editable Region Starting #
################################
def my_decode(model):
################################
#  Non Editable Region Ending  #
################################
    k = 64
    num_model_params = k + 1
    num_delays = 4 * k

    y_model = np.array(model)
    if y_model.shape[0] != num_model_params:
        raise ValueError(f"Input model must be of dimension {num_model_params}")

    A = construct_A(k)

    nnls_solver = LinearRegression(positive=True, fit_intercept=False)

    nnls_solver.fit(A, y_model.reshape(-1, 1))

    x_delays = nnls_solver.coef_[0]

    x_delays[x_delays < 0] = 0

    p = x_delays[0::4]
    q = x_delays[1::4]
    r = x_delays[2::4]
    s = x_delays[3::4]

    return p, q, r, s

