import numpy as np
from tqdm import tqdm
import copy

from .mpcproblem import MPC
from .datasetutils import mpc_dataset_import
from .trainampc import import_model, import_model_mpc


# ============================================================
# Base AMPC Controller
# ============================================================

class AMPC:
    """
    Neural approximate MPC controller.

    Supports optional context augmentation:
        NN input = [x]                     (default)
        NN input = [x, context_features]   (if set_context() is used)
    """

    def __init__(self, mpc, model):
        if not isinstance(mpc, MPC):
            raise TypeError("mpc must be of type MPC")

        self._mpc = mpc
        self._model = model
        self._feasible = False
        self._context = None   # optional additional NN input features

    # ---------------------------
    # Properties
    # ---------------------------

    @property
    def mpc(self):
        return self._mpc

    @property
    def feasible(self):
        return self._feasible

    # ---------------------------
    # Context handling
    # ---------------------------

    def set_context(self, context):
        """
        Optional: set additional NN input features.

        Example for vehicle+obs:
            context = [o1x, o1y, o2x, o2y]
        """
        if context is None:
            self._context = None
        else:
            self._context = np.asarray(context, dtype=np.float32).reshape(-1)

    # ---------------------------
    # Initialization
    # ---------------------------

    def initialize(self, x0, V):
        pass

    # ---------------------------
    # NN evaluation
    # ---------------------------

    def _build_nn_input(self, x0):
        x0 = np.asarray(x0, dtype=np.float32).reshape(-1)

        if self._context is None:
            return x0

        return np.concatenate([x0, self._context], axis=0)

    def V(self, x0):
        x_in = self._build_nn_input(x0)

        y = self._model(x_in[None, :]).numpy()

        return np.reshape(y, (self._mpc.N, self._mpc.nu))

    # ---------------------------
    # Closed-loop call
    # ---------------------------

    def __call__(self, x0):
        x0_extra = np.expand_dims(x0, axis=1)
        V = self.V(x0_extra)
        X = self._mpc.forward_simulate_trajectory_clipped_inputs(x0, V)
        self._feasible = self._mpc.feasible(X, V)
        return V[0], X[1]


# ============================================================
# Safe Online Evaluation
# ============================================================

class SafeOnlineEvaluationAMPC(AMPC):

    def __init__(self, mpc, model):
        super().__init__(mpc, model)
        self._V_candidate = None
        self._X_candidate = None
        self.feasible_debug = {}

    # ---------------------------
    # Initialization
    # ---------------------------

    def initialize_candidate(self, x0, V_initialize, verbose=True):
        self._V_candidate = V_initialize
        self._X_candidate = self.mpc.forward_simulate_trajectory_clipped_inputs(x0, V_initialize)

        self._feasible = self.mpc.feasible(self._X_candidate, self._V_candidate, robust=False)

        if not self.mpc.in_state_and_input_constraints(
                self._X_candidate, self._V_candidate, robust=False
        ) and verbose:
            print("WARNING: Initialization infeasible!")

    def initialize(self, x0, V):
        # initialize using NN prediction
        self.initialize_candidate(x0, self.V(x0))

    # ---------------------------
    # Shift logic
    # ---------------------------

    def shift_append_terminal(self, X, V):
        V_shifted = np.zeros((self.mpc.N, self.mpc.nu))
        V_shifted[:-1] = V[1:]
        V_shifted[-1] = self.mpc.terminal_controller(X[-1])

        X_shifted = self.mpc.forward_simulate_trajectory_clipped_inputs(
            X[1], V_shifted
        )

        return X_shifted, V_shifted

    # ---------------------------
    # Safe evaluation
    # ---------------------------

    def safe_evaluate(self, x, V, enforce_cost_decrease=True):

        X = self.mpc.forward_simulate_trajectory_clipped_inputs(x, V)

        # cost decrease check
        cost_decrease = True
        if enforce_cost_decrease:
            cost_decrease = (
                self.mpc.cost(X, V)
                <= self.mpc.cost(self._X_candidate, self._V_candidate)
            )

        candidate_feasible = self.mpc.feasible(self._X_candidate, self._V_candidate)
        new_feasible = self.mpc.feasible(X, V)
        state_feasible = self.mpc.in_state_and_input_constraints(X, V)
        terminal_feasible = self.mpc.in_terminal_constraints(X[-1])

        self.feasible_debug = {
            "candidate_feasible": candidate_feasible,
            "in_state_and_input_constraints": state_feasible,
            "in_terminal_constraint": terminal_feasible,
            "cost_decrease": cost_decrease,
        }

        if (new_feasible and cost_decrease) or (not candidate_feasible):
            self._V_candidate = copy.deepcopy(V)
            self._X_candidate = copy.deepcopy(X)
            self._feasible = new_feasible
        else:
            self._feasible = False

        v = self._V_candidate[0].copy()
        x_next = self._X_candidate[1].copy()

        self._X_candidate, self._V_candidate = self.shift_append_terminal(
            self._X_candidate, self._V_candidate
        )

        return v, x_next

    def __call__(self, x0):
        V = self.V(x0)
        return self.safe_evaluate(x0, V)


# ============================================================
# Safe Init Version
# ============================================================

class SafeOnlineEvaluationAMPCGroundTruthInit(SafeOnlineEvaluationAMPC):

    def initialize(self, x0, V):
        self.initialize_candidate(x0, V)


# ============================================================
# Closed Loop Simulation
# ============================================================

def closed_loop_experiment(x0, controller, Nsim=1000):

    nx = controller.mpc.nx
    nu = controller.mpc.nu

    V_cl = np.zeros((Nsim - 1, nu))
    U_cl = np.zeros((Nsim - 1, nu))
    X_cl = np.zeros((Nsim, nx))
    feasible_ampc = np.zeros((Nsim - 1))

    status = "running"
    X_cl[0] = x0

    for k in range(Nsim - 1):

        xk = X_cl[k]
        v, x_next = controller(xk)
        u = controller.mpc.stabilizing_feedback_controller_clipped_inputs(xk, v)

        V_cl[k] = v
        U_cl[k] = u
        X_cl[k + 1] = x_next
        feasible_ampc[k] = controller.feasible

        feasible_cl = controller.mpc.in_state_and_input_constraints(
            X_cl[: k + 2], V_cl[: k + 1], robust=False
        )

        if not feasible_cl:
            status = "infeasible_cl"
            return status, X_cl[: k + 2], U_cl[: k + 1], V_cl[: k + 1], feasible_ampc[: k + 1]

        if controller.mpc.in_terminal_constraints(x_next, robust=False):
            status = "terminal_set_reached"
            return status, X_cl[: k + 2], U_cl[: k + 1], V_cl[: k + 1], feasible_ampc[: k + 1]

    return "timeout", X_cl, U_cl, V_cl, feasible_ampc


# ============================================================
# Helper: iterate controllers over one initial condition
# ============================================================

def iterate_controllers(x0, V_init, controllers, Nsim=1000):
    results = []
    for controller in controllers:
        controller.initialize(x0, V_init)
        feasible_init = controller.feasible
        status, X, U, V, feasible = closed_loop_experiment(x0, controller, Nsim=Nsim)
        results.append({
            "status": status, "X": X, "U": U, "V": V,
            "feasible": feasible, "feasible_init": feasible_init,
        })
    return results


# ============================================================
# Closed-loop test functions
# ============================================================

def closed_loop_test_on_dataset(dataset, model_name, N_samples=int(1e3), N_sim=200):
    """Closed-loop simulation on dataset of initial conditions."""
    mpc, X, V, _, _ = mpc_dataset_import(dataset)
    X = np.atleast_2d(X)
    V = np.asarray(V)

    if X.shape[0] != V.shape[0]:
        raise ValueError(f"Sample mismatch: X has {X.shape[0]} rows, V has {V.shape[0]} rows")

    rng = np.random.default_rng(42)
    perm = rng.permutation(X.shape[0])
    X, V = X[perm], V[perm]

    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print(f"WARNING: N_samples exceeds dataset size, using N_samples={N_samples}")

    X_test, V_test = X[:N_samples], V[:N_samples]

    model = import_model(modelname=model_name)
    naive_controller = AMPC(mpc, model)
    safe_controller_init = SafeOnlineEvaluationAMPCGroundTruthInit(mpc, model)

    controllers = [naive_controller, safe_controller_init]
    controller_names = ["naive", "safe init"]

    results = []
    print(f"\ntesting controllers on {len(X_test)} initial conditions in closed loop\n")
    for i in tqdm(range(N_samples)):
        simulation_results = iterate_controllers(X_test[i], V_test[i], controllers, Nsim=N_sim)
        results.append(simulation_results)

    for j in range(len(controllers)):
        status_cl = np.array([
            results[i][j]["status"]
            for i in range(N_samples)
            if any(results[i][1]["feasible"]) or results[i][1]["feasible_init"]
        ])
        status_cl_safe_NN_init = np.array([
            results[i][j]["status"]
            for i in range(N_samples)
            if results[i][j]["feasible"][0]
        ])
        terminal_set_reached_cl = np.mean(status_cl == "terminal_set_reached")
        feasible_cl = np.mean(status_cl != "infeasible_cl")
        feasible_cl_safe_NN_init = np.mean(status_cl_safe_NN_init != "infeasible_cl")
        print(f"Results for controller: {controller_names[j]}: "
              f"{terminal_set_reached_cl=}, {feasible_cl=}, {feasible_cl_safe_NN_init=}")

    idx_naive_infeas_safe_feas = [
        i for i in range(N_samples)
        if results[i][0]["status"] == "infeasible_cl" and results[i][1]["status"] != "infeasible_cl"
    ]
    print(f"{idx_naive_infeas_safe_feas=}")
    return [results[idx] for idx in idx_naive_infeas_safe_feas], controller_names, mpc


def closed_loop_test_wtf(dataset, model_name, N_samples=int(1e3)):
    mpc, X, V, _, _ = mpc_dataset_import(dataset)
    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print(f"WARNING: N_samples exceeds dataset size, using N_samples={N_samples}")

    model = import_model(modelname=model_name)
    controller = SafeOnlineEvaluationAMPCGroundTruthInit(mpc, model)

    X_test, V_test = X[:N_samples], V[:N_samples]
    print(f"\ntesting controller on {len(X_test)} initial conditions in closed loop\n")
    for i in tqdm(range(N_samples)):
        controller.initialize(X_test[i], V_test[i])
        status, X_out, U, V_out, feasible = closed_loop_experiment(X_test[i], controller, Nsim=N_samples)
        if status == "infeasible_cl" and feasible[0]:
            print(f"{X_out=},{U=},{V_out=},{feasible=}")


def closed_loop_test_on_sampler(model_name, sampler, N_samples=int(1e3), N_sim=20):
    """Closed-loop simulation on sampled initial conditions."""
    mpc, model = import_model_mpc(modelname=model_name)

    naive_controller = AMPC(mpc, model)
    safe_controller = SafeOnlineEvaluationAMPC(mpc, model)

    controllers = [naive_controller, safe_controller]
    controller_names = ["naive", "safe"]

    results = []
    print(f"\ntesting controllers on {N_samples} initial conditions in closed loop\n")
    for i in tqdm(range(N_samples)):
        x0 = sampler.sample()
        simulation_results = iterate_controllers(x0, None, controllers, Nsim=N_sim)
        results.append(simulation_results)

    for j in range(len(controllers)):
        mu_cl_in_terminal_set = np.mean(np.array([
            results[i][j]["status"] == "terminal_set_reached" for i in range(N_samples)
        ]))
        mu_cl_feasible = np.mean(np.array([
            results[i][j]["status"] != "infeasible_cl" for i in range(N_samples)
        ]))
        print(f"Results for controller: {controller_names[j]}: {mu_cl_feasible=} {mu_cl_in_terminal_set=}")


def closed_loop_test_reason(dataset, model_name, N_samples=int(1e3), N_sim=200):
    mpc, X, V, _, _ = mpc_dataset_import(dataset)
    X = np.atleast_2d(X)
    V = np.asarray(V)

    if X.shape[0] != V.shape[0]:
        raise ValueError(f"Sample mismatch: X has {X.shape[0]} rows, V has {V.shape[0]} rows")

    rng = np.random.default_rng(42)
    perm = rng.permutation(X.shape[0])
    X, V = X[perm], V[perm]

    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print(f"WARNING: N_samples exceeds dataset size, using N_samples={N_samples}")

    X_test, V_test = X[:N_samples], V[:N_samples]
    model = import_model(modelname=model_name)
    controller = SafeOnlineEvaluationAMPCGroundTruthInit(mpc, model)

    results = []
    print(f"\ntesting controller on {len(X_test)} initial conditions in closed loop\n")
    for i in tqdm(range(N_samples)):
        controller.initialize(X_test[i], V_test[i])

        V_cl = np.zeros((N_sim - 1, mpc.nu))
        U_cl = np.zeros((N_sim - 1, mpc.nu))
        X_cl = np.zeros((N_sim, mpc.nx))
        X_cl[0] = X_test[i]

        for k in range(N_sim - 1):
            xk = X_cl[k]
            v, x_next = controller(xk)
            u = controller.mpc.stabilizing_feedback_controller_clipped_inputs(xk, v)

            V_cl[k] = v
            U_cl[k] = u
            X_cl[k + 1] = x_next
            results.append(controller.feasible_debug)

            feasible_cl = controller.mpc.in_state_and_input_constraints(
                X_cl[:k + 2], V_cl[:k + 1], robust=False
            )
            if not feasible_cl:
                break
            if controller.mpc.in_terminal_constraints(x_next, robust=False):
                break

    rejections = [r for r in results
                  if not (r["in_state_and_input_constraints"] and r["in_terminal_constraint"] and r["cost_decrease"])]
    rejection_rate = len(rejections) / len(results) if results else 0.0
    if rejections:
        rejection_from_state = np.mean([not r["in_state_and_input_constraints"] for r in rejections])
        rejection_from_terminal = np.mean([not r["in_terminal_constraint"] for r in rejections])
        rejection_from_cost = np.mean([not r["cost_decrease"] for r in rejections])
    else:
        rejection_from_state = rejection_from_terminal = rejection_from_cost = 0.0

    print(f"{rejection_rate=}")
    print(f"{rejection_from_state=}")
    print(f"{rejection_from_terminal=}")
    print(f"{rejection_from_cost=}")


def evaluate_naive_ampc_on_dataset(
    dataset="latest",
    model_name="latest",
    use_mpc_from_model=False,
    N_samples=None,
    batch_size=4096,
):
    """Evaluate naive AMPC MAE/MSE on dataset."""
    mpc, X, V_true, _, _ = mpc_dataset_import(dataset)

    if use_mpc_from_model:
        _, model = import_model_mpc(modelname=model_name)
    else:
        model = import_model(modelname=model_name)

    if N_samples is not None:
        N_samples = min(N_samples, X.shape[0])
        X = X[:N_samples]
        V_true = V_true[:N_samples]

    V_pred = model.predict(X, batch_size=batch_size, verbose=0)

    V_true = np.asarray(V_true, dtype=np.float32)
    V_pred = np.asarray(V_pred, dtype=np.float32)

    if "RNN" in model_name:
        V_pred = np.swapaxes(V_pred, 2, 1)

    if V_pred.shape != V_true.shape:
        raise ValueError(f"Shape mismatch: V_pred {V_pred.shape} vs V_true {V_true.shape}")

    err = V_pred - V_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))

    res = {"mae": mae, "mse": mse}
    print(res)
    return res