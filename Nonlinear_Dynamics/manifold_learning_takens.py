"""
=====================================================================
Latent Dynamical Systems Learning from Takens Delay Reconstructions
=====================================================================

Overview
--------
This notebook implements an end-to-end framework for learning a low-
dimensional latent representation of a dynamical system from a scalar
time series using Takens delay-coordinate embeddings and neural latent
dynamics.

The primary objective is not merely dimensionality reduction, but the
construction of a latent state space which is simultaneously:

    1. Reconstructive
       - The latent representation should preserve information about
         the original delay-coordinate state.

    2. Predictive
       - The latent representation should evolve according to a learned
         continuous-time latent dynamical system capable of forecasting
         future states.

    3. Geometrically meaningful
       - The latent space should preserve neighborhood structure and
         large-scale geometry of the reconstructed attractor.

Methodology
-----------
The workflow is as follows:

    Scalar time series
            │
            ▼
    Takens delay embedding
            │
            ▼
    x_t ∈ R^m
            │
            ▼
         Encoder
            │
            ▼
    z_t ∈ R^d
            │
            ▼
    Learned latent ODE
        dz/dt = g(z)
            │
            ▼
    Forward integration
            │
            ▼
      z_(t+k)
            │
            ▼
         Decoder
            │
            ▼
      x̂_(t+k)

The encoder learns a nonlinear mapping from the reconstructed Takens
state x_t to a latent coordinate z_t.

The latent vector field g(z) is represented by a neural network and
integrated forward in time using a differentiable RK4 scheme.

The decoder maps latent states back to the reconstructed Takens space.

Training Objective
------------------
The model is trained jointly using two losses:

    L = L_recon + λ L_forecast

where

    Reconstruction loss:

        L_recon
        = || x_t - x̂_t ||²

encourages the latent representation to preserve information about the
current state.

and

    Forecast loss:

        L_forecast
        = || x_(t+k) - x̂_(t+k) ||²

encourages the latent dynamics to evolve in a way that accurately
predicts future states.

The parameter λ controls the trade-off between reconstruction fidelity
and forecasting performance.

Embedding Quality Assessment
----------------------------
After training, the learned embedding is evaluated using several
complementary metrics:

1. Reconstruction Error
   Measures how accurately the decoder reconstructs the current state.

2. Forecast Error
   Measures how accurately future states can be predicted.

3. Pairwise Distance Correlation
   Quantifies preservation of global geometry between the Takens
   reconstruction and the latent representation.

4. k-Nearest-Neighbor Overlap
   Quantifies preservation of local neighborhood structure.

5. Forecast MSE versus Horizon
   Compares long-range predictive performance against a baseline model.

Baseline Comparison
-------------------
Forecast performance is compared against a persistence baseline:

    x̂_(t+k) = x_t

which assumes no evolution of the state.

A useful latent representation should significantly outperform this
baseline over a range of forecast horizons.

Dynamical Analysis
------------------
Once training is complete, the latent trajectories are analyzed
independently of the training process.

The largest Lyapunov exponent is estimated using Wolf's algorithm on
the latent trajectory:

    z_0, z_1, z_2, ...

This estimate is NOT used as a training objective.

Instead, it serves as a diagnostic tool for assessing the degree to
which the learned latent dynamics preserve the instability and
predictability properties of the original dynamical system.

Interpretation
--------------
A successful model should exhibit:

    • Low reconstruction error
    • Low forecast error
    • Strong geometric preservation
    • Forecast performance superior to persistence
    • Latent trajectories with meaningful dynamical structure

The emphasis of this notebook is on learning useful predictive state
representations rather than exact dynamical conjugacies to the original
system.

=====================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1) Takens embedding
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_training_losses(history_obj):
    hist = history_obj.history

    plt.figure(figsize=(8, 5))
    if "loss" in hist:
        plt.plot(hist["loss"], label="train loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val loss")
    if "recon_loss" in hist:
        plt.plot(hist["recon_loss"], label="train recon")
    if "val_recon_loss" in hist:
        plt.plot(hist["val_recon_loss"], label="val recon")
    if "forecast_loss" in hist:
        plt.plot(hist["forecast_loss"], label="train forecast")
    if "val_forecast_loss" in hist:
        plt.plot(hist["val_forecast_loss"], label="val forecast")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def predict_future_from_window(joint_model, x_hist_window, forecast_horizon=None):
    """
    x_hist_window: shape (1, history, state_dim)
    """
    if forecast_horizon is None:
        forecast_horizon = joint_model.horizon

    z_hist = joint_model.encode_seq(tf.convert_to_tensor(x_hist_window, dtype=tf.float32), training=False)
    z0 = z_hist[:, -1, :]
    z_future = rollout_latent(z0, forecast_horizon, joint_model.dt, joint_model.vector_field)
    x_hat_future = joint_model.decoder(z_future, training=False).numpy()
    return x_hat_future


def plot_one_step_forecast_example(joint_model, x_hist, x_future, sample_idx=0, coord=0):
    """
    Plots history + true one-step future + predicted one-step future
    for one selected coordinate.
    """
    xh = x_hist[sample_idx:sample_idx + 1]
    xf_true = x_future[sample_idx]

    x_hat_future = predict_future_from_window(joint_model, xh, forecast_horizon=1)[0]
    mse = float(np.mean((xf_true - x_hat_future) ** 2))

    hist_coord = xh[0, :, coord]
    true_next = xf_true[coord]
    pred_next = x_hat_future[coord]

    t_hist = np.arange(len(hist_coord))
    t_next = len(hist_coord)

    plt.figure(figsize=(9, 4))
    plt.plot(t_hist, hist_coord, label="history", linewidth=2)
    plt.scatter([t_next], [true_next], label="true next", s=70)
    plt.scatter([t_next], [pred_next], label="predicted next", s=70)

    plt.title(f"One-step forecast on coordinate {coord} | MSE = {mse:.6e}")
    plt.xlabel("time index")
    plt.ylabel("value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"One-step forecast MSE = {mse:.6e}")
    return mse


def horizon_forecast_mse_curve(joint_model, emb, history, horizons, stride=1, test_frac=0.2):
    """
    Compare learned model forecast MSE vs raw Takens persistence baseline
    as a function of forecast horizon.
    """
    model_mse = []
    baseline_mse = []

    emb = np.asarray(emb, dtype=np.float32)
    split = int(len(emb) * (1.0 - test_frac))
    emb_test = emb[split:]

    for h in horizons:
        x_hist, x_future = make_windows(emb_test, history=history, horizon=h, stride=stride)
        if len(x_hist) == 0:
            model_mse.append(np.nan)
            baseline_mse.append(np.nan)
            continue

        # Model forecast
        x_hat_future = []
        batch_size = 256
        for i in range(0, len(x_hist), batch_size):
            batch = x_hist[i:i + batch_size]
            z_hist = encode_windows(joint_model.encoder, batch)
            z0 = z_hist[:, -1, :]
            z_future = rollout_latent(
                tf.convert_to_tensor(z0, dtype=tf.float32),
                horizon=h,
                dt=joint_model.dt,
                vf=joint_model.vector_field,
            ).numpy()
            x_hat_future.append(joint_model.decoder(z_future, training=False).numpy())
        x_hat_future = np.concatenate(x_hat_future, axis=0)

        mse_model = float(np.mean((x_future - x_hat_future) ** 2))

        # Raw Takens-space baseline: persistence forecast
        x_hat_base = x_hist[:, -1, :]
        mse_base = float(np.mean((x_future - x_hat_base) ** 2))

        model_mse.append(mse_model)
        baseline_mse.append(mse_base)

    return np.asarray(model_mse), np.asarray(baseline_mse)


def plot_horizon_curve(horizons, model_mse, baseline_mse):
    plt.figure(figsize=(8, 5))
    plt.plot(horizons, model_mse, "o-", label="learned latent model")
    plt.plot(horizons, baseline_mse, "o-", label="raw Takens persistence baseline")
    plt.xlabel("Forecast horizon")
    plt.ylabel("MSE")
    plt.title("Forecast MSE vs horizon")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def takens_embedding(dat, tau, m):
    dat = np.asarray(dat, dtype=np.float32).reshape(-1)
    if len(dat) <= (m - 1) * tau:
        raise ValueError("Signal too short for the requested Takens embedding.")

    delay_vecs = []
    for i in range((m - 1) * tau, len(dat)):
        delay_vec = []
        for j in range(0, m * tau, tau):
            delay_vec.append(dat[i - j])
        delay_vecs.append(delay_vec)

    return np.asarray(delay_vecs, dtype=np.float32)


def make_windows(traj: np.ndarray, history: int, horizon: int, stride: int = 1):
    """
    traj: shape (N, d)
    returns:
      x_hist:   (M, history, d)
      x_future: (M, d)
    """
    traj = np.asarray(traj, dtype=np.float32)
    x_hist, x_future = [], []

    stop = len(traj) - history - horizon + 1
    for i in range(0, stop, stride):
        x_hist.append(traj[i : i + history])
        x_future.append(traj[i + history + horizon - 1])

    return np.asarray(x_hist, dtype=np.float32), np.asarray(x_future, dtype=np.float32)


def chronological_split(x_hist, x_future, train_frac=0.7, val_frac=0.15):
    n = len(x_hist)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    x_hist_train = x_hist[:n_train]
    x_future_train = x_future[:n_train]

    x_hist_val = x_hist[n_train : n_train + n_val]
    x_future_val = x_future[n_train : n_train + n_val]

    x_hist_test = x_hist[n_train + n_val :]
    x_future_test = x_future[n_train + n_val :]

    return (
        x_hist_train, x_future_train,
        x_hist_val, x_future_val,
        x_hist_test, x_future_test,
    )


# ============================================================
# 2) Model components
# ============================================================

def build_encoder(state_dim: int, latent_dim: int, hidden_dim: int = 64) -> keras.Model:
    x_in = keras.Input(shape=(state_dim,))
    x = layers.Dense(hidden_dim, activation="tanh")(x_in)
    x = layers.Dense(hidden_dim, activation="tanh")(x)
    z = layers.Dense(latent_dim, name="z")(x)
    return keras.Model(x_in, z, name="encoder")


def build_decoder(state_dim: int, latent_dim: int, hidden_dim: int = 64) -> keras.Model:
    z_in = keras.Input(shape=(latent_dim,))
    x = layers.Dense(hidden_dim, activation="tanh")(z_in)
    x = layers.Dense(hidden_dim, activation="tanh")(x)
    x_out = layers.Dense(state_dim, name="x_hat")(x)
    return keras.Model(z_in, x_out, name="decoder")


def build_vector_field(latent_dim: int, hidden_dim: int = 64) -> keras.Model:
    z_in = keras.Input(shape=(latent_dim,))
    x = layers.Dense(hidden_dim, activation="tanh")(z_in)
    x = layers.Dense(hidden_dim, activation="tanh")(x)
    z_dot = layers.Dense(latent_dim, name="z_dot")(x)
    return keras.Model(z_in, z_dot, name="vector_field")


# ============================================================
# 3) Latent RK4 rollout
# ============================================================

def tf_rk4_step(z: tf.Tensor, h: tf.Tensor, vf: keras.Model) -> tf.Tensor:
    k1 = vf(z)
    k2 = vf(z + 0.5 * h * k1)
    k3 = vf(z + 0.5 * h * k2)
    k4 = vf(z + h * k3)
    return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_latent(z0: tf.Tensor, horizon: int, dt: float, vf: keras.Model) -> tf.Tensor:
    z = z0
    h = tf.convert_to_tensor(dt, dtype=tf.float32)
    for _ in range(horizon):
        z = tf_rk4_step(z, h, vf)
    return z


def encode_windows(encoder: keras.Model, x_hist: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    x_hist: (B, history, state_dim)
    returns z_hist: (B, history, latent_dim)
    """
    B, T, D = x_hist.shape
    flat = x_hist.reshape(B * T, D)
    z_flat = encoder.predict(flat, batch_size=batch_size, verbose=0)
    return z_flat.reshape(B, T, -1).astype(np.float32)


# ============================================================
# 4) Validation metrics for embedding quality
# ============================================================

def pairwise_distance_corr(a: np.ndarray, b: np.ndarray, n_pairs: int = 5000, seed: int = 0) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = len(a)
    if n < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, n, size=n_pairs)
    idx2 = rng.integers(0, n, size=n_pairs)

    da = np.linalg.norm(a[idx1] - a[idx2], axis=1)
    db = np.linalg.norm(b[idx1] - b[idx2], axis=1)

    if np.std(da) < 1e-12 or np.std(db) < 1e-12:
        return np.nan

    return float(np.corrcoef(da, db)[0, 1])


def knn_overlap_score(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = len(a)
    if n <= k:
        return np.nan

    nn_a = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(return_distance=False)[:, 1:]
    nn_b = NearestNeighbors(n_neighbors=k + 1).fit(b).kneighbors(return_distance=False)[:, 1:]

    overlap = []
    for i in range(n):
        overlap.append(len(set(nn_a[i]).intersection(set(nn_b[i]))) / k)
    return float(np.mean(overlap))


class EmbeddingValidationCallback(keras.callbacks.Callback):
    def __init__(
        self,
        x_val,
        x_future_val,
        sample_size=256,
        k=5,
        seed=0,
    ):
        super().__init__()
        self.x_val = x_val
        self.x_future_val = x_future_val
        self.sample_size = sample_size
        self.k = k
        self.rng = np.random.default_rng(seed)

    def on_epoch_end(self, epoch, logs=None):
        n = len(self.x_val)
        m = min(self.sample_size, n)
        idx = self.rng.choice(n, size=m, replace=False)

        x = self.x_val[idx]
        x_future = self.x_future_val[idx]

        z_hist = encode_windows(self.model.encoder, x)
        z_flat = z_hist.reshape(len(z_hist), -1)
        x_flat = x.reshape(len(x), -1)

        z0 = z_hist[:, -1, :]
        z_future = rollout_latent(
            tf.convert_to_tensor(z0, dtype=tf.float32),
            horizon=self.model.horizon,
            dt=self.model.dt,
            vf=self.model.vector_field,
        ).numpy()

        x_hat_now = self.model.decoder(tf.convert_to_tensor(z0, dtype=tf.float32), training=False).numpy()
        x_hat_future = self.model.decoder(tf.convert_to_tensor(z_future, dtype=tf.float32), training=False).numpy()

        recon_mse = float(np.mean((x[:, -1, :] - x_hat_now) ** 2))
        forecast_mse = float(np.mean((x_future - x_hat_future) ** 2))

        geom_corr = pairwise_distance_corr(x_flat, z_flat, n_pairs=2000, seed=epoch)
        knn_overlap = knn_overlap_score(x_flat, z_flat, k=self.k)

        print(
            f"\nval_recon={recon_mse:.4e} "
            f"val_forecast={forecast_mse:.4e} "
            f"geom_corr={geom_corr:.3f} "
            f"knn_overlap={knn_overlap:.3f}"
        )


# ============================================================
# 5) Joint model: recon + forecast
# ============================================================

class JointDynamicsModel(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        vector_field: keras.Model,
        history: int,
        horizon: int,
        dt: float,
        lambda_forecast: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vector_field = vector_field
        self.history = history
        self.horizon = horizon
        self.dt = dt
        self.lambda_forecast = lambda_forecast

        self.encode_seq = layers.TimeDistributed(self.encoder)
        self.flatten = layers.Flatten()

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_tracker = keras.metrics.Mean(name="recon_loss")
        self.forecast_tracker = keras.metrics.Mean(name="forecast_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.forecast_tracker]

    def call(self, inputs, training=False):
        x_hist = inputs
        z_hist = self.encode_seq(x_hist, training=training)
        z0 = z_hist[:, -1, :]
        z_future = rollout_latent(z0, self.horizon, self.dt, self.vector_field)
        x_hat_now = self.decoder(z0, training=training)
        x_hat_future = self.decoder(z_future, training=training)
        return x_hat_now, x_hat_future

    def train_step(self, data):
        x_hist, x_future = data
        with tf.GradientTape() as tape:
            z_hist = self.encode_seq(x_hist, training=True)
            z0 = z_hist[:, -1, :]
            z_future = rollout_latent(z0, self.horizon, self.dt, self.vector_field)

            x_hat_now = self.decoder(z0, training=True)
            x_hat_future = self.decoder(z_future, training=True)

            recon_loss = tf.reduce_mean(tf.square(x_hist[:, -1, :] - x_hat_now))
            forecast_loss = tf.reduce_mean(tf.square(x_future - x_hat_future))
            loss = recon_loss + self.lambda_forecast * forecast_loss

        vars_ = (
            self.encoder.trainable_variables
            + self.decoder.trainable_variables
            + self.vector_field.trainable_variables
        )
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.forecast_tracker.update_state(forecast_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_hist, x_future = data
        z_hist = self.encode_seq(x_hist, training=False)
        z0 = z_hist[:, -1, :]
        z_future = rollout_latent(z0, self.horizon, self.dt, self.vector_field)

        x_hat_now = self.decoder(z0, training=False)
        x_hat_future = self.decoder(z_future, training=False)

        recon_loss = tf.reduce_mean(tf.square(x_hist[:, -1, :] - x_hat_now))
        forecast_loss = tf.reduce_mean(tf.square(x_future - x_hat_future))
        loss = recon_loss + self.lambda_forecast * forecast_loss

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.forecast_tracker.update_state(forecast_loss)

        return {m.name: m.result() for m in self.metrics}


# ============================================================
# 6) Wolf's algorithm separate after training
# ============================================================

def wolf_largest_lyapunov(
    traj,
    t3=5,
    scale_min=0.05,
    scale_max=np.inf,
    angle_max=0.3,
    use_angle=False,
    dt=1.0,
):
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim == 1:
        traj = traj[:, None]

    n, dim = traj.shape
    if n < t3 + 2:
        raise ValueError("Trajectory is too short.")

    def distance(i, j):
        return np.linalg.norm(traj[i] - traj[j])

    def nearest_neighbor(base_index, excluded=None):
        if excluded is None:
            excluded = set()

        dmin = np.inf
        adj_index = None

        for j in range(n):
            if j == base_index or j in excluded:
                continue
            d = distance(base_index, j)
            if scale_min < d < dmin:
                dmin = d
                adj_index = j

        return adj_index, dmin

    def angle_ok(base_index, adj_index, prev_sep):
        new_sep = traj[adj_index] - traj[base_index]
        n1 = np.linalg.norm(prev_sep)
        n2 = np.linalg.norm(new_sep)
        if n1 == 0.0 or n2 == 0.0:
            return False
        cosang = np.dot(prev_sep, new_sep) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.arccos(cosang) <= angle_max

    sum_log = 0.0
    count = 0
    discarded = set()
    prev_sep = None

    for base_index in range(0, n - t3, t3):
        if base_index + t3 >= n:
            break

        if prev_sep is None or not use_angle:
            adj_index, len_min = nearest_neighbor(base_index)
        else:
            attempts = 0
            adj_index = None
            len_min = np.inf
            while attempts < n:
                adj_index, len_min = nearest_neighbor(base_index, excluded=discarded)
                if adj_index is None:
                    break
                if angle_ok(base_index, adj_index, prev_sep):
                    discarded.clear()
                    break
                discarded.add(adj_index)
                attempts += 1

        if adj_index is None or not np.isfinite(len_min) or len_min <= 0:
            continue

        base_next = base_index + t3
        adj_next = adj_index + t3
        if base_next >= n or adj_next >= n:
            break

        len_max = distance(base_next, adj_next)
        prev_sep = traj[adj_next] - traj[base_next]

        if len_max >= len_min and len_max < scale_max:
            sum_log += np.log(len_max / len_min)
            count += 1

    if count == 0:
        raise RuntimeError("No valid divergence segments were found.")

    lle = sum_log / (count * t3 * dt)
    print(f"largest Lyapunov exponent = {lle:.6f}")
    print(f"segments used = {count}")
    return lle


# ============================================================
# 7) Main pipeline
# ============================================================

def main():
    # Load scalar signal
    data = np.loadtxt("amplitude.dat", dtype=float)

    # Takens parameters
    tau = 8
    m = 7
    emb = takens_embedding(data, tau=tau, m=m)

    # Make supervised windows for training
    history = 32
    horizon = 4
    stride = 1
    x_hist, x_future = make_windows(emb, history=history, horizon=horizon, stride=stride)

    (
        x_hist_train, x_future_train,
        x_hist_val, x_future_val,
        x_hist_test, x_future_test,
    ) = chronological_split(x_hist, x_future)

    state_dim = emb.shape[1]
    latent_dim = 3
    hidden_dim = 64

    encoder = build_encoder(state_dim, latent_dim, hidden_dim)
    decoder = build_decoder(state_dim, latent_dim, hidden_dim)
    vector_field = build_vector_field(latent_dim, hidden_dim)

    joint = JointDynamicsModel(
        encoder=encoder,
        decoder=decoder,
        vector_field=vector_field,
        history=history,
        horizon=horizon,
        dt=1.0,  # replace with your true sampling interval
        lambda_forecast=1.0,
    )
    joint.compile(optimizer=keras.optimizers.Adam(1e-3))

    train_ds = tf.data.Dataset.from_tensor_slices((x_hist_train, x_future_train))
    train_ds = train_ds.shuffle(min(len(x_hist_train), 4096)).batch(64).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_hist_val, x_future_val))
    val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
        ),
        EmbeddingValidationCallback(
            x_val=x_hist_val,
            x_future_val=x_future_val,
            sample_size=256,
            k=5,
        ),
    ]

    history_obj = joint.fit(
        train_ds,
        validation_data=val_ds,
        epochs=200,
        callbacks=callbacks,
        verbose=1,
    )

    # Plot training curves
    plot_training_losses(history_obj)

    print("\nTest-set evaluation:")
    test_metrics = joint.evaluate(
        tf.data.Dataset.from_tensor_slices((x_hist_test, x_future_test)).batch(64),
        verbose=1,
        return_dict=True,
    )
    print(test_metrics)

    # Additional held-out embedding-quality summary
    z_hist_test = encode_windows(encoder, x_hist_test)
    x_flat_test = x_hist_test.reshape(len(x_hist_test), -1)
    z_flat_test = z_hist_test.reshape(len(z_hist_test), -1)

    geom_corr = pairwise_distance_corr(x_flat_test, z_flat_test, n_pairs=5000)
    knn_overlap = knn_overlap_score(x_flat_test, z_flat_test, k=5)

    print("\nEmbedding-quality summary on test set:")
    print(f"pairwise distance correlation = {geom_corr:.3f}")
    print(f"kNN overlap                   = {knn_overlap:.3f}")

    # One-step forecast plot on a held-out one-step test set
    x_hist_1, x_future_1 = make_windows(emb, history=history, horizon=1, stride=stride)
    (
        x_hist_1_train, x_future_1_train,
        x_hist_1_val, x_future_1_val,
        x_hist_1_test, x_future_1_test,
    ) = chronological_split(x_hist_1, x_future_1)

    plot_one_step_forecast_example(
        joint_model=joint,
        x_hist=x_hist_1_test,
        x_future=x_future_1_test,
        sample_idx=0,
        coord=0,
    )

    # Forecast MSE vs horizon, compared with raw Takens-space baseline
    horizons = np.arange(1, 21)
    model_mse, baseline_mse = horizon_forecast_mse_curve(
        joint_model=joint,
        emb=emb,
        history=history,
        horizons=horizons,
        stride=stride,
        test_frac=0.2,
    )
    plot_horizon_curve(horizons, model_mse, baseline_mse)

    print("Horizon-wise model MSE:", model_mse)
    print("Horizon-wise baseline MSE:", baseline_mse)

    # Wolf after training, separate from the joint loss
    latent_series = encoder.predict(emb.astype(np.float32), verbose=0)
    lle = wolf_largest_lyapunov(
        latent_series,
        dt=1.0,   # replace with true sample spacing if known
        t3=5,
        scale_min=0.05,
        scale_max=np.inf,
        use_angle=False,
    )
    print("Wolf largest Lyapunov exponent of embedding space:", lle)

if __name__ == "__main__":
    main()