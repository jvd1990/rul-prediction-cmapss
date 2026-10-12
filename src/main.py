from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------
# 1. Project settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
IMAGES_DIR = PROJECT_DIR / "images"
RESULTS_DIR = PROJECT_DIR / "results"

IMAGES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
RUL_CAP = 125
MAINTENANCE_THRESHOLD = 30

# This threshold applies to FD001 in its original measurement units.
STD_THRESHOLD = 0.01

columns = ["unit", "cycle", "setting1", "setting2", "setting3"]

for number in range(1, 22):
    columns.append(f"s{number}")

measurement_columns = columns[2:]


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def read_engine_data(filename):
    """Read an engine data file and check its structure."""

    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    data = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
    )

    # Remove columns that contain only missing values.
    data = data.dropna(axis=1, how="all")

    if data.shape[1] != len(columns):
        raise ValueError(
            f"{filename}: expected 26 columns, found {data.shape[1]}"
        )

    data.columns = columns

    if data.isna().any().any():
        raise ValueError(f"{filename} contains missing values.")

    if data.duplicated(["unit", "cycle"]).any():
        raise ValueError(
            f"{filename} contains duplicate engine-cycle rows."
        )

    return data.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)


def save_plot(filename):
    """Save and close the current plot."""

    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def calculate_metrics(actual, predicted):
    """Calculate prediction errors in cycles."""

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    return float(rmse), float(mae)


# --------------------------------------------------
# 3. Load the data
# --------------------------------------------------

train = read_engine_data("train_FD001.txt")
test = read_engine_data("test_FD001.txt")

rul = pd.read_csv(
    DATA_DIR / "RUL_FD001.txt",
    sep=r"\s+",
    header=None,
)

rul = rul.dropna(axis=1, how="all")

if rul.shape[1] != 1:
    raise ValueError("The RUL file must contain one column.")

rul.columns = ["RUL"]

if rul.isna().any().any():
    raise ValueError("The RUL file contains missing values.")

# RUL rows correspond to test engines 1, 2, ..., 100.
expected_units = np.arange(1, len(rul) + 1)
test_units = np.sort(test["unit"].unique())

if not np.array_equal(test_units, expected_units):
    raise ValueError(
        "Test engine IDs do not match the RUL file rows."
    )

print("\nDATA SUMMARY")
print("Training shape:", train.shape)
print("Test shape:", test.shape)
print("Training engines:", train["unit"].nunique())
print("Test engines:", test["unit"].nunique())
print("RUL shape:", rul.shape)


# --------------------------------------------------
# 4. Analyze training engine lifetimes
# --------------------------------------------------

engine_lifetimes = train.groupby("unit")["cycle"].max()

print("\nTRAINING ENGINE LIFETIMES")
print(f"Mean: {engine_lifetimes.mean():.2f} cycles")
print(f"Minimum: {engine_lifetimes.min()} cycles")
print(f"Maximum: {engine_lifetimes.max()} cycles")

engine_lifetimes.to_csv(
    RESULTS_DIR / "engine_lifetimes.csv",
    header=["lifetime"],
)

plt.figure(figsize=(8, 5))
plt.hist(engine_lifetimes, bins=15, edgecolor="black")
plt.title("Training Engine Lifetime Distribution")
plt.xlabel("Lifetime (cycles)")
plt.ylabel("Number of engines")
save_plot("engine_lifetime_histogram.png")


# --------------------------------------------------
# 5. Create the training targets
# --------------------------------------------------

# Repeat each engine's maximum cycle across its rows.
max_cycles = train.groupby("unit")["cycle"].transform("max")

train["RUL_raw"] = max_cycles - train["cycle"]

# Keep the original target and create a capped target.
train["RUL"] = train["RUL_raw"].clip(upper=RUL_CAP)

print("\nTRAINING RUL EXAMPLES")
print(train[["unit", "cycle", "RUL_raw", "RUL"]].head())


# --------------------------------------------------
# 6. Split the data by engine
# --------------------------------------------------

# All rows of one engine must remain in the same partition.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_indices, validation_indices = next(
    splitter.split(train, groups=train["unit"])
)

training_data = train.iloc[train_indices].copy()
validation_data = train.iloc[validation_indices].copy()

training_units = set(training_data["unit"])
validation_units = set(validation_data["unit"])

if training_units.intersection(validation_units):
    raise ValueError(
        "An engine appears in both training and validation."
    )

print("\nGROUP SPLIT")
print("Training engines:", len(training_units))
print("Validation engines:", len(validation_units))
print("Training rows:", len(training_data))
print("Validation rows:", len(validation_data))


# --------------------------------------------------
# 7. Remove near-constant measurements
# --------------------------------------------------

# Determine removed features using the training partition only.
measurement_std = training_data[measurement_columns].std()

dropped_columns = measurement_std[
    measurement_std < STD_THRESHOLD
].index.tolist()

feature_columns = []

for column in measurement_columns:
    if column not in dropped_columns:
        feature_columns.append(column)

print("\nREMOVED NEAR-CONSTANT FEATURES")
print(dropped_columns)

print("\nRETAINED FEATURES")
print(feature_columns)

feature_summary = pd.DataFrame({
    "standard_deviation": measurement_std,
    "removed": measurement_std < STD_THRESHOLD,
})

feature_summary.to_csv(
    RESULTS_DIR / "feature_selection.csv"
)

# Do not use engine ID, cycle or target columns as inputs.
X_train = training_data[feature_columns]
y_train = training_data["RUL"]

X_validation = validation_data[feature_columns]
y_validation = validation_data["RUL"]


# --------------------------------------------------
# 8. Plot sensor trends
# --------------------------------------------------

selected_engines = sorted(training_units)[:3]
selected_sensors = ["s2", "s7", "s11"]

fig, axes = plt.subplots(3, 1, figsize=(10, 11))

trend_descriptions = []

for axis, sensor in zip(axes, selected_sensors):
    changes = []

    for unit in selected_engines:
        engine_data = training_data[
            training_data["unit"] == unit
        ].sort_values("cycle")

        axis.plot(
            engine_data["cycle"],
            engine_data[sensor],
            label=f"Engine {unit}",
            alpha=0.8,
        )

        # Compare early-life and late-life averages.
        start_mean = engine_data[sensor].head(20).mean()
        end_mean = engine_data[sensor].tail(20).mean()

        changes.append(end_mean - start_mean)

    if np.mean(changes) > 0:
        direction = "increases"
    else:
        direction = "decreases"

    description = (
        f"{sensor} generally {direction} near failure "
        "in the three inspected engines."
    )

    trend_descriptions.append(description)

    axis.set_title(f"{sensor} over Engine Lifetime")
    axis.set_xlabel("Cycle")
    axis.set_ylabel(sensor)
    axis.legend()
    axis.grid(alpha=0.2)

save_plot("sensor_trends.png")

print("\nSENSOR TRENDS")

for description in trend_descriptions:
    print(description)

(RESULTS_DIR / "sensor_trends.txt").write_text(
    "\n".join(trend_descriptions),
    encoding="utf-8",
)


# --------------------------------------------------
# 9. Calculate and plot correlations
# --------------------------------------------------

sensor_columns = []

for column in feature_columns:
    if column.startswith("s") and not column.startswith("setting"):
        sensor_columns.append(column)

correlation_columns = sensor_columns + ["RUL"]

correlation_matrix = training_data[
    correlation_columns
].corr()

correlation_matrix.to_csv(
    RESULTS_DIR / "correlation_matrix.csv"
)

print("\nSENSOR CORRELATIONS WITH CAPPED RUL")
print(
    correlation_matrix["RUL"]
    .drop("RUL")
    .sort_values()
    .round(3)
)

plt.figure(figsize=(12, 10))
plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    interpolation="nearest",
)

plt.colorbar(label="Pearson correlation")

positions = range(len(correlation_columns))

plt.xticks(
    positions,
    correlation_columns,
    rotation=90,
)
plt.yticks(positions, correlation_columns)

plt.title(
    "Sensor and Capped RUL Correlations - Training Partition"
)
save_plot("correlation_matrix.png")


# --------------------------------------------------
# 10. Train and validate both models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

validation_records = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    rmse, mae = calculate_metrics(
        y_validation,
        predictions,
    )

    validation_records.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
    })

validation_results = pd.DataFrame(validation_records)

validation_results.to_csv(
    RESULTS_DIR / "validation_metrics.csv",
    index=False,
)

print("\nVALIDATION RESULTS - CAPPED RUL, ALL VALIDATION ROWS")
print(
    validation_results.round(3).to_string(index=False)
)

# Select the model using validation results.
best_model_name = validation_results.loc[
    validation_results["RMSE"].idxmin(),
    "Model",
]

print("\nSelected model:", best_model_name)


# --------------------------------------------------
# 11. Refit using all training engines
# --------------------------------------------------

# Keep the feature list selected from the training partition.
X_full = train[feature_columns]
y_full = train["RUL"]

for model in models.values():
    model.fit(X_full, y_full)


# --------------------------------------------------
# 12. Evaluate the last cycle of each test engine
# --------------------------------------------------

test_last = (
    test.sort_values(["unit", "cycle"])
    .groupby("unit")
    .tail(1)
    .sort_values("unit")
    .reset_index(drop=True)
)

X_test = test_last[feature_columns]

# Evaluate against original test RUL values without clipping.
y_test = rul["RUL"].to_numpy()

test_predictions = {}
test_records = []

for name, model in models.items():
    predictions = model.predict(X_test)

    test_predictions[name] = predictions

    rmse, mae = calculate_metrics(
        y_test,
        predictions,
    )

    test_records.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
    })

test_results = pd.DataFrame(test_records)

test_results.to_csv(
    RESULTS_DIR / "test_metrics.csv",
    index=False,
)

print("\nTEST RESULTS - ORIGINAL RUL, ONE ROW PER ENGINE")
print(test_results.round(3).to_string(index=False))

# Use the validation-selected model for maintenance decisions.
selected_predictions = test_predictions[best_model_name]

prediction_table = pd.DataFrame({
    "unit": test_last["unit"].to_numpy(),
    "last_cycle": test_last["cycle"].to_numpy(),
    "actual_RUL": y_test,
    "predicted_RUL": selected_predictions,
})


# --------------------------------------------------
# 13. Plot predicted versus actual RUL
# --------------------------------------------------

lower_limit = min(
    0,
    y_test.min(),
    selected_predictions.min(),
)

upper_limit = max(
    y_test.max(),
    selected_predictions.max(),
) + 10

plt.figure(figsize=(7, 7))
plt.scatter(
    y_test,
    selected_predictions,
    alpha=0.75,
    edgecolors="black",
)

plt.plot(
    [lower_limit, upper_limit],
    [lower_limit, upper_limit],
    "r--",
    label="Perfect prediction: y = x",
)

plt.xlim(lower_limit, upper_limit)
plt.ylim(lower_limit, upper_limit)
plt.gca().set_aspect("equal", adjustable="box")

plt.xlabel("Actual RUL (cycles)")
plt.ylabel("Predicted RUL (cycles)")
plt.title(f"Test RUL Predictions - {best_model_name}")
plt.legend()
plt.grid(alpha=0.2)

save_plot("predicted_vs_actual.png")


# --------------------------------------------------
# 14. Plot Random Forest feature importance
# --------------------------------------------------

forest = models["Random Forest"]

feature_importance = pd.Series(
    forest.feature_importances_,
    index=feature_columns,
).sort_values()

feature_importance.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    header=["importance"],
)

plt.figure(figsize=(9, 7))
plt.barh(
    feature_importance.index,
    feature_importance.values,
)
plt.xlabel("Feature importance")
plt.title("Random Forest Feature Importance")

save_plot("feature_importance.png")


# --------------------------------------------------
# 15. Analyze maintenance decisions
# --------------------------------------------------

actual_maintenance = (
    y_test <= MAINTENANCE_THRESHOLD
)

predicted_maintenance = (
    selected_predictions <= MAINTENANCE_THRESHOLD
)

matrix = confusion_matrix(
    actual_maintenance,
    predicted_maintenance,
    labels=[False, True],
)

true_negative, false_positive, false_negative, true_positive = (
    matrix.ravel()
)

print("\nMAINTENANCE DECISIONS")
print("Model:", best_model_name)
print("Correct alerts:", true_positive)
print("False alerts:", false_positive)
print("Missed maintenance needs:", false_negative)
print("Correct no-alert decisions:", true_negative)

maintenance_results = pd.DataFrame({
    "Outcome": [
        "Correct alert",
        "False alert",
        "Missed maintenance need",
        "Correct no-alert decision",
    ],
    "Count": [
        true_positive,
        false_positive,
        false_negative,
        true_negative,
    ],
})

maintenance_results.to_csv(
    RESULTS_DIR / "maintenance_results.csv",
    index=False,
)

prediction_table["actual_maintenance"] = actual_maintenance
prediction_table["predicted_maintenance"] = predicted_maintenance

prediction_table.to_csv(
    RESULTS_DIR / "test_predictions.csv",
    index=False,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["No maintenance", "Maintenance"],
)

fig, axis = plt.subplots(figsize=(7, 6))

display.plot(
    ax=axis,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

axis.set_title(
    f"Maintenance Decisions - Threshold "
    f"{MAINTENANCE_THRESHOLD} Cycles"
)

save_plot("maintenance_confusion_matrix.png")


# --------------------------------------------------
# 16. Finish
# --------------------------------------------------

# README and configuration files are maintained separately.
print("\nPROJECT FINISHED")
print("Plots:", IMAGES_DIR)
print("Results:", RESULTS_DIR)