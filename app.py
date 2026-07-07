from flask import Flask, request, jsonify, render_template
from model import BenchmarkModel
from dat import (
    create_cpu_benchmark_histogram,
    create_cpu_boxplot,
    create_cpu_category_bar,
    create_cpu_category_pie,
    create_cores_vs_tdp_scatter,
    create_gpu_benchmark_histogram,
    create_gpu_boxplot,
    create_gpu_category_bar,
    create_gpu_category_pie,
    create_tdp_vs_price_scatter,
)
import numpy as np
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

# Create the Flask application and wire in the benchmark prediction models.
app = Flask(__name__)
# Use the shared benchmark model for CPU predictions and recommendations.
model = BenchmarkModel()
# Track whether the CPU model has been trained successfully.
trained = False
# File names and feature definitions for the CPU dataset.
DATA_FILE = "CPU_benchmark_v4.csv"
FEATURES = ["cores", "TDP"]
CLASS_LABELS = ["Low-end", "Mid-end", "High-end"]
CATEGORY_THRESHOLDS = [10000.0, 18000.0]
# Store normalized CPU metadata after loading the dataset.
cpu_info = []
# Store normalized GPU metadata after loading the dataset.
gpu_info = []
# Use a second benchmark model instance for GPU-specific predictions.
gpu_model = BenchmarkModel()
# Track whether the GPU model has been trained successfully.
gpu_trained = False
# File names and feature definitions for the GPU datasets.
GPU_DATA_FILE = "GPU_benchmarks_v7.csv"
ADDITIONAL_GPU_DATA_FILE = "GPU_CPU_benchmark.csv"
GPU_FEATURES = ["TDP", "price"]
GPU_CLASS_LABELS = ["Entry-level", "Mid-range", "High-end"]
GPU_CATEGORY_THRESHOLDS = [12000.0, 24000.0]


def initialize_model():
    # Train the CPU models from the CSV data when the app starts.
    global trained, cpu_info, gpu_trained, gpu_info
    X, X_knn, y_reg, y_class, info = load_cpu_dataset()
    if X.size == 0 or y_reg.size == 0 or y_class.size == 0:
        raise RuntimeError("No valid rows found in CPU dataset to train.")
    cpu_info = info
    model.fit(X, y_reg, y_class, X_knn)
    trained = True
    # Initialize the GPU models immediately after the CPU model is ready.
    initialize_gpu_model()


def initialize_gpu_model():
    # Train the GPU models from the CSV data once the CPU setup completes.
    global gpu_trained, gpu_info
    X, X_knn, y_reg, y_class, info = load_gpu_dataset()
    if X.size == 0 or y_reg.size == 0 or y_class.size == 0:
        raise RuntimeError("No valid rows found in GPU dataset to train.")
    gpu_info = info
    gpu_model.fit(X, y_reg, y_class, X_knn)
    gpu_trained = True


def parse_float(value):
    # Convert a CSV field into a float while tolerating empty or missing values.
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_cpu_dataset():
    # Read CPU benchmark data from the local CSV file and build the training features.
    file_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
    X = []
    y_reg = []
    y_class = []
    info = []
    with open(file_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        X_knn = []
        for row in reader:
            target = parse_float(row.get("cpuMark"))
            if target is None:
                continue
            feature_values = [parse_float(row.get(col)) for col in FEATURES]
            if any(v is None for v in feature_values):
                continue
            cpu_name = row.get("cpuName", "Unknown CPU").strip()
            category = CLASS_LABELS[model.classify_score(target, CATEGORY_THRESHOLDS)]
            X.append(feature_values)
            X_knn.append([target, feature_values[0], feature_values[1]])
            y_reg.append(target)
            y_class.append(model.classify_score(target, CATEGORY_THRESHOLDS))
            info.append({
                "name": cpu_name,
                "cpuMark": target,
                "cores": feature_values[0],
                "TDP": feature_values[1],
                "category": category,
            })
    X_knn_arr = np.array(X_knn, dtype=float) if len(X_knn) > 0 else np.empty((0,3), dtype=float)
    return np.array(X, dtype=float), X_knn_arr, np.array(y_reg, dtype=float), np.array(y_class, dtype=int), info


def compute_model_metrics():
    # Evaluate the current CPU modeling approach with a simple holdout split.
    X, _, y_reg, y_class, _ = load_cpu_dataset()
    if X.size == 0 or y_reg.size == 0 or y_class.size == 0:
        return {}

    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_reg_train)
    regression_pred = regression_model.predict(X_test)

    knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    knn_model.fit(X_train, y_class_train)
    knn_pred = knn_model.predict(X_test)

    tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree_model.fit(X_train, y_class_train)
    tree_pred = tree_model.predict(X_test)

    return {
        "linear_regression": {
            "r2": f"{r2_score(y_reg_test, regression_pred):.4f}",
            "mae": f"{mean_absolute_error(y_reg_test, regression_pred):.2f}",
            "rmse": f"{np.sqrt(mean_squared_error(y_reg_test, regression_pred)):.2f}",
        },
        "knn_classifier": {
            "accuracy": f"{accuracy_score(y_class_test, knn_pred):.4f}",
        },
        "decision_tree": {
            "accuracy": f"{accuracy_score(y_class_test, tree_pred):.4f}",
        },
    }


def load_gpu_dataset():
    # Read GPU benchmark data from the primary file and enrich it with the secondary dataset.
    file_path = os.path.join(os.path.dirname(__file__), GPU_DATA_FILE)
    X = []
    y_reg = []
    y_class = []
    info = []
    with open(file_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        X_knn = []
        for row in reader:
            target = parse_float(row.get("G3Dmark"))
            if target is None:
                continue
            feature_values = [parse_float(row.get(col)) for col in GPU_FEATURES]
            if any(v is None for v in feature_values):
                continue
            gpu_name = row.get("gpuName", "Unknown GPU").strip()
            category = GPU_CLASS_LABELS[gpu_model.classify_score(target, GPU_CATEGORY_THRESHOLDS)]
            X.append(feature_values)
            X_knn.append([target, feature_values[0], feature_values[1]])
            y_reg.append(target)
            y_class.append(gpu_model.classify_score(target, GPU_CATEGORY_THRESHOLDS))
            info.append({
                "name": gpu_name,
                "G3Dmark": target,
                "TDP": feature_values[0],
                "price": feature_values[1],
                "category": category,
                "source": GPU_DATA_FILE,
            })

    additional_file_path = os.path.join(os.path.dirname(__file__), ADDITIONAL_GPU_DATA_FILE)
    with open(additional_file_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            target = parse_float(row.get("Median Score"))
            if target is None:
                continue
            gpu_name = row.get("Device Name", "Unknown Device").strip()
            category = row.get("Compute Type", "Unknown")
            feature_values = [0.0, 0.0]
            X.append(feature_values)
            X_knn.append([target, feature_values[0], feature_values[1]])
            y_reg.append(target)
            y_class.append(gpu_model.classify_score(target, GPU_CATEGORY_THRESHOLDS))
            info.append({
                "name": gpu_name,
                "G3Dmark": target,
                "TDP": 0.0,
                "price": 0.0,
                "category": category,
                "source": ADDITIONAL_GPU_DATA_FILE,
            })

    X_knn_arr = np.array(X_knn, dtype=float) if len(X_knn) > 0 else np.empty((0,3), dtype=float)
    return np.array(X, dtype=float), X_knn_arr, np.array(y_reg, dtype=float), np.array(y_class, dtype=int), info


def get_similar_cpus(feature_values, k=5):
    # Use the trained k-NN model to find CPUs with similar benchmark characteristics.
    distances, indices = model.find_neighbors(feature_values, k=k)
    result = []
    for dist, idx in zip(distances[0], indices[0]):
        entry = cpu_info[idx]
        result.append({
            "name": entry["name"],
            "cpuMark": entry["cpuMark"],
            "cores": entry["cores"],
            "TDP": entry["TDP"],
            "category": entry["category"],
            "distance": float(dist),
        })
    return result


def find_cpu_by_name(name):
    # Match a user-provided CPU name against the catalog, preferring exact matches.
    if name is None:
        return None
    normalized = name.strip().casefold()
    best_match = None
    for index, entry in enumerate(cpu_info):
        if entry["name"].casefold() == normalized:
            best_match = {**entry, "index": index}
            break
    if best_match:
        return best_match
    for index, entry in enumerate(cpu_info):
        if normalized in entry["name"].casefold():
            best_match = {**entry, "index": index}
            break
    return best_match


def recommend_cpus(cpu_name, k=5):
    # Find a matching CPU by name and then return the closest alternatives.
    selected = find_cpu_by_name(cpu_name)
    if selected is None:
        return None

    feature_values = [selected["cpuMark"], selected["cores"], selected["TDP"]]
    n_neighbors = min(len(cpu_info), k + 1)
    distances, indices = model.find_neighbors(feature_values, k=n_neighbors)
    recommendations = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == selected["index"]:
            continue
        entry = cpu_info[idx]
        recommendations.append({
            "name": entry["name"],
            "cpuMark": entry["cpuMark"],
            "cores": entry["cores"],
            "TDP": entry["TDP"],
            "category": entry["category"],
            "distance": float(dist),
        })
        if len(recommendations) >= k:
            break
    return {"selected_cpu": selected, "recommended_cpus": recommendations}


def get_similar_gpus(feature_values, k=5):
    # Use the GPU k-NN model to recommend similar graphics cards by benchmark profile.
    distances, indices = gpu_model.find_neighbors(feature_values, k=k)
    result = []
    for dist, idx in zip(distances[0], indices[0]):
        entry = gpu_info[idx]
        result.append({
            "name": entry["name"],
            "G3Dmark": entry["G3Dmark"],
            "TDP": entry["TDP"],
            "price": entry["price"],
            "category": entry["category"],
            "distance": float(dist),
        })
    return result


def find_gpu_by_name(name):
    # Match a user-provided GPU name against the catalog, preferring exact matches.
    if name is None:
        return None
    normalized = name.strip().casefold()
    best_match = None
    for index, entry in enumerate(gpu_info):
        if entry["name"].casefold() == normalized:
            best_match = {**entry, "index": index}
            break
    if best_match:
        return best_match
    for index, entry in enumerate(gpu_info):
        if normalized in entry["name"].casefold():
            best_match = {**entry, "index": index}
            break
    return best_match


def recommend_gpus(gpu_name, k=5):
    # Find a matching GPU by name and return the nearest neighbors as recommendations.
    selected = find_gpu_by_name(gpu_name)
    if selected is None:
        return None

    feature_values = [selected["G3Dmark"], selected["TDP"], selected["price"]]
    n_neighbors = min(len(gpu_info), k + 1)
    distances, indices = gpu_model.find_neighbors(feature_values, k=n_neighbors)
    recommendations = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == selected["index"]:
            continue
        entry = gpu_info[idx]
        recommendations.append({
            "name": entry["name"],
            "G3Dmark": entry["G3Dmark"],
            "TDP": entry["TDP"],
            "price": entry["price"],
            "category": entry["category"],
            "distance": float(dist),
        })
        if len(recommendations) >= k:
            break
    return {"selected_gpu": selected, "recommended_gpus": recommendations}


# Train the models once at startup so the API endpoints can use them immediately.
initialize_model()


# Generate visualization base64 strings once the datasets are loaded.
model_metrics = compute_model_metrics()
cpu_pie_chart = None
cpu_histogram = None
cpu_scatter = None
cpu_bar_chart = None
cpu_boxplot = None
gpu_pie_chart = None
gpu_histogram = None
gpu_scatter = None
gpu_bar_chart = None
gpu_boxplot = None

if len(cpu_info) > 0:
    # Build the CPU charts so the dashboard can render them immediately.
    cpu_pie_chart = create_cpu_category_pie(cpu_info)
    cpu_histogram = create_cpu_benchmark_histogram(cpu_info)
    cpu_scatter = create_cores_vs_tdp_scatter(cpu_info)
    cpu_bar_chart = create_cpu_category_bar(cpu_info)
    cpu_boxplot = create_cpu_boxplot(cpu_info)

if len(gpu_info) > 0:
    # Build the GPU charts so the dashboard can render them immediately.
    gpu_pie_chart = create_gpu_category_pie(gpu_info)
    gpu_histogram = create_gpu_benchmark_histogram(gpu_info)
    gpu_scatter = create_tdp_vs_price_scatter(gpu_info)
    gpu_bar_chart = create_gpu_category_bar(gpu_info)
    gpu_boxplot = create_gpu_boxplot(gpu_info)




# Serve the main HTML page and pass the prepared chart data into the template.
@app.route("/", methods=["GET"])
def index():
    # Render the dashboard shell with the precomputed chart images and lookup data.
    return render_template(
        "index.html",
        result=None,
        message=None,
        cpus=cpu_info,
        gpu_names=[entry["name"] for entry in gpu_info],
        cpu_pie_chart=cpu_pie_chart,
        cpu_histogram=cpu_histogram,
        cpu_bar_chart=cpu_bar_chart,
        cpu_boxplot=cpu_boxplot,
        gpu_pie_chart=gpu_pie_chart,
        gpu_histogram=gpu_histogram,
        gpu_bar_chart=gpu_bar_chart,
        gpu_boxplot=gpu_boxplot,
        model_metrics=model_metrics,
    )


# API endpoint for CPU prediction using the trained regression model.
@app.route("/api/predict", methods=["GET"])
def api_predict():
    if not trained:
        return jsonify({"error": "Model is not trained yet. Please restart the app to retrain."}), 500

    feature_values = []
    for feature in FEATURES:
        param = request.args.get(feature.lower())
        if param is None:
            return jsonify({"error": f"Missing query parameter '{feature.lower()}'. Use the browser form or /api/predict?cores=...&tdp=..."}), 400
        parsed = parse_float(param)
        if parsed is None:
            return jsonify({"error": f"Query parameter '{feature.lower()}' must be a number."}), 400
        feature_values.append(parsed)

    prediction, class_pred = model.predict_with_category(feature_values, CATEGORY_THRESHOLDS)
    # Build the k-NN feature vector using the predicted score as part of the similarity input.
    knn_features = [float(prediction), feature_values[0], feature_values[1]]
    similar_cpus = get_similar_cpus(knn_features, k=5)
    result = {
        "cores": feature_values[0],
        "tdp": feature_values[1],
        "predicted_cpuMark": float(prediction),
        "predicted_category": CLASS_LABELS[class_pred],
        "similar_cpus": similar_cpus,
    }
    return jsonify(result)


# API endpoint for CPU recommendation based on a known processor name.
@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    if not trained:
        return jsonify({"error": "Model is not trained yet. Please restart the app to retrain."}), 500

    cpu_name = request.args.get("cpu_name")
    if not cpu_name:
        return jsonify({"error": "Query parameter 'cpu_name' is required."}), 400

    # Look up a matching CPU and return its nearest neighbors as recommendations.
    recommendation = recommend_cpus(cpu_name, k=5)
    if recommendation is None:
        return jsonify({"error": f"CPU '{cpu_name}' not found. Please choose a valid CPU from the list."}), 404

    return jsonify(recommendation)


# API endpoint for GPU prediction using the trained regression model.
@app.route("/api/gpu/predict", methods=["GET"])
def api_gpu_predict():
    if not gpu_trained:
        return jsonify({"error": "GPU model is not trained yet. Please restart the app to retrain."}), 500

    tdp = request.args.get("tdp")
    price = request.args.get("price")
    if tdp is None or price is None:
        return jsonify({"error": "Query parameters 'tdp' and 'price' are required."}), 400

    # Parse and validate the requested GPU inputs before running inference.
    parsed_tdp = parse_float(tdp)
    parsed_price = parse_float(price)
    if parsed_tdp is None or parsed_price is None:
        return jsonify({"error": "Query parameters 'tdp' and 'price' must be numbers."}), 400

    # Predict the GPU benchmark score and classify it into a market tier.
    prediction, class_pred = gpu_model.predict_with_category([parsed_tdp, parsed_price], GPU_CATEGORY_THRESHOLDS)
    knn_features = [float(prediction), parsed_tdp, parsed_price]
    similar_gpus = get_similar_gpus(knn_features, k=5)
    result = {
        "tdp": parsed_tdp,
        "price": parsed_price,
        "predicted_g3dmark": float(prediction),
        "predicted_category": GPU_CLASS_LABELS[class_pred],
        "similar_gpus": similar_gpus,
    }
    return jsonify(result)


# API endpoint for GPU recommendation based on a known graphics card name.
@app.route("/api/gpu/recommend", methods=["GET"])
def api_gpu_recommend():
    if not gpu_trained:
        return jsonify({"error": "GPU model is not trained yet. Please restart the app to retrain."}), 500

    gpu_name = request.args.get("gpu_name")
    if not gpu_name:
        return jsonify({"error": "Query parameter 'gpu_name' is required."}), 400

    # Look up a matching GPU and return the nearest neighbors as recommendations.
    recommendation = recommend_gpus(gpu_name, k=5)
    if recommendation is None:
        return jsonify({"error": f"GPU '{gpu_name}' not found. Please choose a valid GPU from the list."}), 404

    return jsonify(recommendation)


# HTML route that renders the prediction page and the same prediction logic for browser requests.
@app.route("/predict", methods=["GET"])
def predict():
    if not trained:
      return render_template(
          "index.html",
          result=None,
          message="Model is not trained yet. Please restart the app to retrain.",
          cpus=cpu_info,
          gpu_names=[entry["name"] for entry in gpu_info],
          cpu_pie_chart=cpu_pie_chart,
          cpu_histogram=cpu_histogram,
          cpu_bar_chart=cpu_bar_chart,
          cpu_boxplot=cpu_boxplot,
          gpu_pie_chart=gpu_pie_chart,
          gpu_histogram=gpu_histogram,
          gpu_bar_chart=gpu_bar_chart,
          gpu_boxplot=gpu_boxplot,
      ), 500

    feature_values = []
    for feature in FEATURES:
        param = request.args.get(feature.lower())
        if param is None:
            return render_template(
                "index.html",
                result=None,
                message=f"Missing query parameter '{feature.lower()}'. Use the browser form or /predict?cores=...&tdp=...",
                cpus=cpu_info,
                gpu_names=[entry["name"] for entry in gpu_info],
                cpu_pie_chart=cpu_pie_chart,
                cpu_histogram=cpu_histogram,
                cpu_bar_chart=cpu_bar_chart,
                cpu_boxplot=cpu_boxplot,
                gpu_pie_chart=gpu_pie_chart,
                gpu_histogram=gpu_histogram,
                gpu_bar_chart=gpu_bar_chart,
                gpu_boxplot=gpu_boxplot,
            ), 400
        parsed = parse_float(param)
        if parsed is None:
            return render_template(
                "index.html",
                result=None,
                message=f"Query parameter '{feature.lower()}' must be a number.",
                cpus=cpu_info,
                gpu_names=[entry["name"] for entry in gpu_info],
                cpu_pie_chart=cpu_pie_chart,
                cpu_histogram=cpu_histogram,
                cpu_bar_chart=cpu_bar_chart,
                cpu_boxplot=cpu_boxplot,
                gpu_pie_chart=gpu_pie_chart,
                gpu_histogram=gpu_histogram,
                gpu_bar_chart=gpu_bar_chart,
                gpu_boxplot=gpu_boxplot,
            ), 400
        feature_values.append(parsed)

    prediction, class_pred = model.predict_with_category(feature_values, CATEGORY_THRESHOLDS)
    knn_features = [float(prediction), feature_values[0], feature_values[1]]
    similar_cpus = get_similar_cpus(knn_features, k=5)
    result = {
        "cores": feature_values[0],
        "tdp": feature_values[1],
        "predicted_cpuMark": prediction.item(),
        "predicted_category": CLASS_LABELS[class_pred],
        "similar_cpus": similar_cpus,
    }
    return render_template(
        "index.html",
        result=result,
        message="Prediction completed successfully.",
        cpus=cpu_info,
        gpu_names=[entry["name"] for entry in gpu_info],
        cpu_pie_chart=cpu_pie_chart,
        cpu_histogram=cpu_histogram,
        cpu_bar_chart=cpu_bar_chart,
        cpu_boxplot=cpu_boxplot,
        gpu_pie_chart=gpu_pie_chart,
        gpu_histogram=gpu_histogram,
        gpu_bar_chart=gpu_bar_chart,
        gpu_boxplot=gpu_boxplot,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
