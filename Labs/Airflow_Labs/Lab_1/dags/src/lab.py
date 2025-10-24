import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import pickle
import os
import base64

def load_data():
    """
<<<<<<< HEAD
    Loads data from a CSV file and serializes it.

    Returns:
        bytes: Serialized dataframe.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    return serialized_data


def data_preprocessing(data):
    """
    Deserializes, cleans, and scales data for clustering.

    Args:
        data (bytes): Serialized dataframe.

    Returns:
        bytes: Serialized scaled data.
=======
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
>>>>>>> ec2eb14780820681766b73e6c4136b3f4fda1d89
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
<<<<<<< HEAD
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(clustering_data)
    serialized_scaled = pickle.dumps(scaled)
    return serialized_scaled
=======

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")
>>>>>>> ec2eb14780820681766b73e6c4136b3f4fda1d89


def build_save_model(data_b64: str, filename: str):
    """
<<<<<<< HEAD
    Builds a K-Medoids clustering model, saves it, and computes silhouette scores.

    Args:
        data (bytes): Serialized, preprocessed data.
        filename (str): File name to save the model.

    Returns:
        list: Silhouette scores for each cluster count.
    """
    df = pickle.loads(data)
    scores = []
    best_score = -1
    best_model = None

    for k in range(2, 10):
        model = KMedoids(n_clusters=k, random_state=42)
        labels = model.fit_predict(df)
        score = silhouette_score(df, labels)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_model = model

    # Save best model
=======
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # NOTE: This saves the last-fitted model (k=49), matching your original intent.
>>>>>>> ec2eb14780820681766b73e6c4136b3f4fda1d89
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
<<<<<<< HEAD
        pickle.dump(best_model, f)

    return scores


def load_model_elbow(filename, scores):
    """
    Loads saved K-Medoids model, reports best k, and predicts clusters.

    Args:
        filename (str): Model file name.
        scores (list): Silhouette scores list.

    Returns:
        str: Cluster label for first test sample.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))

    best_k = scores.index(max(scores)) + 2  # since we started from 2
    print(f"Optimal number of clusters (K-Medoids): {best_k}")
    print(f"Best silhouette score: {max(scores):.4f}")

    predictions = loaded_model.predict(df)
    return predictions[0]
=======
        pickle.dump(kmeans, f)

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved (last-fitted) model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # elbow for information/logging
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # predict on raw test data (matches your original code)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(df)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred
>>>>>>> ec2eb14780820681766b73e6c4136b3f4fda1d89
