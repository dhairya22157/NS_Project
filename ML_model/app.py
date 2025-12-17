from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import networkx as nx
from joblib import load

app = Flask(__name__)

# Load once at startup
df_filter = pd.read_csv("filtered_products.csv")
import pickle

with open("copurchase_graph.gpickle", "rb") as f:
    G = pickle.load(f)

model = load("copurchase_prediction_model.joblib")

feature_names = [
    'common_neighbors', 'pref_attachment', 'jaccard', 
    'adamic_adar', 'resource_allocation', 
    'salesrank_diff', 'rating_product'
]

# Create dropdown options
book_options = df_filter[['id', 'title']].dropna().values.tolist()

def predict_pair(book1, book2):
    G_undir = G.to_undirected()
    neighbors = lambda x: set(G_undir.neighbors(x))
    u_neighbors = neighbors(book1)
    v_neighbors = neighbors(book2)
    common = u_neighbors & v_neighbors
    union = u_neighbors | v_neighbors

    features = {
        'common_neighbors': len(common),
        'pref_attachment': G.out_degree(book1) * G.in_degree(book2),
        'jaccard': len(common) / len(union) if union else 0,
        'adamic_adar': sum(1 / np.log(G.degree(w)) for w in common if G.degree(w) > 1),
        'resource_allocation': sum(1 / G.degree(w) for w in common if G.degree(w) > 0),
        'salesrank_diff': abs(G.nodes[book1].get('salesrank', 0) - G.nodes[book2].get('salesrank', 0)),
        'rating_product': G.nodes[book1].get('rating', 0) * G.nodes[book2].get('rating', 0)
    }
    X = np.array([[features[k] for k in feature_names]])
    return model.predict_proba(X)[0][1]

# Book recommendation function
import random
def recommend_future_books(G, model, df_filter, feature_names, book_id, top_k=5):
    G_undir = G.to_undirected()
    neighbors_dict = {node: set(G_undir.neighbors(node)) for node in G_undir.nodes()}
    degree_dict = dict(G.degree())
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())

    if book_id not in G:
        return []

    # Get neighbors of the selected book
    u_neighbors = neighbors_dict.get(book_id, set())

    # Candidates: All books except the selected book and its neighbors
    candidates = set(G.nodes()) - u_neighbors - {book_id}

    # Randomly sample a smaller subset of candidates if there are too many
    sampled_candidates = random.sample(list(candidates), min(50, len(candidates)))  # Convert set to list

    scores = []

    for v in sampled_candidates:
        common_neighbors = neighbors_dict[book_id].intersection(neighbors_dict[v])
        union = neighbors_dict[book_id].union(neighbors_dict[v])

        # Calculate the feature vector for the pair of books
        feature_vec = [
            len(common_neighbors),  # Common neighbors
            out_degree_dict.get(book_id, 0) * in_degree_dict.get(v, 0),  # Preferential Attachment
            len(common_neighbors) / len(union) if len(union) > 0 else 0,  # Jaccard coefficient
            sum(1 / np.log(degree_dict.get(w, 1)) for w in common_neighbors if degree_dict.get(w, 1) > 1),  # Adamic-Adar
            sum(1 / degree_dict.get(w, 1) for w in common_neighbors if degree_dict.get(w, 1) > 0),  # Resource Allocation
            abs(G.nodes[book_id].get('salesrank', 0) - G.nodes[v].get('salesrank', 0)),  # Sales rank difference
            G.nodes[book_id].get('rating', 0) * G.nodes[v].get('rating', 0)  # Rating product
        ]
        
        # Get the probability for co-purchase likelihood
        prob = model.predict_proba([feature_vec])[0][1]
        scores.append((v, prob))

    # Sort by the probability score (descending)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Get the top-k recommendations
    top = scores[:top_k]

    # Retrieve book titles and return recommendations
    recommendations = []
    for pid, prob in top:
        title = df_filter[df_filter['id'] == pid]['title'].values[0]
        recommendations.append((title, round(prob, 4)))

    return recommendations



@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    selected_titles = ["", ""]

    if request.method == "POST":
        id1 = int(request.form["book1"])
        id2 = int(request.form["book2"])
        selected_titles[0] = df_filter[df_filter["id"] == id1]["title"].values[0]
        selected_titles[1] = df_filter[df_filter["id"] == id2]["title"].values[0]
        
        if id1 not in G.nodes or id2 not in G.nodes:
            error = "There is no known copurchase relationship between these books."
        else:
            prediction = predict_pair(id1, id2)
    
    return render_template(
        "index.html",
        book_options=book_options,
        prediction=prediction,
        selected_titles=selected_titles,
        error=error
    )

@app.route("/reco", methods=["GET", "POST"])
def recommendations():
    recommendations = []
    error = None
    selected_title = ""

    if request.method == "POST":
        book_id = int(request.form["book1"])
        selected_title = df_filter[df_filter["id"] == book_id]["title"].values[0]
        
        if book_id not in G.nodes:
            error = "❌ No co-purchase relationship found for the selected book."
        else:
            recommendations = recommend_future_books(G, model, df_filter, feature_names, book_id)
    
    return render_template(
        "recommendations.html",
        book_options=book_options,
        recommendations=recommendations,
        selected_title=selected_title,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
