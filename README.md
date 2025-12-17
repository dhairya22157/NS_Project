# Structural and Topological Analysis of Amazon Co-Purchase Network

This project explores the structure and topology of the Amazon product co-purchase network using Network Science techniques. The analysis involves computing various network metrics, identifying communities, and comparing the real-world network with random graph models.

## Project Overview

The goal of this project is to understand the connectivity patterns of products on Amazon. In a co-purchase network, a directed edge exists from product $i$ to product $j$ if product $j$ is frequently purchased with product $i$.

Key analyses performed in this project include:

1.  **Degree Distribution Analysis**: Computing in-degree and out-degree distributions and visualizing them on a log-log scale to check for power-law properties (scale-free nature).
2.  **Clustering Coefficient**: Calculating the average clustering coefficient to measure the tendency of nodes to cluster together (transitivity).
3.  **Assortativity**: Measuring the correlation between degrees of connected nodes to see if high-degree nodes connect to other high-degree nodes.
4.  **Connected Components**: Identifying and analyzing strongly and weakly connected components to understand network fragmentation/connectivity.
5.  **Random Network Comparison**: Comparing the properties of the Amazon network with an Erdős–Rényi random graph of the same size to highlight non-random structural features (like Small-World properties).
6.  **Community Detection**: Using the Louvain method to detect communities (clusters of products) and calculating the modularity score.

## Dataset

The analysis uses the following dataset:
*   `copurchase.csv`: Contains the edges of the co-purchase network (Source, Target).
*   `products.csv`: Contains metadata about the products (ID, title, group, salesrank, reviews, etc.).


## Key Findings

*   **Degree Distribution**: The network likely exhibits a heavy-tailed degree distribution, characteristic of scale-free networks where a few "hub" products have many connections.
*   **Clustering**: The network shows a high average clustering coefficient compared to a random graph, indicating a high level of local connectivity.
*   **Communities**: The network can be partitioned into distinct communities with a high modularity score, suggesting strong grouping of related products.

## Files

*   `NS_Project.ipynb`: The main Jupyter Notebook containing the code and analysis.
*   `copurchase.csv`: The network edge list data.
*   `products.csv`: The product metadata.
*   `Network_science_Project.pdf`: Project report.
*   `Exploration of Amazon’s Co-Purchase Network.pptx`: Project presentation.
