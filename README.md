# Modularity on the Karate Club Graph
### DSC212: Graph Theory Module — Research Assignment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 👤 Student & Submission Details

| **Field** | **Information** |
|:---|:---|
| **Student Name** | **Rudrapratap Hemant Baandal** |
| **Roll Number** | **IMS25178** |
| **Course** | DSC212: Graph Theory Module |
| **Assignment Topic** | Modularity on the Karate Club Graph (Spectral Community Detection) |
| **Submission Mode** | Jupyter Notebook (`.ipynb`) with top-to-bottom reproducible outputs & GitHub Repository |
| **Benchmark Dataset** | Zachary's Karate Club ($n = 34$ nodes, $m = 78$ edges) |

---

### ✍️ Authorship & Academic Honesty Declaration
> *"I can proudly say that most of this assignment was written by me personally. I have however, used AI, and in doing so I have tried to use it in places that will not hinder my learning. I have used it to debug visual errors in matplotlib plots and also for general info on some concepts."*  
> — **Rudrapratap Hemant Baandal (IMS25178)**

---

## 📌 Project Overview & Brief

This research project presents an end-to-end investigation, from-scratch algorithmic implementation, and empirical analysis of **modularity-based community detection** using **recursive spectral bisection** on the classic **Zachary Karate Club network** ($n = 34$, $m = 78$).

### Key Objectives & Deliverables
1. **Mathematical Derivation & Proofs:** Full mathematical treatment of the modularity matrix $B = A - \frac{\mathbf{k}\mathbf{k}^T}{2m}$, including formal proofs for:
   - Reality and symmetry of $B$ ($B = B^T$).
   - Spectral Theorem properties (real eigenvalues and orthogonal eigenbasis).
   - Zero eigenvalue property ($B\mathbf{1} = \mathbf{0}$) and zero modularity for the trivial partition ($Q = 0$).
   - Rayleigh-Ritz theorem justifying continuous relaxation and leading eigenvector maximization.
   - Theorem 3: Subgraph restriction $B^{(C)} = B[C, C]$ and the strict eigenvalue stopping criterion $\lambda_1^{(C)} \le 0 \implies \Delta Q \le 0$.
2. **From-Scratch Implementation:** Implementation of unweighted modularity and recursive spectral bisection using only standard scientific Python libraries (`numpy`, `scipy`, `networkx`).
3. **Step-by-Step Graph Visualizations:** Progressive visualization of the karate club network at each split iteration on a **fixed spring layout** (nodes remain at fixed coordinates), with distinct community colors and labeled node IDs.
4. **Dynamic Topological Metric Tracking:** NetworkX computation and tracking of four fundamental metrics after each split across the entire 34-node population:
   - **Degree Centrality ($C_D$)**
   - **Betweenness Centrality ($C_B$)**
   - **Closeness Centrality ($C_C$)**
   - **Clustering Coefficient ($C$)**
5. **In-Depth Analytical Discussion:** Detailed answers to core assignment questions:
   - Identification of persistently central nodes across splits (Node 0 - Mr. Hi, Node 33 - Officer, and their lieutenants Node 2 & Node 32).
   - Analysis of structural changes driving metric shifts (betweenness collapse for boundary spanners, clustering peaks in local cliques).
   - Evaluation against Zachary's empirical anthropological ground truth (**97.1% accuracy**).

---

## 🔄 Algorithmic Pipeline

```
                                  [ Zachary Karate Club Network ]
                                      (n = 34 nodes, m = 78 edges)
                                                    │
                                                    ▼
                             [ Unweighted 0-1 Adjacency Matrix A ]
                                                    │
                                                    ▼
                          [ Global Modularity Matrix B = A - k k^T / 2m ]
                                 (Verified: B = B^T, B * 1 = 0)
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
      [ Theoretical Proofs ]                                      [ Recursive Spectral Bisection ]
    • Reality & Symmetry                                           Candidate Communities Queue
    • Orthonormal Basis                                                              │
    • Rayleigh-Ritz Relaxation                                                       ▼
    • Eigenvalue Stopping Criterion                             Extract Submatrix B^(C) = B[C, C]
                                                                                     │
                                                                                     ▼
                                                                     Compute Leading Eigenpair (λ1, u1)
                                                                                     │
                                                                 ┌───────────────────┴───────────────────┐
                                                                 ▼                                       ▼
                                                          λ1 > 0 & ΔQ > 0                         λ1 <= 0 or trivial
                                                       (Bipartition via sign(u1))                 (Group is Indivisible)
                                                                 │                                       │
                                                                 ▼                                       ▼
                                                      [ Accept Split & Recurse ]                    [ Stop Split ]
                                                                 │                                       │
                   ┌─────────────────────────────────────────────┴───────────────────────────────────────┘
                   ▼
         [ Iteration Snapshots ]
    (Iteration 0 -> 1 -> 2 -> 3 -> 4)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[ Graph Visualizations ]  [ Centrality & Metric Analysis ]
• Fixed Spring Layout     • Degree Centrality (C_D)
• Distinct Color Coding   • Betweenness Centrality (C_B)
• Ground Truth Validation • Closeness Centrality (C_C)
                          • Clustering Coefficient (C)
                          • Metric Evolution Trajectories
                          • Comprehensive 34-Node Heatmaps
```

---

## 🥋 1. Background: The Karate Club Story
In the early 1970s, anthropologist **Wayne W. Zachary** observed the social interactions among 34 members of a university karate club over a two-year period (1970–1972). Nodes represent members, and edges represent friendship and shared activities outside formal karate classes.

During the study, an escalating conflict arose between the martial arts instructor (**Mr. Hi**, node 0) and the club administrator (**John A / Officer**, node 33) over whether club revenues should fund instructor salaries or new equipment. The dispute eventually split the club into two rival factions:
- **Mr. Hi's Faction:** Followers of the instructor who departed to found a new studio.
- **The President's Faction:** Members who remained with the administrator at the university.

Because Zachary collected the interaction network *before* the schism took place, the Karate Club network ($n = 34$, $m = 78$) is the canonical benchmark for community detection: **Can an algorithm operating purely on graph topology uncover the latent social fault lines that fractured the club?**

---

## 📐 2. Mathematical Formulation & Supporting Proofs

### 2.1 The Modularity Quality Function
While naive **minimum cut** partitioning frequently separates isolated leaf nodes (trivial solutions), **modularity ($Q$)** measures the density of intra-community links compared to a degree-preserving null model:

$$B = A - \frac{\mathbf{k}\mathbf{k}^T}{2m}, \quad B_{ij} = A_{ij} - \frac{k_i k_j}{2m}$$

where:
- $A \in \{0, 1\}^{n \times n}$ is the unweighted adjacency matrix ($m = 78$).
- $\mathbf{k} = (k_1, \dots, k_n)^T$ is the node degree vector ($\sum_i k_i = 2m = 156$).
- For a bipartition $\mathbf{s} \in \{-1, +1\}^n$, the modularity score is:
  $$Q = \frac{1}{4m} \mathbf{s}^T B \mathbf{s}$$

---

### 2.2 Formal Proofs

#### Proposition 1. Reality and Symmetry of $B$
- **Reality:** $A \in \mathbb{R}^{n \times n}$, $\mathbf{k}\mathbf{k}^T \in \mathbb{R}^{n \times n}$, and $2m > 0 \implies B \in \mathbb{R}^{n \times n}$.
- **Symmetry:** Because $G$ is undirected, $A^T = A$. Furthermore, $(\mathbf{k}\mathbf{k}^T)^T = \mathbf{k}\mathbf{k}^T$ (rank-1 outer product is symmetric). Hence:
  $$B^T = A^T - \frac{(\mathbf{k}\mathbf{k}^T)^T}{2m} = A - \frac{\mathbf{k}\mathbf{k}^T}{2m} = B \quad \blacksquare$$

#### Theorem 1. Real Eigenvalues & Orthonormal Eigenbasis
- For any complex eigenpair $(\lambda, \mathbf{v})$ with $\mathbf{v} \ne \mathbf{0}$, $\mathbf{v}^\ast B \mathbf{v} = \lambda \|\mathbf{v}\|_2^2$.
- Since $B = B^T = B^\ast\ast$, $(\mathbf{v}^\ast B \mathbf{v})^* = \mathbf{v}^\ast B \mathbf{v} \in \mathbb{R}$. Because $\|\mathbf{v}\|_2^2 > 0$ is real, $\lambda \in \mathbb{R}$.
- By the Spectral Theorem, distinct eigenspaces are orthogonal, providing an orthonormal basis for $\mathbb{R}^n$. $\blacksquare$

#### Proposition 2. $B\mathbf{1} = \mathbf{0}$ (Zero Eigenvalue)
$$B\mathbf{1} = A\mathbf{1} - \frac{\mathbf{k}(\mathbf{k}^T\mathbf{1})}{2m} = \mathbf{k} - \frac{\mathbf{k}(2m)}{2m} = \mathbf{k} - \mathbf{k} = \mathbf{0}$$
Thus, the all-ones vector $\mathbf{1}$ is an eigenvector of $B$ associated with eigenvalue $\lambda = 0$. $\blacksquare$

#### Corollary 1. Modularity of Trivial Partition is Zero
$$Q = \frac{1}{4m}\mathbf{1}^T B \mathbf{1} = \frac{1}{4m}\mathbf{1}^T \mathbf{0} = 0 \quad \blacksquare$$

#### Theorem 2. Rayleigh-Ritz Characterization
Relaxing $s_i \in \{-1, +1\}$ to $\|\mathbf{s}\|_2 = 1$, the continuous quadratic form $\max_{\|\mathbf{s}\|_2=1} \mathbf{s}^T B \mathbf{s} = \lambda_1$, attained at the leading eigenvector $\mathbf{u}_1$. Rounding by sign produces the discrete bipartition:
$$


s_i = \begin{cases} 
+1 & \text{if } (\mathbf{u}_1)_i > 0 \\
-1 & \text{otherwise} 
\end{cases} \quad \blacksquare


$$

#### Theorem 3. Subgraph Restriction & Eigenvalue Stopping Criterion
For a candidate community $C \subseteq V$, the restricted modularity submatrix is $B^{(C)} = B[C, C]$. The modularity gain from splitting $C$ via $\mathbf{s} \in \{-1, +1\}^{|C|}$ is:
$$\Delta Q = \frac{1}{4m} \mathbf{s}^T B^{(C)} \mathbf{s} \le \frac{1}{4m} \lambda_1^{(C)} \|\mathbf{s}\|_2^2 = \frac{1}{4m} \lambda_1^{(C)} |C|$$
**Stopping Criterion:** If $\lambda_1^{(C)} \le 0$, then $\Delta Q \le 0$ for all possible splits $\mathbf{s}$. No bipartition can increase modularity; thus, recursion terminates on $C$. $\blacksquare$

---

## 🔬 3. Algorithmic Workflow & Results

### 3.1 Recursive Bisection Trace
The algorithm recursively bisects communities until all candidate subgraphs satisfy $\lambda_1^{(C)} \le 0$:

| Iteration | Action / Split Description | Subset Sizes | $\lambda_1^{(C)}$ | $\Delta Q$ | Total Modularity $Q$ |
|:---:|:---|:---:|:---:|:---:|:---:|
| **0** | Baseline: Undivided Network ($V$) | 34 | — | — | **0.0000** |
| **1** | Primary Split ($V \to C_1, C_2$) | 16 & 18 | 4.9771 | +0.3715 | **0.3715** |
| **2** | Split $C_1$ (Mr. Hi's group) | 10 & 6 | 2.9441 | +0.1112 | **0.3980** |
| **3** | Split $C_2$ (President's group) | 13 & 5 | 2.5041 | +0.1112 | **0.4071** |
| **4** | Split sub-community of size 6 | 5 & 1 | 2.1480 | +0.0338 | **0.4094** |
| **End** | **Convergence reached** | 5 Comms | $\le 0$ / trivial | — | **0.4094** |

---

## 📊 4. Visualizations

### 4.1 Step-by-Step Community Evolution (Fixed Spring Layout)
Nodes remain at fixed coordinates across all iterations, displaying progressive community fission:

![Community Evolution Across Splits](assets/community_evolution_snapshots.png)

### 4.2 Alignment with Zachary's Ground Truth Factions
The initial spectral bipartition matches the actual historical split with **97.1% accuracy** (33 of 34 nodes correctly classified). The sole ambiguous node is **Node 8**, which Zachary documented as an undecided student maintaining close ties with both rival leaders:

![Ground Truth Comparison](assets/ground_truth_comparison.png)

---

## 📈 5. Network Metrics & Centrality Evolution

### 5.1 Trajectory of Centrality Measures Across Splits
After each split, we evaluate the 4 required NetworkX metrics:
1. **Degree Centrality ($C_D$):** Direct local connectivity fraction.
2. **Betweenness Centrality ($C_B$):** Shortest-path broker score.
3. **Closeness Centrality ($C_C$):** Inverse shortest-path distance.
4. **Clustering Coefficient ($C$):** Local triadic closure density.

![Centrality Evolution Lines](assets/centrality_evolution_lines.png)

### 5.2 Global Metric Heatmaps (All 34 Nodes)
![Centrality Heatmaps](assets/centrality_heatmaps.png)

---

## 💬 6. Key Insights & Discussion (Answers to Task 5)

### 6.1 Which Nodes Consistently Remain Central Across Splits?
- **Node 0 (Mr. Hi - Instructor)** and **Node 33 (John A / Officer - President)** remain the undisputed hubs throughout all iterations:
  - In Iteration 0, they possess the highest initial degrees ($k_0 = 16$, $k_{33} = 17$).
  - Even after inter-faction edges dissolve, Node 0 preserves an internal degree of 10 and Node 33 preserves an internal degree of 14 within their respective primary factions. Throughout all iterations, they remain the dominant hubs of their respective communities.
- **Persistent Factional Lieutenants (Node 2 and Node 32):**
  - Node 2 is the primary lieutenant to Mr. Hi, maintaining the second-highest internal degree and shortest path density within the instructor's faction.
  - Node 32 serves as the primary lieutenant to the Club President, consistently anchoring the secondary hub of the administrator's faction.

### 6.2 How Community Structure Influences Network Metrics
1. **Collapse of Global Betweenness for Boundary Spanners:**
   - In the undivided graph, Node 0 and Node 33 control over 74% of all shortest paths in the club ($C_B(0) = 0.4376$, $C_B(33) = 0.3041$).
   - Upon the primary split (Iteration 1), inter-faction edges are cut. Global betweenness collapses by **68.6%** for Node 0 and **51.7%** for Node 33. Their high betweenness was an artifact of bridging two distinct social clusters. Within their own isolated communities, however, they remain the dominant internal brokers.
2. **Closeness Centrality Reflects Structural Isolation:**
   - Closeness contracts systematically as cross-community paths vanish. Reachable components become smaller, illustrating the physical fragmentation of the club.
3. **Clustering Coefficient Highlights Dense Cliques vs. Structural Holes:**
   - Club leaders exhibit low clustering coefficients ($C(0) = 0.150$, $C(33) = 0.110$) because they span structural holes across separate sub-groups.
   - Peripheral and tightly knit members (e.g., Nodes 14, 15, 18, 20, 22) exhibit clustering coefficients near **1.0**, forming dense local training cliques.

---

## 💻 7. Verification & Reproducibility

### Environment & Dependencies
- Python 3.10+
- `numpy>=1.24`
- `scipy>=1.10`
- `networkx>=3.0`
- `matplotlib>=3.7`
- `pandas>=2.0`
- `nbformat>=5.9`
- `nbclient>=0.8`

### Top-to-Bottom Execution Test
To verify top-to-bottom execution without manual intervention:
```bash
python -m nbconvert --to notebook --execute Karate_Club_Modularity.ipynb --output Karate_Club_Modularity.ipynb
```

---

## 📚 References
1. **Newman, M. E. J. (2006).** "Modularity and community structure in networks." *Proceedings of the National Academy of Sciences (PNAS)*, 103(23), 8577–8582.
2. **Zachary, W. W. (1977).** "An Information Flow Model for Conflict and Fission in Small Groups." *Journal of Anthropological Research*, 33(4), 452–473.
3. **Newman, M. E. J. (2004).** "Fast algorithm for detecting community structure in networks." *Physical Review E*, 69(6), 066133.
