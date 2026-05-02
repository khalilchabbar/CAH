"""
CAH – Critère du Maximum (lien complet)
Affiche et enregistre le tableau de distances à chaque itération.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os, itertools

# ══════════════════════════════════════════════════════════════
# 0. CONFIG
# ══════════════════════════════════════════════════════════════
OUT_DIR = "./"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. DONNÉES BRUTES
# ══════════════════════════════════════════════════════════════
individus = ['I1','I2','I3','I4','I5','I6','I7','I8','I9',
             'I10','I11','I12','I13','I14','I15','I16','I17','I18']

data_raw = np.array([
    [170,-4.0,77,15.0,63.7,17],
    [181,-5.0,49,15.0,45.1,11],
    [192,-5.1,50,16.1,46.2,15],
    [173,-4.1,70,15.5,63.5,17],
    [170,-4.0,70,12.5,64.3,19],
    [175,-4.3,72,12.4,61.6,18],
    [170,-4.4,70,12.0,65.6,10],
    [168,-4.0,76,11.0,64.0, 7],
    [166,-4.0,76,10.0,64.0, 8],
    [181,-5.3,48,15.2,50.2,10],
    [186,-4.7,55,15.5,51.0,14],
    [180,-4.6,50,12.0,51.7,16],
    [185,-4.8,50,12.8,49.7,19],
    [192,-5.0,48,11.5,45.6,17],
    [191,-4.9,45,11.3,45.9,16],
    [192,-4.9,43,10.5,48.9,18],
    [192,-5.1,50,10.5,45.0,16],
    [195,-5.3,50,15.1,47.1,19],
], dtype=float)

# ══════════════════════════════════════════════════════════════
# 2. CENTRAGE-RÉDUCTION
# ══════════════════════════════════════════════════════════════
mu    = data_raw.mean(axis=0)
sigma = data_raw.std(axis=0, ddof=0)
data_cr = (data_raw - mu) / sigma

print("=" * 70)
print("CENTRAGE-RÉDUCTION")
print("=" * 70)
vars_ = ['TAI','VIT','DET','PAS','LEG','STA']
print(f"{'':5s} | " + " | ".join(f"{v:>7s}" for v in vars_))
print("-" * 55)
for i, ind in enumerate(individus):
    row = "  ".join(f"{data_cr[i,j]:+7.3f}" for j in range(6))
    print(f"{ind:5s} | {row}")

# ══════════════════════════════════════════════════════════════
# 3. MATRICE DE DISTANCES EUCLIDIENNES INITIALE
# ══════════════════════════════════════════════════════════════
N = len(individus)

def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

# On va travailler avec un dict  { frozenset({a,b}): distance }
# Les groupes sont des frozensets d'indices originaux
groups = [frozenset([i]) for i in range(N)]   # chaque individu = un groupe

def group_label(g):
    """Nom lisible d'un groupe."""
    names = sorted([individus[i] for i in g])
    if len(names) == 1:
        return names[0]
    return "{" + ",".join(names) + "}"

# Distance entre deux groupes (max des distances inter-individus)
# On stocke d'abord les distances inter-individus (fixes)
indiv_dist = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        indiv_dist[i, j] = euclidean(data_cr[i], data_cr[j])

def group_dist_max(g1, g2):
    return max(indiv_dist[i, j] for i in g1 for j in g2)

# Matrice de distances entre groupes (dict)
def build_dist_matrix(groups):
    mat = {}
    for a, b in itertools.combinations(range(len(groups)), 2):
        d = group_dist_max(groups[a], groups[b])
        mat[(a, b)] = d
    return mat

# ══════════════════════════════════════════════════════════════
# 4. UTILITAIRE : AFFICHAGE + IMAGE DU TABLEAU
# ══════════════════════════════════════════════════════════════
# Colormap chaud/froid pour les valeurs de distance
CMAP = LinearSegmentedColormap.from_list(
    "cah", ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#f5a623", "#fff"]
)

def save_table_image(groups, dist_mat, iteration, merged_pair, dist_min,
                     highlight_i=None, highlight_j=None):
    """
    Génère une image matplotlib du tableau de distances.
    - dist_mat : dict {(i,j): val}  (indices dans groups)
    - highlight_i, highlight_j : indices (dans groups) du minimum
    """
    n = len(groups)
    labels_g = [group_label(g) for g in groups]

    # Matrice 2D pour affichage
    M = np.full((n, n), np.nan)
    for (a, b), v in dist_mat.items():
        M[a, b] = v
        M[b, a] = v

    # ── figure ──
    cell_w = max(1.6, 9.0 / n)
    cell_h = max(0.55, 7.0 / n)
    fig_w = cell_w * (n + 1) + 1.5
    fig_h = cell_h * (n + 1) + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    ax.axis('off')

    # Titre
    if iteration == 0:
        title = "Matrice des distances initiale (données centrées-réduites)"
    else:
        g1, g2 = merged_pair
        title = (f"Itération {iteration} – Fusion : "
                 f"{group_label(g1)} ⊕ {group_label(g2)}   "
                 f"[distance = {dist_min:.3f}]")
    fig.suptitle(title, color='white', fontsize=max(30, 11 - n // 4),
                 fontweight='bold', y=0.98)

    # Colonnes et lignes
    col_labels = [' '] + labels_g
    fs = max(20, 9 - n // 4)

    # En-têtes colonnes
    for j, lbl in enumerate(col_labels):
        ax.text((j) / (n + 1), 1.0 - 0.5 / (n + 1),
                lbl, transform=ax.transAxes,
                ha='center', va='center',
                color='#aaaaff', fontsize=fs, fontweight='bold'
                )

    # Valeurs max global pour normalisation couleur
    all_vals = [v for v in dist_mat.values() if not np.isnan(v)]
    vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0, 1)

    for i in range(n):
        # En-tête ligne
        ax.text(0.5 / (n + 1), 1.0 - (i + 1.5) / (n + 1),
                labels_g[i], transform=ax.transAxes,
                ha='center', va='center',
                color='#aaaaff', fontsize=fs, fontweight='bold')
        for j in range(n):
            x = (j + 1.5) / (n + 1)
            y = 1.0 - (i + 1.5) / (n + 1)

            if i == j:
                # Diagonale
                color_bg = '#1a1a2e'
                txt = '—'
                txt_color = '#444466'
            elif np.isnan(M[i, j]):
                color_bg = '#1a1a2e'
                txt = ''
                txt_color = 'white'
            else:
                norm = (M[i, j] - vmin) / (vmax - vmin + 1e-9)
                rgba = CMAP(norm)
                color_bg = rgba
                txt = f"{M[i, j]:.3f}"
                brightness = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                txt_color = 'black' if brightness > 0.55 else 'white'

            # Highlight la cellule minimale
            edge_color = 'none'
            lw = 0
            if (highlight_i is not None and highlight_j is not None and
                    i != j and not np.isnan(M[i,j]) and
                    ((i == highlight_i and j == highlight_j) or
                     (i == highlight_j and j == highlight_i))):
                edge_color = '#00ff88'
                lw = 2.5
                color_bg = '#003322'
                txt_color = '#00ff88'

            # Rectangle de cellule
            rect = mpatches.FancyBboxPatch(
                (x - 0.25 / (n + 1), y - 0.2 / (n + 1)),
                0.5/ (n + 1), 0.4 / (n + 1),
                boxstyle="round,pad=0.01",
                linewidth=lw, edgecolor=edge_color,
                facecolor=color_bg,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)
            ax.text(x, y, txt, transform=ax.transAxes,
                    ha='center', va='center',
                    color=txt_color, fontsize=max(10, fs - 1))

    # Note en bas

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(OUT_DIR, f"iter_{iteration:02d}.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Image enregistrée : {fname}")


def print_console_table(groups, dist_mat):
    """Affichage console du tableau de distances."""
    n = len(groups)
    labels_g = [group_label(g) for g in groups]
    col_w = max(8, max(len(l) for l in labels_g) + 2)
    header = f"{'':>{col_w}s} | " + " | ".join(f"{l:>{col_w}s}" for l in labels_g)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{labels_g[i]:>{col_w}s} | "
        for j in range(n):
            if i == j:
                row += f"{'—':>{col_w}s} | "
            else:
                key = (min(i,j), max(i,j))
                v = dist_mat.get(key, float('nan'))
                row += f"{v:>{col_w}.3f} | "
        print(row)


# ══════════════════════════════════════════════════════════════
# 5. CAH ITÉRATIVE
# ══════════════════════════════════════════════════════════════
# Matrice initiale
dist_mat = build_dist_matrix(groups)

print("\n" + "=" * 70)
print("MATRICE DE DISTANCES INITIALE (données centrées-réduites)")
print("=" * 70)
print_console_table(groups, dist_mat)
save_table_image(groups, dist_mat, iteration=0,
                 merged_pair=None, dist_min=None)

# Historique des fusions pour le dendrogramme
history = []   # [(label_g1, label_g2, dist)]

iteration = 1
while len(groups) > 1:
    # ── Trouver le minimum ──────────────────────────────────────
    min_key = min(dist_mat, key=dist_mat.get)
    min_val = dist_mat[min_key]
    idx_a, idx_b = min_key   # indices dans `groups`

    g_a = groups[idx_a]
    g_b = groups[idx_b]
    label_a = group_label(g_a)
    label_b = group_label(g_b)

    print("\n" + "═" * 70)
    print(f"ITÉRATION {iteration}")
    print(f"  Distance minimale = {min_val:.3f}")
    print(f"  Fusion : {label_a}  ⊕  {label_b}")
    print("═" * 70)

    # ── Sauvegarder l'image AVANT fusion (avec highlight) ───────
    save_table_image(groups, dist_mat, iteration,
                     merged_pair=(g_a, g_b), dist_min=min_val,
                     highlight_i=idx_a, highlight_j=idx_b)

    history.append((label_a, label_b, min_val))

    # ── Créer le nouveau groupe fusionné ────────────────────────
    new_group = g_a | g_b
    new_groups = [g for k, g in enumerate(groups) if k not in (idx_a, idx_b)]
    new_groups.append(new_group)

    # ── Mise à jour de la matrice (critère MAX) ─────────────────
    new_dist_mat = {}
    m = len(new_groups)
    for a, b in itertools.combinations(range(m), 2):
        new_dist_mat[(a, b)] = group_dist_max(new_groups[a], new_groups[b])

    groups   = new_groups
    dist_mat = new_dist_mat

    # ── Affichage console du nouveau tableau ─────────────────────
    if len(groups) > 1:
        print(f"\n  Tableau après fusion (critère MAX) :")
        print_console_table(groups, dist_mat)

    iteration += 1

# ══════════════════════════════════════════════════════════════
# 6. RÉSUMÉ DES FUSIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RÉSUMÉ DES FUSIONS")
print("=" * 70)
print(f"{'Itér':>5s} | {'Groupe 1':<35s} | {'Groupe 2':<35s} | {'Distance':>9s}")
print("-" * 90)
for k, (l1, l2, d) in enumerate(history, 1):
    print(f"{k:>5d} | {l1:<35s} | {l2:<35s} | {d:>9.3f}")

# ══════════════════════════════════════════════════════════════
# 7. DENDROGRAMME FINAL
# ══════════════════════════════════════════════════════════════
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

dist_condensed = pdist(data_cr, metric='euclidean')
Z = linkage(dist_condensed, method='complete')

fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

CUT = 3.5
dendrogram(Z, labels=individus, ax=ax,
           color_threshold=CUT,
           above_threshold_color='#556677',
           leaf_font_size=11, leaf_rotation=0)

ax.axhline(y=CUT, color='#ffd166', linewidth=1.8,
           linestyle='--', alpha=0.9, label=f'Seuil de coupe ≈ {CUT}')

ax.set_title('Dendrogramme final – CAH critère du maximum',
             color='white', fontsize=14, fontweight='bold', pad=14)
ax.set_xlabel('Individus', color='#aaaacc', fontsize=11)
ax.set_ylabel('Distance euclidienne', color='#aaaacc', fontsize=11)
ax.tick_params(colors='#ccccdd')
for sp in ax.spines.values():
    sp.set_edgecolor('#333355')
ax.legend(facecolor='#1a1a2e', edgecolor='#333355',
          labelcolor='white', fontsize=9)

plt.tight_layout()
dend_path = os.path.join(OUT_DIR, "dendrogramme_final.png")
plt.savefig(dend_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print(f"\n✅ Tous les fichiers sont dans : {OUT_DIR}/")
print(f"   • iter_00.png  : matrice initiale")
print(f"   • iter_01.png … iter_{iteration-1:02d}.png : une image par itération")
print(f"   • dendrogramme_final.png")
