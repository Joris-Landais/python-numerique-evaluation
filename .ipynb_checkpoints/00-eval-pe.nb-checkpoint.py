# -*- coding: utf-8 -*-
# %% [markdown]
# # Évaluation de python-numérique
#
# ***Analyse morphologique de défauts 3D***
#
# ***

# %% [markdown]
# ## Votre identité

# %% [markdown]
# Ne touchez rien, remplacez simplement les ???
#
# Prénom: Joris
#
# Nom: Landais
#
# Langage-avancé (Python ou C++): Python
#
# Adresse mail: joris.landais@mines-paristech.fr
#
# ***

# %% [markdown]
# ## Quelques éléments de contexte et objectifs
#
# Vous allez travaillez dans ce projet sur des données réelles concernant des défauts observés dans des soudures. Ces défauts sont problématiques car ils peuvent occasionner la rupture prématurée d'une pièce, ce qui peut avoir de lourdes conséquences. De nombreux travaux actuels visent donc à caractériser la **nocivité** des défauts. La morphologie de ces défauts est un paramètre qui influe au premier ordre sur cette nocivité.
#
# Dans ce projet, vous aurez l'occasion de manipuler des données qui caractérisent la morphologie de défauts réels observés dans une soudure volontairement ratée ! Votre mission est de mener une analyse permettant de classer les défauts selon leur forme : en effet, deux défauts avec des morphologies similaires auront une nocivité comparable.

# %% [markdown]
# ### Import des librairies numériques
#
# Importez les librairies numériques utiles au projet, pour le début du projet, il s'agit de `pandas`, `numpy` et `pyplot` de `matplotlib`.

# %%
from utilities import plot_defect
from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Lecture des données
#
# Les données se trouvent dans le fichier `defect_data.csv`. Ce fichier contient treize colonnes de données :
# * la première colonne est un identifiant de défaut (attention, il y a 4040 défauts mais les ids varient entre 1 et 4183) ;
# * les neuf colonnes suivantes sont des descripteurs de forme sur lesquels l'étude va se concentrer ;
# * les trois dernières colonnes sont des indicateurs mécaniques auxquels nous n'allons pas nous intéresser dans ce projet.
#
# Lisez les données dans une data-frame en ne gardant que les 9 descripteurs de forme et en indexant vos lignes par les identifiants des défauts. Ces deux opérations doivent être faites au moment de la lecture des données.
#
# Affichez la forme et les deux dernières lignes de la data-frame.

# %%
df = pd.read_csv("defect_data.csv", usecols=range(10), index_col="id")
df.tail(2)

# %% [markdown]
# ### Parenthèse sur descripteurs morphologiques
#
# **Note: cette section vous donne du contexte quant à la signification des données que vous manipulez et la façon dont elles ont été acquises. Si certains aspects vous semblent nébuleux, cela ne vous empêchera pas de finir le projet !**
#
# Vous allez manipuler dans ce projet des descripteurs morphologiques. Ces descripteurs sont ici utilisés pour caractériser des défauts, observés par tomographie aux rayons X dans des soudures liant deux pièces métalliques. La tomographie consiste à prendre un jeu de radiographies (comme chez le médecin, avec un rayonnement plus puissant) en faisant tourner la pièce entre chaque prise de vue. En appliquant un algorithme de reconstruction idoine à l'ensemble des clichés, il est possible de remonter à une image 3D des pièces scannées. Plus la zone que l'on traverse est dense plus elle est claire (comme chez le médecin : vos os apparaissent plus clair que vos muscles). Dans notre cas, le constraste entre les défauts constitués d'air et le métal est très marqué : on observe donc les défauts en noir et le métal en gris. Un défaut est donc un amas de voxels (l'équivalent des pixels pour une image 3D) noirs. Sur l'image ci-dessous, les défauts ont été extraits et sont représentés en 3D par les volumes rouges.
#
# <img src="media/defects_3D.png" width="400px">
#
# Vous voyez qu'ils sont nombreux, de taille et de forme variées. À chaque volume rouge que vous observez correspond une ligne de votre `DataFrame` qui contient les descripteurs morphologiques du-dit défaut.
#
#
# #### Descripteur $r_1$ (`radius1`)
# En notant $N$ le nombre de voxels constituant le défaut, on obtient le volume du défaut $V=N\times v_0$ (où $v_0$ est le volume d'un voxel).On peut alors définir son *rayon équivalent* comme le rayon de la sphère de même volume soit :
# \begin{equation*}
#  R_{eq} = \left(\frac{3V}{4\pi}\right)^{1/3}
# \end{equation*}
#
# On définit ensuite le *rayon moyen* $R_m$ du défaut comme la moyenne sur tous les voxels de la distance au centre de gravité du défaut.
#
# $R_{eq}$ et $R_m$ portent une information sur la taille du défaut. En les combinant comme suit:
# \begin{equation*}
#  r_1 = \frac{R_{eq} - R_m}{R_m}
# \end{equation*}
# on la fait disparaître : $r_1$ vaut 1/3 pour une sphère quel que soit son rayon.
#
# **Note :** vous aurez remarqué que $r_1$ est donc sans dimension.
#
# #### Descripteurs basés sur la matrice d'inertie ($\lambda_1$ et $\lambda_2$) (`lambda1`, `lambda2`)
# La matrice d'inertie de chaque défaut est calculée. Pour ce faire, on remplace tout simplement les intégrales sur le volume présentes dans les formules que vous connaissez par une somme sur les voxels. Par exemple:
# \begin{equation}
#  I_{xy} = -\sum\limits_{v\in\rm{defect}} (x(v)-\bar{x})(y(v)-\bar{y})\qquad \text{avec } \bar{x} = \frac{1}{N}\sum\limits_{v\in\rm{defect}} x(v) \text{ et } \bar{y} = \frac{1}{N}\sum\limits_{v\in\rm{defect}} y(v)
# \end{equation}
# Cette matrice est symétrique réelle, elle peut donc être diagonalisée. Les trois valeurs propres obtenues $I_1 \geq I_2 \geq I_3$ sont les moments d'inertie du défaut dans son repère principal d'inertie. Ces derniers portent de manière intrinsèque une information sur le volume du défaut. Pour la faire disparaître, il suffit de normaliser ces moments comme suit :
#
# \begin{equation}
#  \lambda_i = \frac{I_i}{I_1+I_2+I_3}
# \end{equation}
#
# On obtient alors trois indicateurs $\lambda_1 \geq \lambda_2 \geq \lambda_3$ vérifiant notamment $\lambda_1 + \lambda_2 + \lambda_3 = 1$ ce qui explique que l'on ne garde que les deux premiers. En utilisant les propriétés des moments d'inertie, on peut montrer que les points obtenus se situent dans le triangle formé par $(1/3, 1/3)$, $(1/2, 1/4)$ et $(1/2, 1/2)$ dans le plan $(\lambda_1, \lambda_2)$. Vous pourrez vérifier cela dans la suite !
#
# La position du point dans le triangle renseigne sur sa forme *globale*, comme indiqué par l'image suivante :
# ∑
# <img src="media/l1_l2.png" width="400px">
#
# #### Convexité (`convexity`)
#
# L'indicateur de convexité utilisé est simplement le rapport entre le volume du défaut et de son convexe englobant. $C = V/V_{CH} \leq 1$. Lorsque qu'un défaut est convexe, il est égal à son convexe englobant et donc $C$ vaut 1.
#
# #### Sphéricité (`sphericity`)
#
# L'indicateur de sphéricité permet de calculer l'écart d'un défaut à une sphère. On utilise ici la caractéristique de la sphère qui minimise la surface extérieure pour un volume donné. La grandeur :
# \begin{equation*}
# I_S = \frac{6\sqrt{\pi}V}{S^{3/2}}
# \end{equation*}
# où $V$ est le volume du défaut et $S$ sa surface vaut 1 pour une sphère et est inférieur à 1 sinon.
#
# #### Indicateurs basés sur la mesure de la courbure (`varCurv`, `intCurv`)
#
# Les deux courbures principales $\kappa_1$ et $\kappa_2$ sont calculées en chaque point de la surface des défauts ([ici pour les plus curieux](https://fr.wikipedia.org/wiki/Courbure_principale)). Ces courbures permettent de caractériser la forme locale du défaut. Elle sont combinées pour calculer la courbure moyenne $H = (\kappa_1+\kappa_2)/2$ et la courbure de Gauss $\gamma = \kappa_1\kappa_2$. Pour s'affranchir de l'information sur la taille (pour une sphère de rayon $R$, on a en tout point $\kappa_1 = \kappa_2 = 1/R$), les défauts sont normalisés en volume avant d'en calculer les courbures.
# Les indicateurs retenus sont les suivants:
#
#  - la variance de la courbure de Gauss (colonne `varCurv`) $Var(H)$ ;
#  - l'intégrale de $\gamma$ sur la surface du défaut(colonne `intCurv`) $\int_S \gamma dS$.
#
# Ces indicateurs valent respectivement $0$ et $4\pi$ pour une sphère.
#
# #### Indicateurs sur la boite englobante $(\beta_1, \beta_2)$ (`b1`, `b2`)
#
# Finalement, c'est une information sur la boite englobante du défaut dans son repère principal d'inertie qui est cachée dans $(\beta_1, \beta_2)$. En notant $B_1, B_2, B_3$ les dimensions (décroissantes) de la boite englobante, on réalise la même normalisation que pour les moments d'inertie :
# \begin{equation}
#  \beta_i = \frac{B_i}{B_1+B_2+B_3}
# \end{equation}
#
# ***

# %% [markdown]
# ## Visualisation des défauts
#
# Pour que vous saissiez un peu mieux la signification des descripteurs morphologiques, nous avons concocté une petite fonction utilitaire qui vous permettra de visualiser les défauts. Pour que vous puissiez interagir avec `pyplot`, il nous est imposé de changer le backend avec la commande `%matplotlib notebook` et de recharger le module. Pour revenir dans le mode de visualisation précédent, vous devrez évaluer la cellule qui contient la commande `%matplotlib inline` qui arrive un peu plus tard !
#
# *Nous n'avons malheureusement pas trouvé de solution pour que ce changement soit transparent pour vous... :(*
#
# Amusez vous à chercher des défauts **extrêmes** pour comprendre. Par exemple le défaut qui maximise $\lambda_2$ sera celui qui a la forme la plus *filaire* alors que celui qui minimise aura la forme la plus *plate*. Pourquoi ne pas jeter un coup d'oeil au défaut le moins convexe ?

# %%
# Ne touchez pas à ça c'est pour la visualisation interactive !
# %matplotlib notebook
reload(plt)
# À partir de maintenant vous pouvez vous amuser !

# On récupère un id intéressant
id_to_plot = df.index[df['convexity'].argmin()]
# On affiche à l'écran les valeurs de ses descripteurs
print(df.loc[id_to_plot])
# On l'affiche
plot_defect(id_to_plot)

# %% [markdown]
# N'oubliez pas d'aller rendre visite au défaut avec l'id `4022` qui a une forme rigolote, avec ses petites excroissances.

# %%
# Le défaut 4022 !
print(df.loc[4022])
plot_defect(4022)

# %% [markdown]
# On vous parlait juste avant de défauts de morphologie proche ! Et si une simple distance euclidienne en dimension 9 fonctionnait ? Calculez le défaut le plus proche du défaut `4022` dans l'espace de dimension 9, et tracez-le ! Se ressemblent-ils ?

# %%
df1 = df-df.loc[4022]
mat = df1.to_numpy()
dist = np.apply_along_axis(np.linalg.norm, 1, mat)
df1 = df1.assign(dist=dist)
df1 = df1.drop(index=4022)

# %%
min_ind = df1.index[df1['dist'].argmin()]
print(df1.loc[min_ind])
plot_defect(min_ind)

# %% [markdown]
# **Eh non!** Le défaut le plus proche du défaut `4022` est une patatoïde quelconque. Deux explications sont possibles :
#  * soit la distance euclidienne n'est pas pertinente ici ;
#  * soit le défaut `4022` est le seul avec des petites excroissances...
# Je vous laisse aller voir le défaut `796` pour trancher entre les deux propositions..
#

# %%
print(df.loc[796])
plot_defect(796)

# %%
# On revient en mode de visu standard après avoir évalué cette cellule
# %matplotlib inline
reload(plt)


# %% [markdown]
# ## Visualisation des données
#
# Avant de commencer toute analyse à proprement parler de données, il est nécessaire de passer un moment à les observer.
#
# Pour ce faire, nous allons écrire des fonctions utilitaires qui se chargeront de tracer des graphes classiques.

# %% [markdown]
# ### Tracé d'un histogramme
#
# Écrire une fonction `histogram` qui trace l'histogramme d'une série de points.
#
#
# Par exemple l'appel `histogram(df['radius1'], nbins=10)` devrait renvoyer quelque chose qui ressemble à ceci:
#
# <img src="media/defects-histogram.png" width="400px">

# %%
def histogram(fonct, nbins):
    """
    Displays a histogram

    Parameters:
        funct (pandas.core.series.Series): function we want to display
        nbins (int): number of histogram column

    Returns:
        nothing 
   """
    plt.hist(fonct, bins=nbins)
    plt.show()


# %%
# pour vérifier
histogram(df['radius1'], nbins=10)


# %%
# pour vérifier
# c'est bien si votre fonction marche aussi avec une dataframe
histogram(df[['radius1']], nbins=10)


# %% [markdown]
# #### *Bonus* : un histogramme adaptable aux goûts de chacun
#
# Modifier la fonction `histogram` pour que l'utilisateur puisse préciser par exemple: le nom des étiquettes des axes, les couleurs à utiliser pour représenter l'histogramme...
#
# Par exemple en appelant
# ```python
# histogram2(df['radius1'], nbins=10,
#        xlabel='radius1', ylabel='occurrences',
#        histkwargs={'color': 'red'})
# ```
#
# on obtiendrait quelquechose comme ceci
#
# <img src="media/defects-histogram2.png" width="400px">

# %%
def histogram2(fonct, nbins, xlabel=None, ylabel=None, hist_kwargs=None):
    """
    Displays a histogram

    Parameters:
        funct (pandas.core.series.Series): function we want to display
        nbins (int): number of histogram column
        xlabel (str): legend in x
        ylabel (str): legend in y
        hist_kwargs (dic): indicating the color of the histogram

    Returns:
        nothing 
   """
    plt.hist(fonct, bins=nbins, color=hist_kwargs['color'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# %%
# pour vérifier

# seulement si la fonction est définie

if 'histogram2' in globals():
    histogram2(df['radius1'], nbins=10,
                     xlabel='radius1',ylabel='occurrences',
               hist_kwargs={'color': 'red'})
else:
    print("vous avez choisi de ne pas faire le bonus")


# %% [markdown]
# ### Tracé de nuages de points
#
# Écrire une fonction `correlation_plot` qui permet de tracer le nuage de points entre deux séries de données.
# L'appel de cette fontion comme suit `correlation_plot(df['lambda1'], df['lambda2'])` devrait donner une image ressemblant à celle-ci :
#
# Ces tracés illustrent le *degré de similarité* des colonnes. Notons, qu'il existe des indices de similarité comme par exemple: la covariance, le coefficient de corrélation de Pearson...
#
#
# <img src="media/defects-correlation.png" width="400px">

# %%
def correlation_plot(fonctx, foncty):
    """
    Displays a point cloud

    Parameters:
        functx (pandas.core.series.Series): first series of data that we want to put in x
        functx (pandas.core.series.Series): second series of data that we want to put in there

    Returns:
        nothing 
   """
    x = list(fonctx)
    y = list(foncty)

    plt.scatter(x, y, s=10)

    plt.title('...')
    plt.xlabel('...')
    plt.ylabel('...')

    plt.show()


# %%
# pour vérifier
correlation_plot(df['lambda1'], df['lambda2'])


# %% [markdown]
# #### *Bonus* les nuages de points pour l'utilisateur casse-pieds (ou daltonien ;) )
# Modifier la fonction `correlation_plot` pour que l'utilisateur puisse préciser des noms pour les axes, et choisir l'aspect des points tracés (couleur, taille, forme, ...).
#
# par exemple en appelant
# ```python
# correlation_plot2(df['lambda1'], df['lambda2'],
#                   xlabel='lambda1', ylabel='lambda2',
#                   plot_kwargs={'marker': 'x', 'color': 'red'})
# ```
# on obtiendrait quelque chose comme
#
# <img src="media/defects-correlation2.png" width="400px">

# %%
def correlation_plot2(fonctx, foncty, xlabel=None, ylabel=None, plot_kwargs=None):
    """
    Displays a custom point cloud

    Parameters:
        functx (pandas.core.series.Series): first series of data that we want to put in x
        functx (pandas.core.series.Series): second series of data that we want to put in there
        xlabel, ylabel (str): represent respectively the str of the abscissa and the ordinate
        plot_kwargs (dic):
            -'marker '(str): type of marker desired
            -'color '(str): desired point color

    Returns:
        nothing 

    """
    x = list(fonctx)
    y = list(foncty)

    plt.scatter(
        x, y, s=30, color=plot_kwargs['color'], marker=plot_kwargs['marker'])

    plt.title('...')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    

# %%
# pour vérifier
# seulement si la fonction est définie
if 'correlation_plot2' in globals():
    correlation_plot2(df['lambda1'], df['lambda2'],
                      xlabel='lambda1', ylabel='lambda2',
                      plot_kwargs={'marker': 'x', 'color': 'red'})
else:
    print("vous avez choisi de ne pas faire le bonus")

# %% [markdown]
# #### *Bonus 2* Tracer le triangle d'inertie en plus
#
# On vous disait plus tôt que les points dans le plan $(\lambda_1, \lambda2)$ sont forcément dans le triangle formé par $(1/3, 1/3)$, $(1/2, 1/4)$ et $(1/2, 1/2)$.
# Essayez de superposer les données au triangle pour mettre cela en évidence !
# Le résultat pourrait ressembler à ceci :
#
# <img src="media/defects-correlation3.png" width="400px">
#
# (Vous pouvez faire ce bonus sans avoir fait le précédent)

# %%
correlation_plot2(df['lambda1'], df['lambda2'],
                  xlabel='lambda1', ylabel='lambda2',
                  plot_kwargs={'marker': '.', 'color': 'black'})

x = [1/3, 1/2, 1/2, 1/3]
y = [1/3, 1/4, 1/2, 1/3]

plt.plot(x, y,"--", color="red", marker=".")
plt.axis('equal')
plt.show()


# %% [markdown]
# ### Affichage de tous les plots des colonnes
#
# Écrire une fonction `plot2D` qui prend en argument une dataframe et qui affiche
# * les histogrammes des colonnes
# * les plots des corrélations des couples des colonnes
# n'affichez qu'une seule fois les corrélations par couple de colonnes
#
# avec `plot2D(df[['radius1', 'lambda1', 'lambda2']])` vous devriez obtenir quelque chose comme ceci
#
# <img src="media/defects-plot2d.png" width="200px">

# %%
def plot2D(functions):
    """
    Attach:
        -The column histograms
        -The correlations plots of the pairs of columns display only once the correlations per pair of columns

    Parameters:
        functions (pandas.core.series.Series): series of points

    Returns:
        nothing 
    """

    head = list(functions.head(0))
    size = len(head)
    for funct in functions:
        plt.hist(df[funct], bins=100)
        plt.xlabel(funct)
        plt.ylabel('Frequency')
        plt.show()
    size = len(list(functions.head(0)))
    for i in range(size):
        for j in range(i):
            x = list(functions.iloc[:, i])
            y = list(functions.iloc[:, j])
            plt.scatter(x, y, s=10)
            plt.xlabel(head[i])
            plt.ylabel(head[j])
            plt.show()


# %%

# pour corriger
plot2D(df[['radius1', 'lambda1', 'lambda2']])


# %% [markdown]
# #### *Bonus++* (dataviz expert-level) le tableau des plots des colonnes
#
# Écrire une fonction `scatter_matrix` qui prend une dataframe en argument et affiche un tableau de graphes avec
# * sur la diagonale les histogrammes des colonnes
# * dans les autres positions les plots des corrélations des couples des colonnes
#
# avec `scatter_matrix(df[['radius1', 'lambda1', 'b2']], nbins=100, hist_kwargs={'fc': 'g'})` vous devriez obtenir à peu près ceci
#
# <img src="media/defects-matrix.png" width="500px">

# %%
def scatter_matrix(functions, nbins, hist_kwargs=None):
    """
    Attach:
     -on the diagonal the histograms of the columns
     -in the other positions the plots of the correlations of the pairs of the columns

    Parameters:
        functions (pandas.core.series.Series): series of points
        nbins (int): histogram definitions
        hist_kwargs (dic): 'fc' (str): color of histograms

    Returns:
        nothing 
    """
    head = list(functions.head(0))
    size = len(head)
    fig, axs = plt.subplots(size, size, figsize=(15, 15))
    for i in range(size):
        for j in range(size):
            if i == j:
                axs[i, j].hist(df[head[i]], bins=nbins,
                               color=hist_kwargs['fc'])
            else:
                x = list(functions.iloc[:, i])
                y = list(functions.iloc[:, j])
                axs[j, i].scatter(x, y, s=10, color='black')
            if i != size-1 and j > 0:
                axs[i, j].tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False)

    for n in range(size-1):
        axs[n, 0].tick_params(left=True, bottom=False,
                              labelleft=True, labelbottom=False)
        axs[size-1, 1+n].tick_params(left=False, bottom=True,
                                     labelleft=False, labelbottom=True)

    [axs[size-1, n].set_xlabel(head[n])for n in range(size)]

    [axs[n, 0].set_ylabel(head[n])for n in range(size)]

    fig.tight_layout()
    plt.show()


# %%
scatter_matrix(df[['radius1', 'lambda1', 'b2']],
               nbins=100, hist_kwargs={'fc': 'g'})

# %% [markdown]
# ### Corrélations entre les données
#
# Utilisez les fonctions que vous venez d'implémenter pour rechercher (visuellement) les meilleures correlations qui ressortent entre les différentes caractéristiques morphologiques.
#
# Plottez la corrélation qui vous semble la plus frappante (i.e. la plus informative), motivez votre choix.
#

# %%
scatter_matrix(df[:], nbins=100, hist_kwargs={'fc': 'g'})

# %% [markdown]
# Plottez la corrélation qui me semble la plus frappante est celle entre b2 et lambda 2. Les points semble s'aligner sur une droite ce qui démontre bien une corrélation.

# %%
help(correlation_plot2)
correlation_plot2(df['b2'], df['lambda2'], xlabel="b2",ylabel="lambda2")


# %% [markdown]
# ## Analyse en composantes principales (ACP)
#
# Les corrélations entre variables mises en évidence précédemment nous indiquent que certaines informations, apportées par les indicateurs, sont parfois redondantes.
#
# L'analyse en composantes principales est une méthode qui va permettre de construire un jeu de *composantes principales* qui sont des combinaisons linéaires des caratéristiques. Ces composantes principales sont indépendantes les unes des autres. Ce type d'analyse est généralement mené pour réduire la dimension d'un problème.
#
# La construction des composantes principales repose sur l'analyse aux valeurs propres d'une matrice indiquant les niveaux de corrélations entre les caractéristiques. En notant $X$ la matrice contenant nos données qui est donc constituée ici de 4040 lignes et 9 colonnes, la matrice de corrélation des caractéristiques demandée ici est $C = X^TX$.

# %% [markdown]
# ### Construction des composantes principales sur les caractéristiques morphologiques
#
# Construisez une matrice des niveaux de corrélation des caractéristiques. Elle doit être carrée de taille 9x9.

# %%
matrix_X = df.to_numpy()
matrix_XT = np.transpose(matrix_X)
C = np.dot(matrix_XT, matrix_X)
"Obtient donc une matrice c de taille 9x9 = ",C


# %% [markdown]
# ### Calcul des vecteur propres et valeurs propres de la matrice de corrélation

# %% [markdown]
# Calculez à l'aide du module `numpy.linalg` les valeurs propres et les vecteurs propres de la matrice $C$. Cette dernière est symétrique définie positive par construction, toutes ses valeurs propres sont strictement positives.

# %%
eigv, vect = np.linalg.eig(C) #On détermine les valeurs propres et les vecteurs propres


# %% [markdown]
# ### Tracé des valeurs propres

# %% [markdown]
# Tracez les différentes valeurs propres calculées en utilisant un axe logarithmique.

# %%
X = [a for a in range(len(eigv))]
Y = [np.log10(a) for a in eigv]
plt.plot(X, Y, color='black')
plt.scatter(X, Y, color='black')
#plt.yscale("log")
plt.xlabel("Puissance de 10")
plt.ylabel("log des valeurs propres")


# %% [markdown]
# ### Analyse de l'importance relative des composantes principales

# %% [markdown]
# Vous devriez constater que les valeurs propres décroissent vite. Cette décroissance traduit l'importance relative des composantes principales.
#
# Dans le cas où les valeurs propres sont ordonnées de la plus grande à la plus petite ($\forall (i,j) \in \{1, \ldots, N\}^2, i>j
#  \Rightarrow \lambda_i \leq \lambda_j$), tracez l'évolution de la quantité suivante:
# \begin{equation*}
#  \alpha_i = \frac{\sum\limits_{j=1}^{j=i}\lambda_j}{\sum\limits_{j=1}^{j=N}\lambda_j}
# \end{equation*}
#
# $\alpha_i$ peut être interprété comme *la part d'information du jeu de données initial contenu dans les $i$ premières composantes principales*.

# %%
eigv_sort = np.sort(eigv)[::-1]
alpha = eigv_sort.copy()
for i in range(len(eigv_sort)):
    alpha[i] = sum(eigv_sort[:i+1])/sum(eigv_sort)
plt.plot(X, alpha, color='black')
plt.scatter(X, alpha, color='black')
plt.xlabel("i")
plt.ylabel("alpha i")

alpha

# %% [markdown]
# ### Quantité d'information contenue par la plus grande composante principale
#
# Affichez la plus grande valeur propre et le vecteur propre correspondant (ça doit correspondre à la première composante principale).
#
# Quelle est la quantité d'information contenue par cette composante ?
#
# Pratiquement toute l'information ! C'est trop beau pour être vrai non ?
#
# Affichez les coefficients de cette composante principale. Que remarquez vous ? (*hint* cherchez la caractéristique dont le coefficient est le plus important en valeur absolue)
#
# En observant les données correspondant à cette caractéristique, avez-vous une idée de ce qui s'est passé ?

# %%
max = np.argmax(eigv)
print("valeur propre maximal", eigv[max],
      "\nvecteur propre associé", vect[max])
# Cette valeur propre contient 99% de l'informations

coo = np.argmax(C)
print((coo+1)//9, 61-coo//9*9)
print(
    "On en déduit que les coordonnées sont donc (7,7) \nLe coefficient le plus important est", C[7, 7])
print("Les coefficients de cette composante sont", C[:, 7])
print("la composante principale correspond à la projection dans l'axe principale")
print(C@np.transpose(vect[0]))


# %% [markdown]
# On remarque que la caractéristique dont le coefficient est le plus important l'ai également avec des autres coefficient. De plus, la matrice de corrélation est la matrice des produits scalaire. Les coefficients de cette composante ont une grande disparité et demande à être normalisé. En effet, étant donné qu'ils ont une valeur beaucoup plus importante que les autres coefficients lors des analyses il dissimule les variations des autres coefficients.
#
#
#

# %% [markdown]
# ## ACP sur les caractéristiques standardisées
#
# Dans la section précédente, la première composante principale ne prenait en compte que la caractéristique de plus grande variance. Un moyen de s'affranchir de ce problème consiste à **standardiser** les données. Pour un échantillon $Y$ de taille $N$, la variable standardisée correspondante est $Y_{std}=(Y-\bar{Y})/\sigma(Y)$ où $\bar{Y}$ est la moyenne empirique de l'échantillon et $\sigma(Y)$ son écart type empirique.
#
# **Notez que** dans notre cas, il faut réaliser la standardisation **caractéristique par caractéristique** (soit colonne par colonne). Si vous n'y avez pas encore pensé, refaites un petit tour sur le cours d'agrégation pour faire ça de manière super efficace ! ;)
#
# Menez la même étude que précédement (i.e. à partir de la section `Analyse en composantes principales`) jusqu'à tracer l'évolution des $\alpha_i$.

# %%

Y_mean = matrix_X.sum(axis=0)*1/(len(matrix_X))
Y_standard_deviation = matrix_X.std(0)

Y_std = (matrix_X-Y_mean)/Y_standard_deviation
Y_std #matrice standardisée


matrix_YT = np.transpose(Y_std)
C_Y = np.dot(matrix_YT, Y_std)#matrice de corélation standardisée 

Y_eigv, Y_vect = np.linalg.eig(C_Y) #valeurs propres et vecteurs propre de la matrice de corélation standardisée 



# %%
X = [a for a in range(len(Y_eigv))]
Y = [np.log10(a) for a in Y_eigv]
plt.plot(X, Y, color='black')
plt.scatter(X, Y, color='black')
plt.xlabel("Puissance de 10")
plt.ylabel("log des valeurs propres")

# %%
Y_eigv_sort = np.sort(Y_eigv)[::-1]
Y_alpha = Y_eigv_sort.copy()
for i in range(len(Y_eigv_sort)):
    Y_alpha[i] = sum(Y_eigv_sort[:i+1])/sum(Y_eigv_sort)
    
plt.plot(X, Y_alpha, color='black')
plt.scatter(X, Y_alpha, color='black')
plt.xlabel("i")
plt.ylabel("alpha i")

# %% [markdown]
# ### Importance des composantes principales
#
# Quelle part d'information est contenue dans les 3 premières composantes principales ?
#

# %%
Y_alpha[2]
# 86% de l'information est contenu par les 3 premières composantes

# %% [markdown]
# Cette part d'information est satisfaisante car nous avons tout de même réduit notre dimension de 9 à 3.

# %% [markdown]
# ### Projection dans la base des composantes principales de nos 4040 défauts
#
# On va convertir les données initiales dans la base des (vecteurs propres des) composantes principales, et les projeter sur l'espace engendré par les premiers vecteurs propres.
#
# Faites attention, vous calculez désormais dans les données standardisées.
#
# Créez une nouvelle dataframe dont les colonnes correspondent aux projections sur le sous-espace des 3 vecteurs propres prépondérants; on appellera ses colonnes P1, P2 et P3

# %%

P1 = Y_std@np.transpose(Y_vect[0])
P2 = Y_std@np.transpose(Y_vect[1])
P3 = Y_std@np.transpose(Y_vect[2])
P4 = Y_std@np.transpose(Y_vect[3])
P5 = Y_std@np.transpose(Y_vect[4])


correlation_plot2(P1, P2, xlabel='P1', ylabel='P2')
plt.show()
correlation_plot2(P1, P3, xlabel='P1', ylabel='P3')
plt.show()


# %% [markdown]
# Tracez les nuages de points correspondants dans les plans (P1, P2) et (P1, P3).

# %% [markdown]
# ## La conclusion
#
# Reprenez maintenant le défaut `4022` et cherchez son plus proche voisin en utilisant la distance euclidienne dans l'espace des composantes principales. Que constatez-vous ?

# %%
# %matplotlib notebook
reload(plt)

# %%
Y_mean_tab = df.sum(axis=0)*1/(len(df))
Y_standard_deviation_tab = df.std(0)
Y_std_tab = (df-Y_mean_tab)/Y_standard_deviation_tab


# %%
dfY = pd.DataFrame({"P1":P1,"P2":P2,"P3":P3,"P4":P4,"P5":P5}) 
df_2=df = pd.read_csv("defect_data.csv", usecols=range(10))
dfY=dfY.set_index(df_2['id'])

"""
si on met seulement P1,P2,P3 
on obtient un patatoïde (1125) 
si on rajoute P4 
on se raproche d'une forme avec des excroissances (3415) 
Si on rajoute enfin P5 
On obtient bien le défaut attendu (796)
"""

# %%
dfY=dfY-dfY.loc[4022]
mat_Y2 = dfY.to_numpy()
dist_Y2 = np.apply_along_axis(np.linalg.norm, 1, mat_Y2)
dfY = dfY.assign(dist=dist_Y2)
dfY = dfY.drop(index=4022)
min_ind = dfY.index[dfY['dist'].argmin()]
print(dfY.loc[min_ind])

plot_defect(min_ind)

# %%
Y_std_tab_1 = Y_std_tab-Y_std_tab.loc[4022]
mat_Y = Y_std_tab_1.to_numpy()
dist_Y = np.apply_along_axis(np.linalg.norm, 1, mat_Y)
Y_std_tab_1 = Y_std_tab_1.assign(dist=dist_Y)
Y_std_tab_1 = Y_std_tab_1.drop(index=4022)
min_ind = Y_std_tab_1.index[Y_std_tab_1['dist'].argmin()]
print(Y_std_tab_1.loc[min_ind])

plot_defect(min_ind)


# %% [markdown]
# Ici on remarque que si on calcul la distance sur la matrice standardisée on obtient bien le même défaut qu'avec les projections.

# %% [markdown]
# ***On trouve bien une patatoïde avec des excroissances ce qui est beaucoup plus proche du défaut 4022!!***
#
#
#

# %%
# %matplotlib inline
reload(plt)

# %% [markdown]
# **Une note de fin :** L'objectif de la démarche n'est pas seulement de trouver le plus proche voisin, mais l'ensemble des voisins (recherchez `triangulation de Delaunay` ou `tesselation de Voronoï` si vous être curieux). Or bien que les algorithmes de construction des triangulations/tessellations sont écrits en dimension quelconque, ils sont beaucoup plus efficaces en dimension faible. Dans le cas actuel, une triangulation en dimension 3 est instantanée (avec `scipy.spatial.Delaunay`) alors qu'elle met au moins une heure en dimension 9 (peut-être beaucoup plus, j'ai dû couper car mon train arrivait à destination...)

# %% [markdown]
# ***
