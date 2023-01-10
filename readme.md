# Précision(Precision) exactitude(Accuracy), et F1 score
## Précision(Precision) exactitude(Accuracy)

La précision et l'exactitude sont deux façons pour les scientifiques de considérer l'erreur.
L'exactitude désigne la proximité d'une mesure par rapport à la valeur réelle ou acceptée.
La précision fait référence à la proximité des mesures d'un même élément les unes par rapport aux autres. La précision est indépendante de l'exactitude.
Cela signifie qu'il est possible d'être très précis mais pas très exact, et qu'il est également possible d'être exact sans être précis.
Les observations scientifiques de la meilleure qualité sont à la fois exactes et précises.


Une façon classique de démontrer la différence entre la précision et l'exactitude est d'utiliser une cible de fléchettes. Imaginez que le centre d'une cible de fléchettes représente la valeur réelle. Plus les fléchettes atterrissent près du centre, plus elles sont précises.

- Si les fléchettes ne sont ni proches du centre, ni proches les unes des autres, il n'y a ni exactitude, ni précision.  
- Si toutes les fléchettes atterrissent très près les unes des autres, mais loin du centre, il y a précision, mais pas d'exactitude.   
- Si les fléchettes sont toutes à peu près à la même distance du centre et espacées de manière égale autour de celui-ci, il y a précision mathématique car la moyenne des fléchettes se trouve au centre. Cela représente des données exactes, mais pas précises. Cependant, si vous jouiez réellement aux fléchettes, cela ne serait pas considéré comme au centre !
- Si les fléchettes atterrissent près du centre et proches les unes des autres, il y a à la fois exactitude et précision.

## Exactitude (Accuracy)
**L'accuracy** est une métrique pour les modèles de classification qui mesure le nombre de prédictions correctes en pourcentage du nombre total de prédictions effectuées. Par exemple, si 90 % de vos prédictions sont correctes, votre précision est simplement de 90 %.

$$ Accuracy = \frac{Nb\~correct\~predictions}{Nb\~total\~predictions}$$

**L'accuracy** est une mesure utile uniquement lorsque la distribution des classes est égale dans votre classification. Cela signifie que si vous avez un cas d'utilisation dans lequel vous observez plus de points de données d'une classe que d'une autre, la précision n'est plus une métrique utile. Prenons un exemple pour illustrer cela :

# Exemple de données déséquilibrées(Imbalanced data)
Imaginez que vous travaillez sur les données de vente d'un site Web. Vous savez que 99 % des visiteurs du site n'achètent pas et que seulement 1 % des visiteurs achètent quelque chose. Vous construisez un modèle de classification pour prédire quels visiteurs du site sont des acheteurs et quels sont ceux qui ne font que regarder.

Imaginez maintenant un modèle qui ne fonctionne pas très bien. Il prédit que 100 % de vos visiteurs ne sont que des spectateurs(lookers) et que 0 % de vos visiteurs sont des acheteurs. Il s'agit clairement d'un modèle très mauvais et inutile.

Que se passerait-il si nous utilisions la formule de l'accuracy sur ce modèle ? Votre modèle n'a fait que 1% de prédictions erronées : tous les acheteurs ont été classés à tort dans la catégorie "lookers". Le pourcentage de prédictions correctes est donc de 99%. **Le problème ici est qu'une Accuracy de 99 % semble être un excellent résultat, alors que les performances de votre modèle sont très faibles.**

En conclusion : **l'Accuracy n'est pas une bonne mesure à utiliser lorsque vous avez un déséquilibre de classe.**

# Résoudre les données déséquilibrées par le rééchantillonnage
L'une des façons de résoudre les problèmes de déséquilibre des classes consiste à travailler sur votre échantillon. Grâce à des méthodes d'échantillonnage spécifiques, vous pouvez rééchantillonner votre ensemble de données de manière à ce que les données ne soient plus déséquilibrées. Vous pouvez alors utiliser à nouveau l'Accuracy comme métrique.

# Résoudre les données déséquilibrées grâce à des métriques
Une autre façon de résoudre les problèmes de déséquilibre des classes consiste à utiliser de meilleures métriques que Accuracy, comme le **score F1**, qui prennent en compte non seulement le nombre d'erreurs de prédiction commises par votre modèle, mais aussi le type d'erreurs commises.

## Precision et Recall: les fondements du score F1
**Precision et Recall** sont les deux métriques les plus courantes qui prennent en compte le déséquilibre des classes. Elles sont également à la base du score F1 ! Examinons de plus près *Precision* et *Recall* avant de les combiner dans le score F1 dans la partie suivante.

# Precision
Precision est la première partie du score F1. Elle peut également être utilisée comme une métrique individuelle d'apprentissage automatique. Sa formule est présentée ici :

$$ Precision = \frac{Nb\~of\~True\~Positives}{Nb\~of\~True\~Positives + Nb\~of\~False\~Positives}$$

Vous pouvez interpréter cette formule comme suit. **Parmi tout ce qui a été prédit comme positif, *Precision* compte le pourcentage qui est correct** :

- Un modèle peu précis peut trouver un grand nombre de positifs, mais sa méthode de sélection est bruyante : il détecte également à tort de nombreux positifs qui ne le sont pas réellement.
- Un modèle précis est très "pur" : il ne trouve peut-être pas tous les positifs, mais ceux que le modèle classe comme positifs sont très probablement corrects.

# Recall
*Recall*  est la deuxième composante du score F1, bien que *Recall* puisse également être utilisé comme une métrique d'apprentissage automatique individuelle. La formule du rappel est présentée ici :

$$ Recall = \frac{Nb\~of\~True\~Positives}{Nb\~of\~True\~Positives + Nb\~of\~False\~Negatives}$$

Vous pouvez interpréter cette formule comme suit. **Parmi tous les cas qui sont effectivement positifs, combien le modèle a-t-il réussi à en trouver ?**

- Un modèle avec un Recall élevé réussit à trouver tous les cas positifs dans les données, même s'il peut aussi identifier à tort certains cas négatifs comme des cas positifs.
- Un modèle à faible Recall n'est pas en mesure de trouver tous (ou une grande partie) des cas positifs dans les données.

## Precision vs Recall
Pour clarifier, pensez à l'exemple suivant d'un supermarché qui a vendu un produit présentant un problème et qui doit le rappeler : il ne s'intéresse qu'à s'assurer de retrouver tous les produits problématiques. Il leur importe peu que les clients renvoient également des produits non problématiques, la *Precision* n'est donc pas intéressante pour ce supermarché.

## Compromis Precision-Recall
Idéalement, nous voudrions les deux : un modèle qui identifie tous nos cas positifs et qui, en même temps, n'identifie que les cas positifs.

Dans la vie réelle, nous devons malheureusement faire face à ce que l'on appelle le compromis Precision-Recall.

Le compromis Precision-Recall représente le fait que, dans de nombreux cas, vous pouvez modifier un modèle pour augmenter Precision au prix d'une baisse du Recall ou, au contraire, augmenter le Recall au prix d'une baisse de Precision.

# Le score F1 : combinaison de Precision et Recall

*Precision* et *Recall* sont les deux éléments constitutifs du **score F1**. L'objectif du score F1 est de combiner les mesures de *Precision* et de *Recall* en une seule mesure. En même temps, le score F1 a été conçu pour fonctionner correctement sur des données déséquilibrées.

$$ F1\~score = 2 * \frac{Precision * Recall}{Precision + Recall}$$

Comme le score F1 est une moyenne de *Precision* et *Recall*, cela signifie que le score F1 donne un poids égal Precision et Recall :

- Un modèle obtiendra un score F1 élevé si *Precision* et *Recall* sont tous deux élevés.
- Un modèle obtiendra un score F1 faible si *Precision* et *Recall* sont tous deux faibles.
- Un modèle obtiendra un score F1 moyen si l'un des critères *Precision* et *Recall* est faible et l'autre élevé.
