#Précision(Precision) exactitude(Accuracy), et F1 score
##Précision(Precision) exactitude(Accuracy)

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

##Exactitude (Accuracy)

**L'accuracy** est une métrique pour les modèles de classification qui mesure le nombre de prédictions correctes en pourcentage du nombre total de prédictions effectuées. Par exemple, si 90 % de vos prédictions sont correctes, votre précision est simplement de 90 %.

$$Accuracy = \frac{Nb~correct~predictions}{Nb~total predictions}$$

**L'accuracy** est une mesure utile uniquement lorsque la distribution des classes est égale dans votre classification. Cela signifie que si vous avez un cas d'utilisation dans lequel vous observez plus de points de données d'une classe que d'une autre, la précision n'est plus une métrique utile. Prenons un exemple pour illustrer cela :