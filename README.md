# Online Principal Componente Analysis
In this repo, a comparison between batch PCA (traditional) and online PCA (PCA updated every new observation _arrives_) is carried out using a random forest classifier on the first 50 principal components of the MNIST dataset. (Spoiler alert: online PCA performed better.)

Online PCA is implemented using auxiliary functions from [onlinePCA package]('https://cran.r-project.org/web/packages/onlinePCA/onlinePCA.pdf).
