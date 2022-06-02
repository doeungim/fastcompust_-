# Part 3: Density-based Clustering -----------------------------------------
install.packages("factoextra")
install.packages("dbscan")

library(factoextra)
library(dbscan)

data("multishapes")
df_multishapes <- multishapes[, 1:2]

set.seed(123)

# K-Means clustering & Visulization
KMC_multishapes <- kmeans(df_multishapes, 5, nstart = 25)
fviz_cluster(KMC_multishapes, df_multishapes, ellipse = TRUE, geom = "point")

# DBSCAN & Visualization 1
DBSCAN_multishapes_1 <- dbscan(df_multishapes, eps = 0.15, minPts = 5)
fviz_cluster(DBSCAN_multishapes_1, df_multishapes, ellipse = FALSE, geom = "point",
             show.clust.cent = FALSE)

# DBSCAN & Visualization 2
DBSCAN_multishapes_2 <- dbscan(df_multishapes, eps = 0.2, minPts = 7)
fviz_cluster(DBSCAN_multishapes_2, df_multishapes, ellipse = FALSE, geom = "point",
             show.clust.cent = FALSE)

# DBSCAN & Visualization 3
DBSCAN_multishapes_3 <- dbscan(df_multishapes, eps = 0.1, minPts = 3)
fviz_cluster(DBSCAN_multishapes_3, df_multishapes, ellipse = FALSE, geom = "point",
             show.clust.cent = FALSE)
