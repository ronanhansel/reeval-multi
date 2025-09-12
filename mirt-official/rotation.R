require(arrow)
require(GPArotation)
a <- as.matrix(read_feather("/Users/ronan/Developer/reeval-multi/mirt-official/a_matrix.feather"))

# rotate_sparse.R

# Run simplimax rotation (sparse)
rot <- GPFoblq(a, method = "simplimax")

# Extract rotated loadings
rot_loadings <- rot$loadings

# Save rotated loadings to CSV
write.csv(rot_loadings, file = "a_matrix_rotated.csv")
