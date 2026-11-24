"""Fixed embedding tables shared by question-aware retriever tests."""

TABLE = {
    "who directed the movies starring Al Pacino": [1, 1, 0],
    "inverse of starred actors": [1, 0, 0],
    "directed by": [0, 1, 0],
    "starred actors": [0, 0, 1],
    "release year": [0, 0, 1],
    "inverse of directed by": [0, 0, 1],
    "inverse of release year": [0, 0, 1],
}
