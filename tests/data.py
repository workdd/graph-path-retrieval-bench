"""Shared public movie-trivia test graph."""

MOVIE_TRIPLES = [
    ("Heat", "directed_by", "Michael Mann"),
    ("Heat", "starred_actors", "Al Pacino"),
    ("Heat", "starred_actors", "Robert De Niro"),
    ("Heat", "release_year", "1995"),
    ("The Irishman", "starred_actors", "Al Pacino"),
    ("The Irishman", "directed_by", "Martin Scorsese"),
    ("Collateral", "directed_by", "Michael Mann"),
    ("Collateral", "release_year", "2004"),
]
# Question: "who directed the movies starring Al Pacino"
# Start Al Pacino, answers {Michael Mann, Martin Scorsese}
# (2 hops: inverse of starred_actors -> directed_by)
