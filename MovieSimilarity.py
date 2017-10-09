val INPUT_FILENAME = "data/ratings.tsv"

/**
 * Read in the input and give each field a type and name.
 */
val ratings = Tsv(INPUT_FILENAME, ('user, 'movie, 'rating))

/**
 * Let's also keep track of the total number of people who rated each movie.
 */
val numRaters =
  ratings
    // Put the number of people who rated each movie into a field called "numRaters".    
    .groupBy('movie) { _.size }.rename('size -> 'numRaters)

// Merge `ratings` with `numRaters`, by joining on their movie fields.
val ratingsWithSize =
  ratings.joinWithSmaller('movie -> 'movie, numRaters)

val TRAIN_FILENAME = "ua.base"
val MOVIES_FILENAME = "u.item"

// Spark programs require a SparkContext to be initialized
val sc = new SparkContext(master, "MovieSimilarities")

// extract (userid, movieid, rating) from ratings data
val ratings = sc.textFile(TRAIN_FILENAME)
  .map(line => {
    val fields = line.split("\t")
    (fields(0).toInt, fields(1).toInt, fields(2).toInt)
})

// get num raters per movie, keyed on movie id
val numRatersPerMovie = ratings
  .groupBy(tup => tup._2)
  .map(grouped => (grouped._1, grouped._2.size))

// join ratings with num raters on movie id
val ratingsWithSize = ratings
  .groupBy(tup => tup._2)
  .join(numRatersPerMovie)
  .flatMap(joined => {
    joined._2._1.map(f => (f._1, f._2, f._3, joined._2._2))
})

val ratings2 =
  ratingsWithSize
    .rename(('user, 'movie, 'rating, 'numRaters) -> ('user2, 'movie2, 'rating2, 'numRaters2))

/**
 * Now find all pairs of co-rated movies (pairs of movies that a user has rated) by
 * joining the duplicate rating streams on their user fields, 
 */
val ratingPairs =
  ratingsWithSize
    .joinWithSmaller('user -> 'user2, ratings2)
    // De-dupe so that we don't calculate similarity of both (A, B) and (B, A).
    .filter('movie, 'movie2) { movies : (String, String) => movies._1 < movies._2 }
    .project('movie, 'rating, 'numRaters, 'movie2, 'rating2, 'numRaters2)
    
val ratings2 = ratingsWithSize.keyBy(tup => tup._1)

// join on userid and filter movie pairs such that we don't double-count and exclude self-pairs
val ratingPairs =
  ratingsWithSize
  .keyBy(tup => tup._1)
  .join(ratings2)
  .filter(f => f._2._1._2 < f._2._2._2)
  
val vectorCalcs =
  ratingPairs
    // Compute (x*y, x^2, y^2), which we need for dot products and norms.
    .map(('rating, 'rating2) -> ('ratingProd, 'ratingSq, 'rating2Sq)) {
      ratings : (Double, Double) =>
      (ratings._1 * ratings._2, math.pow(ratings._1, 2), math.pow(ratings._2, 2))
    }
    .groupBy('movie, 'movie2) { group =>
        group.size // length of each vector
        .sum('ratingProd -> 'dotProduct)
        .sum('rating -> 'ratingSum)
        .sum('rating2 -> 'rating2Sum)
        .sum('ratingSq -> 'ratingNormSq)
        .sum('rating2Sq -> 'rating2NormSq)
        .max('numRaters) // Just an easy way to make sure the numRaters field stays.
        .max('numRaters2)
        // All of these operations chain together like in a builder object.
    }
    
val vectorCalcs =
  ratingPairs
  .map(data => {
    val key = (data._2._1._2, data._2._2._2)
    val stats =
      (data._2._1._3 * data._2._2._3, // rating 1 * rating 2
        data._2._1._3,                // rating movie 1
        data._2._2._3,                // rating movie 2
        math.pow(data._2._1._3, 2),   // square of rating movie 1
        math.pow(data._2._2._3, 2),   // square of rating movie 2
        data._2._1._4,                // number of raters movie 1
        data._2._2._4)                // number of raters movie 2
    (key, stats)
  })
  .groupByKey()
  .map(data => {
    val key = data._1
    val vals = data._2
    val size = vals.size
    val dotProduct = vals.map(f => f._1).sum
    val ratingSum = vals.map(f => f._2).sum
    val rating2Sum = vals.map(f => f._3).sum
    val ratingSq = vals.map(f => f._4).sum
    val rating2Sq = vals.map(f => f._5).sum
    val numRaters = vals.map(f => f._6).max
    val numRaters2 = vals.map(f => f._7).max
    (key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2))
  })
  
val PRIOR_COUNT = 10
val PRIOR_CORRELATION = 0

val similarities =
  vectorCalcs
    .map(('size, 'dotProduct, 'ratingSum, 'rating2Sum, 'ratingNormSq, 'rating2NormSq, 'numRaters, 'numRaters2) ->
      ('correlation, 'regularizedCorrelation, 'cosineSimilarity, 'jaccardSimilarity)) {

      fields : (Double, Double, Double, Double, Double, Double, Double, Double) =>

      val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields

      val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
      val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, PRIOR_COUNT, PRIOR_CORRELATION)
      val cosSim = cosineSimilarity(dotProduct, math.sqrt(ratingNormSq), math.sqrt(rating2NormSq))
      val jaccard = jaccardSimilarity(size, numRaters, numRaters2)

      (corr, regCorr, cosSim, jaccard)
    }
    
  val PRIOR_COUNT = 10
val PRIOR_CORRELATION = 0

// compute similarity metrics for each movie pair
val similarities =
  vectorCalcs
  .map(fields => {

    val key = fields._1
    val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields._2

    val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
    val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum,
      ratingNormSq, rating2NormSq, PRIOR_COUNT, PRIOR_CORRELATION)
    val cosSim = cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))
    val jaccard = jaccardSimilarity(size, numRaters, numRaters2)

    (key, (corr, regCorr, cosSim, jaccard))
  })
