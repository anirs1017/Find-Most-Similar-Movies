# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:47:31 2020

@author: sinha
"""

from pyspark import SparkContext, SparkConf
import sys
import math

def loadMovieNames():
    mov_names = {}
    with open('ml-100k/u.item') as f:
        for line in f:
            fields = line.split('|')
            movID = int(fields[0])
            movie = fields[1]
            mov_names[movID] = movie
    
    return mov_names

def findCosineSimilarity(ratingPairs):
    n_pairs = 0
    total_xSquared = 0
    total_ySquared = 0
    total_xy = 0
    
    for rating1, rating2 in ratingPairs:
        total_xSquared += rating1*rating1
        total_ySquared += rating2*rating2
        total_xy += rating1*rating2
        n_pairs += 1
    
    numerator = total_xy
    cosine = 0
    denominator = math.sqrt(total_xSquared * total_ySquared)
    
    if (denominator):
        cosine = numerator / float(denominator)
    
    return (cosine, n_pairs)
    
def preprocessData(line):
    fields = line.split()
    userID = int(fields[0])
    movID = int(fields[1])
    rating = float(fields[2])
    
    return (userID, (movID, rating))

def removeDuplicates(movie_pairs):
    pair = movie_pairs[1]
    (movie1, rating1) = pair[0]
    (movie2, rating2) = pair[1]
    
    return movie1 < movie2

def makePairs( movie_pairs ):
    pair = movie_pairs[1]
    (movie1, rating1) = pair[0]
    (movie2, rating2) = pair[1]
    
    return ((movie1, movie2), (rating1, rating2))
    
def main():
    
    # Defining the Master on multiple cores of the local system - local[*]
    conf = SparkConf().setMaster('local[*]').setAppName('FindSimilarMovies')
    sc = SparkContext(conf = conf)
    
    # Define the RDD
    movie_data = sc.textFile("file:///Spark course/ml-100k/u.data")
    
    print ('\n------------ LOADING MOVIE NAMES -----------------\n')
    movie_names = loadMovieNames()
    print (movie_names[50])
    '''
    PREPROCESSING AND REARRANGING THE DATA
    '''
    # Step 1 : Map the ratings to (key,value) pairs as userID: (movieID, rating)
    user_movie_rating = movie_data.map(preprocessData)
    
    # Step 2 : Combine all pairs of movies watched and rated by each user with (movie, rating) pair
    combined_userMovie = user_movie_rating.join(user_movie_rating)
    
    # Step 3: Current structure of data - userID: ((movie1, rating1), (movie2, rating2))
    # Remove all duplicate occurrences of movie pairs for the same user
    # Change structure of data to (movie1, movie2) : (rating1, rating2) as key:value pairs
    # to only maintain movie pairs and corresponding rating pairs to find similarity
    nonDuplicate_moviePairs = combined_userMovie.filter(removeDuplicates)
    
    # Step 4: Change the key structure to (movie1, movie2) for every pair
    moviePairs = nonDuplicate_moviePairs.map(makePairs)
    
    # Step 5: Accummulate all rating pairs (given by various users) for each movie pair and
    # compute similarity
    # Data structure after accummulation: (movie1, movie2) : ((rating1, rating2), (rating1, rating2), ...)
    accumulate_moviePairs = moviePairs.groupByKey()
    moviePair_similarities = accumulate_moviePairs.mapValues(findCosineSimilarity)
    
    # Cache this result for future
    moviePair_similarities = moviePair_similarities.cache()
    
    '''
    EXTRACTING SIMILARITIES FOR THE INPUT MOVIE PROVIDED
    '''
    if (len(sys.argv) > 1):
        threshold = 0.95
        coOccurence_threshold = 50
        
        input_movieID = int(sys.argv[1])
        
        # Filter all similar movies find beyond the threshold values
        filtered_similarities = moviePair_similarities.filter(lambda moviePair: (moviePair[0][0] == input_movieID or moviePair[0][1] == input_movieID) \
                                                              and moviePair[1][0] > threshold and moviePair[1][1] > coOccurence_threshold)
        
        # FLip the data and Sort the values with the scores and keep top 10 results
        results = filtered_similarities.map(lambda moviePair: (moviePair[1], moviePair[0])).sortByKey(ascending = False)
        top10 = results.take(10)
        
        print ("\nTop 10 similar movies for " + movie_names[input_movieID] + " :\n")
        for item in top10:
            (score, pair) = item
            
            similarMovie = pair[0]
            if (similarMovie == input_movieID):
                similarMovie = pair[1]
            
            print (movie_names[similarMovie] + "\tscore: " + str(score[0]) + "\tstrength: " + str(score[1]))
    
if __name__ == "__main__":
    main()    