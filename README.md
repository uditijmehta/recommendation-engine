# recommendation-engine

A recommendation engine, also known as a recommender system, is a software that analyzes available data to make suggestions for something that a website user might be interested in, such as a book, a video or a job, among other possibilities. Amazon was one of the first sites to use a recommendation system. When the company was essentially an online book store, it began using software to suggest books the user might be interested in, based on data gathered about their previous activity, as well as the activity of other users who made similar choices. The recommendation system is usually categorized into two types: 1) Content based and 2) Collaborative filtering.

In a content-based recommender system, keywords are used to describe the items and a user profile is built to indicate the type of item this user likes. In other words, these algorithms try to recommend items that are similar to those that a user liked in the past (or is examining in the present).

On the other hand, collaborative filtering is the process of filtering for information or patterns using techniques involving collaboration among multiple agents, viewpoints, data sources, etc. In the newer, narrower sense, it is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating).

In this project, we’ve used content based filtering for recommending movies to a user.
The movies metadata (contains multiple files like movies metadata, credits, keywords, etc.) obtained from kaggle.
The dataset movies_metadata contains data of around 45000 movies, out of which we use around 10000 movies for our recommendation for processing purposes.
Simple Recommender:
First, we implemented a simple recommender, that will filter only the top movies based on the imdb ratings and votes given to a specific movie by users.
Content based recommender:
First we use only the data available in the metadata dataset. We choose the movie Batman Begins as a reference and show how the recommended movies change as we add more data to the system.
However this is very basic as it does not consider cast and crew members, director, genre, etc.
Then we add in other metadata (cast and crew)  fields to help improve recommendations.
In this project, we learnt about how content based filtering works by using it on the system which we created, i.e, the movie recommender system, which was able to recommend relevant movies as per the user input.
