# Capstone Movielens Project Code
# Objective is to create best recommendation model to predict movie ratings by users
# which has RMSE less than 0.86490

# Installing required packages for the project
if(!require(plyr)) install.packages("plyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")
if(!require(lubridate)) install.packages("lubridate")
if(!require(recosystem)) install.packages("recosystem")
if(!require(RColorBrewer)) install.packages("RColorBrewer")

# Loading required libraries and data
library(plyr)
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(recosystem)
library(RColorBrewer)

# Movielens 10M dataset
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Since my system has predefined timeout as 60(see getOption("timeout")), downloading 10M Movielens dataset requires more time.
# So, I allocate timeout as 500 by setting the options to download 62.5 MB data
getOption("timeout")
options(timeout = 500)
# Downloading 10M Movielens dataset
dll <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dll)
# Extract ratings and movies information from downloaded file and integrate into *movielens* dataset.
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dll, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dll, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Exploring the overview of the whole dataset "movielens"
head(movielens) 
# It consists of 6 features namely userId, movieId, rating, timestamp(date and time of rating), title(with release year) and genres.

# size of the entire dataset
nrow(movielens) 

# number of discrete movies in the dataset
n_distinct(movielens$movieId) 

# number of discrete users in the dataset
n_distinct(movielens$userId) 

# Splitting the "movielens" dataset into edx and validation set
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
# Make sure userId & movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Bring back removed rows from validation to edx set 
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dll, ratings, movies, test_index, temp, movielens, removed) #Clear out unwanted files

# Overview of the edx dataset
head(edx) #initial 6 rows with 6 columns

# Glimpse of the edx dataset
glimpse(edx)

# Quick summary of the dataset
summary(edx)

# Distinct movies and users in edx set
n_distinct(edx$movieId)
n_distinct(edx$userId)

# Number of observations(rows and columns) in the edx set
nrow(edx) 
ncol(edx) 

# Ratings distribution
unique(edx$rating) #10 distinct ratings ranging from 0.5 to 5.0

# How many zeros were given as ratings in the edx dataset?
sum(edx$rating == 0)
# How many threes were given as ratings in the edx dataset?
sum(edx$rating == 3)

# Number of ratings for each rating score
edx %>% group_by(rating) %>% summarize(count = n())
# Histogram of the distribution of ratings
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25,bins = 10, color = "black") +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  geom_vline(xintercept = 3.5,color = "red",linetype = 2) +
  ggtitle("Distribution of the ratings with histogram")
# This shows that half-star ratings are less common than the whole full star ratings.

# Top 25 Highest Movie ratings
top25<- edx %>% group_by(title) %>%
  summarize(count = n()) %>%
  top_n(25,count) %>%
  arrange(desc(count))
top25
# Barplot of top 25 movie titles with highest number of ratings
top25 %>% ggplot(aes(reorder(title,count),count)) +
  geom_bar(stat='identity') +
  coord_flip(y=c(0, 40000)) + 
  xlab("") +
  geom_text(aes(label = count), hjust=-0.1, size=3) +
  ggtitle("Top 25 Movie titles with \n highest ratings")
# Barplot shows movie Pulp Fiction (1994) has highest movie rating of 31362.

# Most given ratings from most to least
edx %>% group_by(rating) %>% summarize(count = n()) %>% 
  arrange(desc(count))
# This shows rating 4 has given most and 0.5 as least

# Movie Ratings Distribution
edx %>%
  group_by(movieId) %>%
  summarize(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.25,bins = 10,color = "black", fill = "lightblue") +
  scale_x_log10() +
  ggtitle("Histogram of Movie ratings distribution")
# This shows that some movies have been rated very few times showing popular and non-popular movies.

# User Ratings Distribution
edx %>%
  group_by(userId) %>%
  summarize(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.25,bins = 10,color = "black", fill = "pink") +
  scale_x_log10() +
  ggtitle("Histogram of User ratings distribution")
# This shows not every user gives rating to all movies. Some are active while others not. This shows effect in the model.

# Exploration of Year Effect
# Extracting Year(release year) variable from title in the edx set
edx_year <- edx %>% mutate(year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))))
head(edx_year)
plot(table(edx_year$year),type = "h", xlab = "year",ylab = "number of ratings",main = "Movie release year distribution of ratings")
# Plot shows latest movies gets rated by many users than the oldest ones

year_avg <- edx_year %>% group_by(year) %>% summarize(avg = mean(rating))
year_avg
year_avg %>% ggplot(aes(year,avg)) + geom_point() + geom_smooth() + ggtitle("Average ratings over years")
# Plot shows newer movies have low average ratings over decades. Because they have less time to rate than the older movies

# Exploration of Time Range Effect
# Extracting the year when movie is rated by user(movie rated year) from timestamp variable
# Calculating Time Elapse(range) between movie release year and year rated by user
edx_year <- edx_year %>% mutate(year_rated = year(as_datetime(timestamp)),time_range = year_rated - year)
head(edx_year)
plot(table(edx_year$time_range),type = "h", xlab = "Time Range",ylab = "count")
# Time range plot shows that there is not large time range for highest rated movies. Time range of 1 is the most common among movies.

edx_year %>% group_by(time_range) %>% 
  ggplot(aes(time_range)) + geom_bar() + scale_x_binned() + ggtitle("Time range with number of ratings")
# This implies recent movies (within 25 years at the time of rating) have more ratings.

time_elapse_avg <- edx_year %>% group_by(time_range) %>% summarize(avg = mean(rating))
time_elapse_avg %>%
  ggplot(aes(time_range,avg)) +
  geom_point() + geom_smooth() +
  geom_hline(yintercept = 3.5,color = "orange",linetype = 2) +
  ggtitle("Average of time range with number of ratings")
# Plot shows time range greater than 20 has highest average number of ratings. This shows some effect in the model.

# Exploration of Time Effect
# Extract date from timestamp variable using round_date() function  and convert it into weekly format.
edx %>% mutate(date_week = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date_week) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(date_week, avg)) +
  geom_point() +
  geom_smooth() +
  xlab("Time effect by week") +
  ggtitle("Average rating of the Time effect in week")

# Exploration of Genre EFfect
# Extract number of movie ratings for each genre
gen_split <- edx %>% separate_rows(genres, sep = "\\|")
genre_split <- gen_split %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
genre_split
# Drama genre has highest number of movie ratings of 3910127.
# Whereas (no genres listed) contains no genre information has lowest ratings of 7.

# Barplot of genre effect with number of movie ratings
genre_split %>%
  ggplot(aes(reorder(genres,count),count)) +
  geom_bar(stat='identity',fill = "lightgreen",color = "black") +
  coord_flip() + 
  ylab("Number of movie ratings") +
  xlab("Genres") +
  ggtitle("Movie ratings for individual genre")

# Find number of movies for each genre
genre_names <- genre_split$genres
gen_mov <- sapply(genre_names, function(g){
  edx %>% 
    filter(str_detect(genres,g)) %>%
    group_by(movieId) %>%
    summarize(n = n())
})
m = c()
for (i in 1:40) { 
  m[i] = length(gen_mov[[i]])
}
count <- unique(m)
genre_movie <- data.frame(genre_names,count)
genre_movie %>% ggplot(aes(reorder(genre_names,count),count)) +
  geom_bar(stat='identity',fill = "lightblue",color = "black") +
  coord_flip() +
  ylab("Movies per genre") +
  xlab("Genres") +
  ggtitle("Number of movies for each genre")
# Plot shows Drama genre has more movies over 5000.

# Find how movie ratings per year in each genre differs?
genre_year <- edx_year %>% 
  separate_rows(genres, sep = "\\|") %>% 
  select(movieId, year, genres) %>% 
  group_by(year, genres) %>% 
  summarise(count = n()) %>% arrange(desc(year))
# There are only 12 colors in the palette.
# So, define the number of colors you want to increase the color choices for 20 genres.
# This can be done with the helpof the package RcolorBrewer.
nb.cols <- 20
mycolors <- colorRampPalette(brewer.pal(12, "Paired"))(nb.cols)
genre_year %>%
  ggplot(aes(x = year, y = count,fill = genres)) +
  geom_bar(stat = "identity",position = 'stack') + 
  scale_fill_manual(values = mycolors) + 
  ggtitle('Popularity per year by Genre')
# Plot shows there is not constant genre popularity over years. It varies for every year.

# Plot how genres affect movie ratings.
gen_split %>% group_by(genres) %>% summarise(avg = mean(rating)) %>% ggplot(aes(reorder(genres,avg),avg)) +
  geom_bar(stat='identity',fill = "lightgreen",color = "black") +
  coord_flip() + 
  ylab("Average number of ratings per genre") +
  xlab("Genres") +
  ggtitle("Average ratings per genre")
# Film-Noir genre has more average number of ratings compared to others.
# One can see, there is not much genre effect in the model from the plot.

# Plot genres distribution of ratings without splitting the data.
edx %>% group_by(genres)%>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 50000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  ggtitle("Error barplots of genres")
# Genre Horror has low average number of ratings compared to Horror|Thriller or Comedy.

# Lets build the model.
# Splitting edx dataset into training and test sets
set.seed(1,sample.kind = "Rounding")
y = edx$rating
edx_test_index <- createDataPartition(y, times = 1, p = 0.1, list = FALSE)
training_edx <- edx %>% slice(-edx_test_index)
test_temp <- edx %>% slice(edx_test_index)
# Make sure userId & movieId in test_edx set are also in train_edx set
test_edx <- test_temp %>%
  semi_join(training_edx, by = "movieId") %>%
  semi_join(training_edx, by = "userId")
# Bring back removed rows from test_edx to training_edx set
removed <- anti_join(test_temp, test_edx)
training_edx <- rbind(training_edx, removed)
rm(test_temp,removed) #clear unwanted variables

# Create a function to calculate Root Mean Square error(RMSE)
RMSE <- function(actual_value, predicted_value){
  sqrt(mean((actual_value - predicted_value)^2))
}

# Model 1: Simple average method
# Consider all movies have same ratings regardless of users by taking average of all ratings.
average <- mean(training_edx$rating)
average 
# Calculate RMSE
simple_model_rmse <- RMSE(test_edx$rating, average)
simple_model_rmse 
# Lets tabulate the results
RMSE_results <- tibble(Model = "Simple Average Model", rmse = simple_model_rmse)
RMSE_results
# From the above method, RMSE is of about 1.
# As instructed in this project, RMSE should be less than 0.86490.
# So different approaches should be included to further improve this model.

# Model 2: Movie Effect model
# Since different movies are rated differently, movie bias effect(b_i) is considered.
mu <- mean(training_edx$rating)
mu
# Calculate movie bias(b_i)
movie_average <- training_edx %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
# Plot the distribution of the ratings of the Movie effect
movie_average %>% ggplot(aes(b_i)) + geom_histogram(bins = 10, color = "black")
# How much prediction improves by adding b_i term to the model?
predicted_value <- join(x=test_edx,y=movie_average,by = "movieId", type = "left")
pred_calc <- predicted_value %>% mutate(pred = mu + b_i) %>% pull(pred)
Movie_model <- RMSE(test_edx$rating,pred_calc)
Movie_model 
# Tabulate the results
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Movie Model", rmse = Movie_model))
RMSE_results %>% knitr::kable()
# By modeling movie effects, there is an quite improvement of RMSE. Still can do better!

# Model 3: Movie and User Effect
# Since some users rate every movie while others are grouchy towards rating, user specific effect (b_u) is taken into consideration along with movie effect
# Plot the distribution of the ratings of User effect who have rated 100 or more movies
training_edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  qplot(b_u,geom = "histogram" , data = ., bins = 30 ,color = I("black"))
# Calculate User effect,b_u.
user_average <- merge(x=training_edx,y=movie_average,by = "movieId",all.x = TRUE) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
# How much prediction improves by adding b_u term along with b_i in the model?
predicted1<- merge(x=test_edx,y=movie_average,by = "movieId",all.x = TRUE)
predicted2 <- merge(x=predicted1,y=user_average,by = "userId",all.x = TRUE)
predicted_value1 <- predicted2 %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
Movie_User_Model <- RMSE(test_edx$rating,predicted_value1)
Movie_User_Model
# Tabulate the results
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Movie & User Effect Model", rmse = Movie_User_Model))
RMSE_results %>% knitr::kable()
# There is an quite improvement to RMSE all the way from 0.9429 to 0.8646 by modelling user specific effects along with movie effect.
# Therefore, Movie and User effects can be included in the model.

# Model 4: Movie Release Year Effect
# Since different movies are released in different years, there is variability across movies release year.
# To take that into account, year bias effect(b_y) is considered
# Variable title has movie name along with release year.
# Extract release year from title column using str_extract() for both training_edx and test_edx sets.
training_edx_1<- training_edx %>%
  mutate(year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))))
head(training_edx_1)
test_edx_1 <- test_edx %>%
  mutate(year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))))
head(test_edx_1)
# Calculate release year of movie bias(b_y)
year_average <- training_edx_1 %>%
  left_join(movie_average,by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))
# How much prediction improves by adding b_y term along with b_i and b_u in the model?
predicted_value2 <- test_edx_1 %>% left_join(movie_average, by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  left_join(year_average,by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_y) %>% pull(pred)
Movie_release_year_model <- RMSE(test_edx$rating,predicted_value2)
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Movie & User & Release Year Effect Model", rmse = Movie_release_year_model))
RMSE_results %>% knitr::kable() 
# Movie release year model reduces RMSE only by 0.0003%. Therefore it is not significant to consider it in main model.

# Model 5: Time Range Effect
# Time Range(the time range between the year movie released and the year movie is rated by user)
# Intermediate time describes how long the user takes to rate movie after being released in accordance with movies popularity.
# timestamp variable contains the year user gave rating to particular movie
# Extract when the movie was rated from timestamp variable using as_datetime() and year().
# Find timerange (Movie rated year by user - movie released year) for both training_edx and test_edx.
training_edx_1<- training_edx_1 %>% mutate(year_rated = year(as_datetime(timestamp)),time_range = year_rated - year) 
head(training_edx_1)
test_edx_1 <- test_edx_1 %>% mutate(year_rated = year(as_datetime(timestamp)),time_range = year_rated - year) 
head(test_edx_1)
# Calculate Time range effect(Rated year - release year), b_tr.
time_range_average <- training_edx_1 %>% 
  left_join(movie_average,by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  group_by(time_range) %>%
  summarize(b_tr = mean(rating - mu - b_i - b_u))
# How much prediction improves by adding b_tr term along with b_i and b_u?
predicted_value3 <- test_edx_1 %>% left_join(movie_average, by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  left_join(time_range_average,by = "time_range") %>%
  mutate(pred = mu + b_i + b_u + b_tr) %>% pull(pred)
Time_Range_Model <- RMSE(test_edx$rating,predicted_value3) 
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Movie & User & Time Range(Year rated - Released year) Effect Model", rmse = Time_Range_Model))
RMSE_results %>% knitr::kable() 
# Above model has very slightly improved RMSE value to 0.8642 compared to Model 3. 
# Therefore, this effect can be included in the model.

# Model 6: Time Effect
# timestamp variable contains date and time of rating given by user for a particular movie
# Extract date of a rating using as_datetime() from timestamp variable
# Extract week of a rating using round_date() for both training and test sets.
training_edx_1 <- mutate(training_edx_1, date = as_datetime(timestamp),week =  round_date(date, unit = "week"))
head(training_edx_1)
test_edx_1 <- mutate(test_edx_1, date = as_datetime(timestamp),week =  round_date(date, unit = "week"))
head(test_edx_1)
# Calculate time effect,b_t.
time_average <- training_edx_1 %>%
  left_join(movie_average,by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  group_by(week) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))
# How much prediction improves by adding b_t term along with b_i and b_u.
predicted_value4 <- test_edx_1 %>% 
  left_join(movie_average, by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  left_join(time_average,by = "week") %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  pull(pred)
Simple_Time_model <- RMSE(test_edx$rating,predicted_value4)
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Movie & User & Time Effect Model", rmse = Simple_Time_model))
RMSE_results %>% knitr::kable() 
# Time model does not reduce RMSE. Instead, it increases RMSE compared to Model 5 and Model 4.
# Therefore, it is not eligible to include in the model.

# Model 7: Genre Effect - Simply grouping genres
# Some genres have more ratings compare to others is described in the error plot of data exploration.
# So, lets check whether genres have effect in the model prediction.
genre_average <- training_edx %>%
  left_join(movie_average,by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))
# How much prediction improves by adding b_g term along with b_i and b_u in the model?
predicted_value5 <- test_edx %>% left_join(movie_average, by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  left_join(genre_average,by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
Simple_Genre_model <- RMSE(test_edx$rating,predicted_value5)
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Movie + User + Simple Genre Effect Model", rmse = Simple_Genre_model))
RMSE_results %>% knitr::kable()
# Above model is slightly better than Model 6.
# But, taking distinct genre combinations has apprximately over 800 genres in the dataset.
# This leads to overfitting of the model.
# So, this model cannot be considered in the model.

# Model 8: Genre Effect - Columnize genres.
# To avoid overfitting and overtraining(seperate genre by rows), lets put individual genre into each column.
# And calculate b_gk's for all genre columns with every possibilities(0 or 1)
# Remove "(no genres listed)" as it contains no genre information.
training_edx_2 <- training_edx_1 %>% separate_rows(genres,sep = "\\|") %>% mutate(option=1)
training_edx_3 <- training_edx_2 %>% spread(genres, option, fill=0) %>% select(-"(no genres listed)")
head(training_edx_3)
test_edx_2 <- test_edx_1 %>% separate_rows(genres,sep = "\\|") %>% mutate(op=1)
test_edx_3 <- test_edx_2 %>% spread(genres, op, fill=0) 
head(test_edx_3)
# How much prediction improves by adding b_gk term along with b_i and b_u in the model?
g_av <- training_edx_3 %>%
  left_join(movie_average,by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  group_by(Action,Adventure,Animation,Children,Comedy,Crime,Documentary,Drama,Fantasy,`Film-Noir`,Horror,IMAX,Musical,Mystery,Romance,`Sci-Fi`,Thriller,War,Western) %>% 
  summarize(b_gk = mean(rating - mu - b_i - b_u))
g_av
predicted_value6 <- test_edx_3 %>%
  left_join(movie_average, by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  left_join(g_av) %>%
  mutate(pred = mu + b_i + b_u + b_gk) %>%
  pull(pred)
Genre_model <- RMSE(test_edx$rating,predicted_value6)
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Movie + User + Genre Effect Model(Columnize genre format)", rmse = Genre_model))
RMSE_results %>% knitr::kable()
# Columnize Genre model produces RMSE same as above model.
# Since there is slight effect without overtraining and overfitting, it can be included along with Timerange model.

# Model 9: Movie, User, Time Range and Genre(Columnize) model
# Time range and Genres by column format slightly improves RMSE as seen in Model 5 & 8.
# So, add these effects along with user and movie effects. Find whether it improves the model further.
predicted_rating10 <- test_edx_3 %>%
  left_join(movie_average, by = "movieId") %>%
  left_join(user_average,by = "userId") %>%
  left_join(time_range_average,by = "time_range") %>%
  left_join(g_av,by = c("Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western")) %>%
  mutate(pred = mu + b_i + b_u + b_tr + b_gk) %>%
  pull(pred)
Movie_User_Timerange_genre_model <- RMSE(predicted_rating10, test_edx$rating)
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Movie,User,Timerange,Genre Effect Model", rmse = Movie_User_Timerange_genre_model))
RMSE_results %>% knitr::kable()
# On combining models, RMSE decreased to 0.8639419.

# Model 10: Regularisation - Movie Effect
# Some movies (good or worst) are rated by few users leading to uncertainty. 
# Regularization helps to penalize large estimates that are formed using small sample sizes.
# Instead of minimizing least squares equation, regularization adds penalty term to minimize the equation.
# Consider a window size to select lambda(tuning parameter) by cross validation.
lambdas <- seq(0, 10, 0.25)
mu <- mean(training_edx$rating)
# Calculate regularized movie effect,b_i and return RMSE for respective lambdas
movie_average <- training_edx %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
just_the_sum <- training_edx %>% 
  group_by(movieId) %>% 
  summarize(s_i = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_value7 <- join(test_edx,just_the_sum,by = "movieId",type = "left") %>% 
    mutate(b_i = s_i/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_value7, test_edx$rating))
})
# Plot different values of lambda with their RMSEs.
qplot(lambdas, rmses)  
# Find the optimal lambda which gives minimum RMSE value
lambda = lambdas[which.min(rmses)] 
lambda 
# Find the minimum RMSE value
Regularised_Movie_model <- min(rmses)
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Regularised Movie Effect Model", rmse = Regularised_Movie_model))
RMSE_results %>% knitr::kable()
# By penalized movie estimates, there is not much improvement in RMSE compared to least squares estimates(Model 2)

# Model 11: Regularised Movie & User Effect
# Regularization for estimating user effects along with movie effect. Since some users rate differently.
# Using cross validation, choose lambda.
lambdas <- seq(0, 10, 0.25)
# Calculate regularized movie and user effects,b_i and b_u and return RMSE for respective lambdas
rmses1 <- sapply(lambdas, function(l){
  mu <- mean(training_edx$rating)
  b_i <- training_edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- merge(x=training_edx,y=b_i,by = "movieId",all.x = TRUE) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted3<- merge(x=test_edx,y=b_i,by = "movieId",all.x = TRUE)
  predicted4 <- merge(x=predicted3,y=b_u,by = "userId",all.x = TRUE)
  predicted_value8 <- predicted4 %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
  return(RMSE(predicted_value8, test_edx$rating))
})
# Plot different values of lambda with their RMSEs.
qplot(lambdas, rmses1)  
# Find the optimal lambda which gives minimum RMSE value
lambda <- lambdas[which.min(rmses1)]
lambda
# Use this lambda to find the required RMSE
reg_b_i <- training_edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
reg_b_u <- merge(x=training_edx,y=reg_b_i,by = "movieId",all.x = TRUE) %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
predicted5<- merge(x=test_edx,y=reg_b_i,by = "movieId",all.x = TRUE)
predicted6 <- merge(x=predicted5,y=reg_b_u,by = "userId",all.x = TRUE)
predicted_value9 <- predicted6 %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
Regularised_Movie_User_model <- RMSE(predicted_value9, test_edx$rating)
# Tabulate the results
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Regularised Movie & User Effect Model", rmse = Regularised_Movie_User_model))
RMSE_results %>% knitr::kable()
#By penalized movie and user estimates,RMSE has slightly improved from least squares estimates of 0.864684 to 0.864136.

# Model 12: Matrix Factorization with parallel stochastic gradient descent
# Our previous models fails to describe the similarity pattern between users to users or movies to movies or users to movies. 
# Matrix factorization discovers these patterns based on the residuals of the movie and user effects.
# Convert the train and test sets into recosystem input format
set.seed(1, sample.kind = "Rounding")
train_data <-  with(training_edx, data_memory(user_index = userId, 
                                              item_index = movieId, 
                                              rating     = rating))
test_data <- with(test_edx, data_memory(user_index = userId, 
                                        item_index = movieId, 
                                        rating     = rating))
# Create the model object
r <-  recosystem::Reco()
# Select the best tuning parameters
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30), 
                                       lrate = c(0.1, 0.2),
                                       costp_l1 = 0, 
                                       costq_l1 = 0,
                                       nthread  = 1, niter = 10))
# Train the algorithm  
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))
# Predict the values 
y_hat <-  r$predict(test_data, out_memory())
Matrix_Factorization_model <- RMSE(test_edx$rating,y_hat)
# Tabulate the results
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Matrix Factorization Model", rmse = Matrix_Factorization_model))
RMSE_results %>% knitr::kable()
# It gives lower RMSE compared to other models. 
# So, it is chosen as a final model to train against validation set. 

# Model 13: Final Model using Matrix Factorization(against validation)
# Since Matrix Factorization with parallel stochastic gradient descent is the only model
# which gives minimum RMSE compared to other models,
# it is used as a final model against validation set for predicting movie rating.
# Convert the train and test sets into recosystem input format
set.seed(1, sample.kind = "Rounding")
train_data1 <-  with(edx, data_memory(user_index = userId, 
                                      item_index = movieId, 
                                      rating     = rating))
test_data1 <- with(validation, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
# Create the model object
r <-  recosystem::Reco()
# Select the best tuning parameters
opts <- r$tune(train_data1, opts = list(dim = c(10, 20, 30), 
                                        lrate = c(0.1, 0.2),
                                        costp_l1 = 0, 
                                        costq_l1 = 0,
                                        nthread  = 4, niter = 10))
# Train the algorithm  
r$train(train_data1, opts = c(opts$min, nthread = 4, niter = 20))
# Predict the values 
y_hat1 <-  r$predict(test_data1, out_memory())
Final_Matrix_Factorization_model <- RMSE(validation$rating,y_hat1) 
# Tabulate the results
RMSE_results <- bind_rows(RMSE_results, tibble(Model = "Final model using Matrix Factorization(using Validation)", rmse = Final_Matrix_Factorization_model))
RMSE_results %>% knitr::kable()
Final_Matrix_Factorization_model # is the rmse produced by final model using Matrix Factorization

# Conclusion
# Therefore, RMSE produced by final model is less than 0.86490 as similar to project instructions.
# By summing-up the prediction of different models, it is concluded that Matrix Factorization 
# with parallel stochastic gradient descent against validation set reduces the RMSE drastically
# Thus, Movielens project helps us to create better recommendation system algorithms to predict
# rating for a movie by user using 10M of the movielens dataset
# Since Matrix Factorization  makes predictions based on users past preferences to movies and finding a similarity pattern between them,
# it appears as a strong model for predicting movie ratings thereby reducing RMSE to a lower level compared to others
# For future works, one can use Restricted Boltzmann machines, neighbourhood models such as item- item and user-user approach,
# ensemble methods by combining different algorithms thereby exploiting the individual strength of each model to make predictions.