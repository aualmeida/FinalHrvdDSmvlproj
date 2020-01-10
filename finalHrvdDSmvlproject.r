#Installing if necessary and loading the required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(foreach)) install.packages("foreach", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(pls)) install.packages("pls", repos = "http://cran.us.r-project.org")

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

######################################################################################################

# View structure of the dataset
str(edx)

#View summary of the dataset
summary(edx)

# View first 5 rows of the dataset

head(edx,5)

# identify unique users and movies
edx %>% 
  summarize(users = n_distinct(userId),
            movies = n_distinct(movieId))

# plot distribution of ratings by movieId

edx %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, fill = "blue", color = "black") +
  scale_x_log10()+
  ggtitle("Distribution or ratings by movieID")

# plot distribution of ratings by userId

edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, fill = "green", color = "red") +
  scale_x_log10()+
  ggtitle("Distribution or ratings by userID")

# Using a bar plot to visualise rating distribution
edx %>% group_by(rating) %>% 
  ggplot(aes(rating)) + 
  geom_bar(fill = "yellow", color = "red") +
  ggtitle("Distribution or ratings")

# Create funtion that will calculate RMSE
mvl_RMSE <- function(true_ratings,predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2))
}

# Set seed and Create a subset
set.seed(10, sample.kind = "Rounding")
edxsub <- edx %>% sample_n(50000)

# Create data partition
index <- createDataPartition(edxsub$rating,times = 1, p = 0.8, list = FALSE)

# Use index to create train and test objects
mvltrain <- edxsub[index,]
mvltest <- edxsub[-index,]
mvltest <- mvltest %>% 
  semi_join(mvltrain, by = "movieId") %>%
  semi_join(mvltrain, by = "userId")

# Create the control and grid that the training model can use as inputs

control <- trainControl(method = "cv",number = 10, p=0.1)
grid <- data.frame(ncomp = c(1,2))

# Train the machine learning model on the training subset of the edx dataset
plst <- train(rating~movieId+userId, data = mvltrain,
              method  = "pls",
              trControl = control,
              tuneGrid = grid,
              preProcess = "scale",
              allowParallel = TRUE)

# Apply the trained model on the subset of test subset of edx
yhatplst <- predict(plst,mvltest)

# Calculate RMSE using the function provided
mvl_RMSE(mvltest$rating,yhatplst)

# Remove memory objects no longer needed
rm(edxsub,mvltest,mvltrain,plst,yhatplst,control,grid,index)

# The recosystem package applying matrix factorisation
# Step 1; SUbset edx and validation retain only the required 3 fields, rename and create matrix

edx1 <-  edx %>% select("userId","movieId","rating")
valid1 <- validation %>% select("userId","movieId","rating")

# Rename columns in the newly subsetted files
names(edx1) <- c("user", "item", "rating")
names(valid1) <- c("user", "item", "rating")

# Change the files to a matrix
edx1 <- as.matrix(edx1)
valid1 <- as.matrix(valid1)

# Step 2; Write the matrix files to tables on disk
write.table(edx1, file = "train.txt", sep = " ", row.names = FALSE, col.names = FALSE)
write.table(valid1, file = "test.txt", sep = " ", row.names = FALSE, col.names = FALSE)

# Step 3; Use data_file to create a data source object that links the written file

dir <- getwd()
train <- file.path(dir,"train.txt")
test <- file.path(dir,"test.txt")

# Use set seed before running data_file
set.seed(10, sample.kind = "Rounding")
train_1 <- data_file(train)
test_1 <- data_file(test)

# Step 4; Create the reco recommender object and applying tuning parameters to the model

r <- Reco()

# Apply tuning parameters

tparam <- r$tune(train_1, opts = list(dim = c(10, 20, 30), 
                                      lrate = c(0.1, 0.2),
                                      costp_l1 = 0,
                                      costq_l1 = 0,
                                      nthread = 3,
                                      niter = 10))
tparam

# Step 5; Train the recommender model on the train file and make predictions using test

r$train(train_1, opts = c(tparam$min, nthread = 5, niter = 20))

preds = tempfile()

r$predict(test_1, out_file(preds)) 

# Step 6; Assign the actual and predicted ratings to objects and calculate RMSE

actualrat <- read.table("test.txt", header = FALSE, sep = " ")$V3
predrat <- scan(preds)

reco_RMSE <- mvl_RMSE(actualrat, predrat)
reco_RMSE
















