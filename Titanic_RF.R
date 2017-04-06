# Titanic: Machine Learning from Disaster

setwd("C:/Users/Jessica/OneDrive/Estudos/Data Science/R/Kaggle Competitions/Titanic/")

############## Load packages #############################
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('Amelia') # missing data
library('caTools') # basic utility functions
library('randomForest') # classification

############## Load csv ############################
train <- read.csv('train.csv', stringsAsFactors = F)
test  <- read.csv('test.csv', stringsAsFactors = F)

full  <- bind_rows(train, test)

str(full)

############## Viewing the data ############################

# Number of survivors
ggplot(full, 
       aes(Survived, fill = factor(Survived,labels = c("No","Yes")))) + 
  geom_bar()  + ylab("Number") +
  labs(fill="Survived?")

# Amount by sex
ggplot(full,
       aes(Sex)) + 
  geom_bar(aes(fill = factor(Sex)), alpha = 0.5) +
  labs(fill = "Sex")

# Number of live passengers by sex
ggplot(full,
       aes(factor(Sex, labels = c("Fem","Masc")))) + 
  geom_bar(aes(fill = factor(Survived, labels = c("No", "Yes")))) +
  labs(fill = "Survived?") +
  xlab("") + ylab("Number")

# Quantity per Ticket Class
ggplot(full,
       aes(Pclass)) + 
  geom_bar(aes(fill = factor(Pclass, labels = c("1st","2st", "3st"))), alpha = 0.5) +
  labs(fill = "Ticket Class")

# Distribution of ages
ggplot(full,
       aes(Age)) + 
  geom_histogram(fill = 'red', bins = 20, alpha = 0.5)

# Number of brothers / sisters
ggplot(full,aes(SibSp)) + geom_bar(fill = 'green', alpha = 0.5)

# Passenger fares
ggplot(full,
       aes(Fare)) + 
  geom_histogram(fill = 'green', color = 'black', alpha = 0.5)

########### Exploratory data analysis ######################
# Testing for Missing Values
sum(is.na(full))

# Definir o volume de dados Missing
missmap(full, main = "Titanic Data - Mapa de Dados Missing", 
        col = c("red", "black"), legend = FALSE)

sum(is.na(full$Age))
sum(is.na(full$Fare))

# Richer Passengers tend to be olders
ggplot(full, aes(Pclass,Age)) +
  geom_boxplot(aes(group = Pclass, fill = factor(Pclass), 
                   apha = 0.4)) + 
  scale_y_continuous(breaks = seq(min(0), max(80), by = 2))

# Fill with the media data depending on the class
impute_age <- function(age,class){
  out <- age
  for (i in 1:length(age)){
    
    if (is.na(age[i])){
      
      if (class[i] == 1){
        out[i] <- 37
        
      }else if (class[i] == 2){
        out[i] <- 29
        
      }else{
        out[i] <- 24
      }
    }else{
      out[i]<-age[i]
    }
  }
  return(out)
}

fixed.ages <- impute_age(full$Age,full$Pclass)
full$Age <- fixed.ages


# No more Missing Age data
missmap(full, main = "Titanic Data - Mapa de Dados Missing", 
        col = c("red", "black"), legend = FALSE)

# Fare distribuition 
ggplot(full,
       aes(Fare)) + 
  geom_histogram(fill = 'blue', bins = 20, alpha = 0.5)

# Show row 1044
full[1044, ]

# Third class passenger, departed from Southampton 
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#12527c', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()

# Replace missing fare value with median fare for class/embarkment
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

# Boarding gate
# C = Cherbourg, Q = Queenstown, S = Southampton
# Verify that there are empty fields
ggplot(full,
       aes(Embarked)) + 
  geom_bar(
    aes(fill = factor(Embarked, labels =  c("", "Cherbourg", "Queenstown", "Southampton"))), alpha = 0.5) +
    labs(fill = "Port of Embarkation")

# IDs 62 and 830 are missing the embarkment
full[c(62, 830), 'Embarked']
full[c(62, 830), 'Fare']

# Visualize embarkment, passenger class, & median fare
ggplot(full, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='blue', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

# The average of IDs 62 and 830 hits right with the average paid by the
# Class C passengers
full$Embarked[c(62, 830)] <- 'C'
full[c(62, 830), 'Embarked']

# Make variables factors into factors
str(full)
factor_vars <- c('Sex','Embarked','Survived','Pclass')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))
str(full)

########### Training the Model ###########

# Data cleaning to remove irrelevant columns
full.clean <- select(full, -PassengerId, -Name, -Ticket, -Cabin)
full.clean.solution <- select(full,  -Name, -Ticket, -Cabin)

str(full.clean)

# Split the data back into a train set and a test set
train <- full.clean[1:891,]
test <- full.clean[892:1309,]
test.solution <- full.clean.solution[892:1309,]

# Set a random seed
set.seed(754)

# RandomForest model
model.rf <- randomForest(Survived ~ ., data = train)

# Confusion Matrix
model.rf$confusion

# summary
summary(model.rf)

# View the forest results.
print(model.rf) 

# Show model error
plot(model.rf, ylim=c(0,0.36))
legend('topright', colnames(model.rf$err.rate), col=1:3, fill=1:3)

# Importance of each predictor.
importance    <- importance(model.rf)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'white') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

########### Prediction ###########

# Predicting Accuracy
prediction <- predict(model.rf, test)

# Probability
prob <- predict(model.rf, newdata = test, type = 'prob')

# Create a DataFrame for the solution 
solution <- data.frame(PassengerID = test.solution$PassengerId, Survived = prediction)

# Write the solution to CSV
write.csv(solution, file = 'model_rf_titanic_Solution.csv', row.names = F)
