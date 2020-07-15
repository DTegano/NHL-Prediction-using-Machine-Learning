# NHL Prediction using Machine Learning

<b> Project Background: </b> Using the 2017-2018 and 2018-2019 NHL Seasons as my training data, I will predict the 2019-2020 NHL games using a few different Machine Learning Models. I will treat this as a binary classification problem and predict two categories: win, or loss based on the outcome of the home team. My variables will include: Date of the game, Home Team, Away Team (date and team names will be removed for machine learning but will be present for exploratory analysis), Result (training only - reflects Home Team), Home Goals, Home Shots, Home PIM (Penalties in Minutes), Away Goals, Away PIM, Away Shots, Home Corsi (all situations), Away Corsi, Home Offensive Zone Start %, Away Offensive Zone Start %, Home Hits, Away Hits, Home Blocked Shots, Away Blocked Shots, Game Length (Regulation, Overtime, or Shootout), Empty_Netters (reflects empty net goals for the winning team), Home Save %, Away Save %, Home Shooting %, Away Shooting %, Home SPSV%/PDO, Away SPSV%/PDO, Home Goals Against, Away Goals Against, Home Differential, Away Differential, Home Wins, Away Wins, Home Shots Against, Away Shots Against, Home Points and Away Points. 

This is part 2 of my main overall project. In order to predict games, the NHL data needs to be structured in a specific way. If you're unfamiliar with this process or haven't read Part 1 of this project, please check out this link: https://github.com/DTegano/Web-Scraping-NHL-Data-for-Prediction-using-Machine-Learning

# Libraries 

Here are my libraries:

```
library(readxl) # importing xls & xlsx files
library(dplyr) # used for nesting/chaining
library(tidyr) # data cleaning
library(stringr) # Used to detect OT and Shootouts
library(writexl) # exporting to xlsx file
library(chron) # Date conversion to Time
library(stats) # Group Functional
library(plyr) # ddpylr for aggregation
```

# Re-Structuring the Data

I'll be the first to admit when I'm wrong about something. If you looked at part 1 of this project, you'll notice that most of my data was structured to use a cumulative sum - which meant that stats were aggregating for each team as the season went on. Unfortunately, it didn't take my long in my analysis to notice the problem with that. So before I jump into anything else, let me show you the issue I noticed:

``` 
> cor(ttrain$Home_Goals, ttrain$Home_BS)
[1] 0.9595307
> plot(ttrain$Home_Goals ~ ttrain$Home_BS)
```

<img src = "https://user-images.githubusercontent.com/39016197/87491348-240a1c80-c605-11ea-8040-f6be85cdcbf7.png" width = 470 height = 250>

The plot above shows the plot relationship between home goals and blocked shots. Notice anything? Typically, we wouldn't expect to see much of a relationship between these two variables - goals are more known to have a relationship with shots on goal. You can't score if you don't shoot the puck. However, since I used a cumulative sum for these stats, there is now a huge linear relationship between all of my predictor variables (since all of my stats are increasing with each other for every game). In the data science world, this is known as collinearity -  and a high amount of that in this case.

To fix this problem, I had to go back to my raw data and apply the same scripting methods that I applied the first time around. However, this time I used a cumulative mean for all of my variables. With the new formatted data, we finally get the results that we would expect to see between these two variables: 

```
> cor(dtrain$Home_Goals, dtrain$Home_BS)
[1] 0.06437274
> plot(dtrain$Home_Goals ~ dtrain$Home_BS)
```

<img src = "https://user-images.githubusercontent.com/39016197/87491760-3173d680-c606-11ea-8902-cd9be499187a.png" width = 470 height = 250>

A correlation value of 0.064 makes much more sense than 0.959 between goals and blocked shots. We're finally ready to move on to the analysis. 

# Importing the Data

As we can see from the above, my machine learning data set will be imported into R as "Dtrain" and "Dtest". However, I also think there is a lot of value of also looking at the raw data - which is the data that shows each game's stats and results before they were manipulated into an aggregated data set. For this raw data, I'll import both my training and test set and combine both sets into one data frame called "dt"  (I've always gotten into the habit of using dt for 'data table' and because my initials happen to be dt):

```
dtrain = read_excel("Training Final CM.xlsx", col_names = TRUE)
dtest = read_excel("Test Final CM.xlsx", col_names = TRUE)

raw_2018 = read_excel("2017-2018 Training Base.xlsx", col_names = TRUE)
raw_2019 = read_excel("2018-2019 Training Base.xlsx", col_names = TRUE)
raw_2020 = read_excel("2019-2020 Test Base.xlsx", col_names = TRUE)

dt = rbind(raw_2018, raw_2019, raw_2020)
```

Once I have combined my training set with my entire test set, I'll create new set and remove the Results column from my test set - as this will be the variable we will be predicting. I want to keep my Results variable in my original test set so that I can compare my machine learning results later on. I'll also remove the date and team names from my training and test set - as I will not need these variables when I make my predicitons. Finally, I need to make sure that my Prediction variable is set as a factor, as well as the the result, team names, and game length for the raw data analysis:

```
test = dtest[,-4]


dtrain = dtrain[, -c(1:3)]
dtest = dtest[, -c(1:3)]

dtrain$Result = as.factor(dtrain$Result)
dtest$Result = as.factor(dtest$Result)

dt$Result = as.factor(dt$Result)
dt$Home = as.factor(dt$Home)
dt$Away = as.factor(dt$Away)
dt$Game_Length = as.factor(dt$Game_Length)
```



# Analysis - Raw Data 

Before conducting analysis, it's always a good idea to view the data and get familiar with its' contents before. Since I'm using R, it's also a good idea to attach your data set so that variables are much easier to call. Once this has been completed, one of the first commands you should run with a new data frame is the structure and summary functions. Since my teams and results loaded in as characters, I'll first change these variables to factors. It's also very important to deal with any missing values. I did already fix all of my missing values in Part 1, but it still can't hurt to check just in case:

```
 table(is.na(dt))

 FALSE 
130464  
```


Once the change has been made, here are the corresponding strucutre and summary commands:

``` 
> str(dt)

Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	3624 obs. of  36 variables:
 $ Date         : POSIXct, format: "2017-10-04" ...
 $ Home         : Factor w/ 31 levels "Anaheim Ducks",..: 12 23 24 31 1 3 14 11 4 20 ...
 $ Away         : Factor w/ 31 levels "Anaheim Ducks",..: 5 25 22 27 2 17 22 15 16 8 ...
 $ Result       : Factor w/ 2 levels "L","W": 2 1 1 1 2 2 2 2 1 1 ...
 $ Home_G       : num  3 4 3 2 5 4 2 4 2 2 ...
 $ Home_PIM     : num  7 8 10 8 18 19 10 8 8 12 ...
 $ Home_S       : num  45 33 35 37 41 32 27 31 45 39 ...
 $ Away_G       : num  0 5 5 7 4 3 0 2 2 4 ...
 $ Away_PIM     : num  9 10 12 16 8 15 10 10 10 16 ...
 $ Away_S       : num  27 34 31 31 30 29 35 39 40 26 ...
 $ Home_Corsi   : num  58.4 56.2 51.6 57.3 52.7 54 47.1 47.4 52.3 61.1 ...
 $ Home_OFS     : num  61.3 55.1 45.2 61.1 60.9 53.2 57.8 49 57.1 65.3 ...
 $ Home_H       : num  31 27 14 18 27 23 21 13 20 23 ...
 $ Home_BS      : num  18 10 17 14 18 7 11 11 15 9 ...
 $ Away_Corsi   : num  41.6 43.8 48.4 42.7 47.3 46 52.9 52.6 47.7 38.9 ...
 $ Away_OFS     : num  39.7 46.7 57.5 43 41.4 50.9 45 54.8 45.2 37.9 ...
 $ Away_H       : num  29 36 21 16 18 25 19 19 17 21 ...
 $ Away_BS      : num  17 19 17 24 11 10 13 10 8 15 ...
 $ Game_Length  : Factor w/ 3 levels "O","R","S": 2 1 2 2 2 2 2 2 3 2 ...
 $ Empty_Netters: num  1 0 1 0 0 1 0 0 0 1 ...
 $ Home_SV      : num  1 0.853 0.867 0.774 0.867 ...
 $ Away_SV      : num  0.955 0.879 0.914 0.946 0.878 ...
 $ Home_SH      : num  0.0667 0.1212 0.0857 0.0541 0.122 ...
 $ Away_SH      : num  0 0.147 0.161 0.226 0.133 ...
 $ Home_PDO     : num  1.067 0.974 0.952 0.828 0.989 ...
 $ Away_PDO     : num  0.955 1.026 1.076 1.172 1.011 ...
 $ Home_GA      : num  0 5 5 7 4 3 0 2 2 4 ...
 $ Away_GA      : num  3 4 3 2 5 4 2 4 2 2 ...
 $ Home_DIF     : num  3 -1 -2 -5 1 1 2 2 0 -2 ...
 $ Away_DIF     : num  -3 1 2 5 -1 -1 -2 -2 0 2 ...
 $ Home_W       : num  1 0 0 0 1 1 1 1 0 0 ...
 $ Away_W       : num  0 1 1 1 0 0 0 0 1 1 ...
 $ Home_SA      : num  27 34 31 31 30 29 35 39 40 26 ...
 $ Away_SA      : num  45 33 35 37 41 32 27 31 45 39 ...
 $ Home_P       : num  2 1 0 0 2 2 2 2 1 0 ...
 $ Away_P       : num  0 2 2 2 0 0 0 0 2 2 ...
 
> summary(dt)

      Date                                       Home     
 Min.   :2017-10-04 00:00:00   Detroit Red Wings   : 119  
 1st Qu.:2018-02-18 00:00:00   Montreal Canadiens  : 119  
 Median :2018-12-20 12:00:00   Ottawa Senators     : 119  
 Mean   :2018-12-13 11:09:08   Vegas Golden Knights: 119  
 3rd Qu.:2019-10-27 06:00:00   Winnipeg Jets       : 119  
 Max.   :2020-03-11 00:00:00   Anaheim Ducks       : 118  
                               (Other)             :2911  
                 Away      Result       Home_G          Home_PIM     
 Arizona Coyotes   : 119   L:1649   Min.   : 0.000   Min.   : 0.000  
 Calgary Flames    : 119   W:1975   1st Qu.: 2.000   1st Qu.: 4.000  
 Colorado Avalanche: 119            Median : 3.000   Median : 6.000  
 Edmonton Oilers   : 119            Mean   : 3.105   Mean   : 8.132  
 Chicago Blackhawks: 118            3rd Qu.: 4.000   3rd Qu.:10.000  
 Los Angeles Kings : 118            Max.   :10.000   Max.   :64.000  
 (Other)           :2912                                             
     Home_S          Away_G         Away_PIM          Away_S     
 Min.   :13.00   Min.   :0.000   Min.   : 0.000   Min.   :13.00  
 1st Qu.:28.00   1st Qu.:2.000   1st Qu.: 4.000   1st Qu.:26.00  
 Median :32.00   Median :3.000   Median : 8.000   Median :30.00  
 Mean   :32.27   Mean   :2.821   Mean   : 8.772   Mean   :30.96  
 3rd Qu.:37.00   3rd Qu.:4.000   3rd Qu.:10.000   3rd Qu.:35.00  
 Max.   :58.00   Max.   :9.000   Max.   :84.000   Max.   :60.00  
                                                                 
   Home_Corsi       Home_OFS         Home_H         Home_BS     
 Min.   :25.90   Min.   :18.00   Min.   : 4.00   Min.   : 2.00  
 1st Qu.:46.20   1st Qu.:44.80   1st Qu.:17.00   1st Qu.:11.00  
 Median :51.30   Median :52.10   Median :22.00   Median :14.00  
 Mean   :51.18   Mean   :51.81   Mean   :22.35   Mean   :13.99  
 3rd Qu.:56.42   3rd Qu.:59.00   3rd Qu.:27.00   3rd Qu.:17.00  
 Max.   :74.80   Max.   :86.50   Max.   :54.00   Max.   :33.00  
                                                                
   Away_Corsi       Away_OFS        Away_H         Away_BS     
 Min.   :25.20   Min.   :13.9   Min.   : 2.00   Min.   : 2.00  
 1st Qu.:43.58   1st Qu.:42.6   1st Qu.:16.00   1st Qu.:11.00  
 Median :48.70   Median :49.4   Median :21.00   Median :14.00  
 Mean   :48.82   Mean   :49.8   Mean   :21.41   Mean   :14.52  
 3rd Qu.:53.80   3rd Qu.:56.9   3rd Qu.:26.00   3rd Qu.:18.00  
 Max.   :74.10   Max.   :82.8   Max.   :67.00   Max.   :33.00  
                                                               
 Game_Length Empty_Netters       Home_SV          Away_SV      
 O: 541      Min.   :0.0000   Min.   :0.6667   Min.   :0.6364  
 R:2807      1st Qu.:0.0000   1st Qu.:0.8750   1st Qu.:0.8743  
 S: 276      Median :0.0000   Median :0.9167   Median :0.9118  
             Mean   :0.3005   Mean   :0.9097   Mean   :0.9056  
             3rd Qu.:1.0000   3rd Qu.:0.9487   3rd Qu.:0.9444  
             Max.   :2.0000   Max.   :1.0000   Max.   :1.0000  
                                                               
    Home_SH           Away_SH           Home_PDO         Away_PDO     
 Min.   :0.00000   Min.   :0.00000   Min.   :0.6937   Min.   :0.6667  
 1st Qu.:0.05556   1st Qu.:0.05128   1st Qu.:0.9507   1st Qu.:0.9434  
 Median :0.09375   Median :0.08696   Median :1.0095   Median :0.9951  
 Mean   :0.09929   Mean   :0.09422   Mean   :1.0090   Mean   :0.9998  
 3rd Qu.:0.13636   3rd Qu.:0.13043   3rd Qu.:1.0685   3rd Qu.:1.0587  
 Max.   :0.41176   Max.   :0.33333   Max.   :1.3403   Max.   :1.3063  
                                                                      
    Home_GA         Away_GA          Home_DIF          Away_DIF      
 Min.   :0.000   Min.   : 0.000   Min.   :-8.0000   Min.   :-9.0000  
 1st Qu.:2.000   1st Qu.: 2.000   1st Qu.:-1.0000   1st Qu.:-2.0000  
 Median :3.000   Median : 3.000   Median : 1.0000   Median :-1.0000  
 Mean   :2.821   Mean   : 3.105   Mean   : 0.2837   Mean   :-0.2837  
 3rd Qu.:4.000   3rd Qu.: 4.000   3rd Qu.: 2.0000   3rd Qu.: 1.0000  
 Max.   :9.000   Max.   :10.000   Max.   : 9.0000   Max.   : 8.0000  
                                                                     
     Home_W          Away_W         Home_SA         Away_SA     
 Min.   :0.000   Min.   :0.000   Min.   :13.00   Min.   :13.00  
 1st Qu.:0.000   1st Qu.:0.000   1st Qu.:26.00   1st Qu.:28.00  
 Median :1.000   Median :0.000   Median :30.00   Median :32.00  
 Mean   :0.545   Mean   :0.455   Mean   :30.96   Mean   :32.27  
 3rd Qu.:1.000   3rd Qu.:1.000   3rd Qu.:35.00   3rd Qu.:37.00  
 Max.   :1.000   Max.   :1.000   Max.   :60.00   Max.   :58.00  
                                                                
     Home_P          Away_P     
 Min.   :0.000   Min.   :0.000  
 1st Qu.:0.000   1st Qu.:0.000  
 Median :2.000   Median :1.000  
 Mean   :1.203   Mean   :1.022  
 3rd Qu.:2.000   3rd Qu.:2.000  
 Max.   :2.000   Max.   :2.000  
          

```

As we can see....

