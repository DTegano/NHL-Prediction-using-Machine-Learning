# NHL Prediction using Machine Learning

<b> Project Background: </b> Using the 2017-2018 and 2018-2019 seasons as my training data*, I will predict the 2019-2020 NHL games using a few different Machine Learning Models. I will treat this as a binary classification problem and predict two categories: win, or loss based on the outcome of the home team. My variables will include: Date of the game, Home Team, Away Team (date and team names will be removed for machine learning but will be present for exploratory analysis), Result (training only - reflects Home Team), Home Goals, Home Shots, Home PIM (Penalties in Minutes), Away Goals, Away PIM, Away Shots, Home Corsi (all situations), Away Corsi, Home Offensive Zone Start %, Away Offensive Zone Start %, Home Hits, Away Hits, Home Blocked Shots, Away Blocked Shots, Game Length (Regulation, Overtime, or Shootout), Empty_Netters (reflects empty net goals for the winning team), Home Save %, Away Save %, Home Shooting %, Away Shooting %, Home SPSV%/PDO, Away SPSV%/PDO, Home Differential, Away Differential, Home Wins, Away Wins, Home Shots Against, Away Shots Against, Home Points, Away Points, and all previous stats listed as 'against' (such as home goals against, away goals against, home shots against, away shots against, etc.).

This is part 2 of my main overall project. In order to predict games, I believe the NHL data needs to be structured in a specific way. If you haven't read Part 1 of this project, feel free to check out this link: https://github.com/DTegano/Web-Scraping-NHL-Data-for-Prediction-using-Machine-Learning.

<b> * Editor's Note: </b> Towards the end of my project when optimizing my model, I'll note that I added the 2015-2016 and 2016-2017 seasons as additional training data.

# Library Packages 

Here are the different library packages used for this project:

```
library(readxl) # importing xls & xlsx files
library(dplyr) # used for nesting/chaining
library(tidyr) # data cleaning
library(stringr) # Used to detect OT and Shootouts
library(writexl) # exporting to xlsx file
library(chron) # Date conversion to Time
library(stats) # Group Functional
library(plyr) # ddpylr for aggregation
library(caret) # Confusion Matrix
library(gmodels) #cross table
library(ggplot2) # Model plots
library(cowplot) 
library(tidyverse)
library(broom)
library(kernlab) #used for SVM
library(neuralnet) #used for ANN
library(keras) # used for more compex neural networks
library(tensorflow)
library(deepviz)
library(magrittr)
```

The main packages used for the analysis portion was the caret, stats, kernlab, keras, and tensorflow packages. The confusion matrix within the caret package is extremely handy for when you're comparing a model's prediction results. Kernlab was used for the Support Vector Machine and the Keras/Tensorflow packages were used to build a more complex neural network than the models offered in the neuralnet package. I highly recommend the switch for anyone with an interest in building neural networks using R.

# Re-Structuring the Data

I'll be the first to admit when I'm wrong about something. If you looked at part 1 of this project, you'll notice that most of my data was structured to use a cumulative sum - which meant that stats were aggregating for each team as the season went on. Unfortunately, it didn't take me long in my analysis to notice the problem with that. Before I jump into anything else, let me show you the issue I noticed:

``` 
> cor(ttrain$Home_Goals, ttrain$Home_BS)
[1] 0.9595307
> plot(ttrain$Home_Goals ~ ttrain$Home_BS)
```

<img src = "https://user-images.githubusercontent.com/39016197/87491348-240a1c80-c605-11ea-8040-f6be85cdcbf7.png" width = 470 height = 250>

The plot above shows the plot relationship between home goals and blocked shots. Notice anything? Typically, we wouldn't expect to see much of a relationship between these two variables - goals are more known to have a relationship with shots on goal. You can't score if you don't shoot the puck. Blocked shots is a statistic used on the defensive side of the ice – we would almost expect to see a negative relationship since more blocked shots can indicate that a team is playing more in their defensive end than the offensive zone. However, since I used a cumulative sum for these stats, there is now a huge linear relationship between all of my predictor variables (since all of my stats are increasing with each other for every game). In the data science world, this is known as collinearity and there appears to be high amount in this case.

To fix this problem, I had to go back to my raw data and apply the same scripting methods that I applied the first time around. However, this time I used a cumulative mean for all of my variables. With the change, this alters how my data looks. For example, if predicting game 4 of the season for the Colorado Avalanche, let’s say they previously scored 3,2, and 4 goals. Instead of using a goal value of 9 goals (cumulative sum), I would now predict the result of game 4 using a value of 3 for the goal column – since that is their goal average prior to game 4. This method not only captures if a high-scoring team is more likely to win or not, but this also makes my data more scalable for when I use machine learning.

With the new formatted data, I can finally get the results that we would expect to see between these two variables:


```
> cor(dtrain$Home_Goals, dtrain$Home_BS)
[1] 0.06437274
> plot(dtrain$Home_Goals ~ dtrain$Home_BS)
```

<img src = "https://user-images.githubusercontent.com/39016197/87491760-3173d680-c606-11ea-8902-cd9be499187a.png" width = 470 height = 250>

A correlation value of 0.064 makes much more sense than the 0.959 value between goals and blocked shots. With that corrected, I can move on to the next step.

# Importing the Data

As we can see from the above, my machine learning data set will be imported into R as "dtrain" and "dtest". However, I think there is a lot of value to first look at the raw data - which is the data that shows each game's stats and results before they were manipulated into an aggregated data set. For this raw data, I'll import both my training and test set and combine both sets into one data frame called "dt"  (I've always gotten into the habit of using dt for 'data table' and because my initials happen to be dt):

```
dtrain = read_excel("Training Final CM.xlsx", col_names = TRUE)
dtest = read_excel("Test Final CM.xlsx", col_names = TRUE)

raw_2018 = read_excel("2017-2018 Training Base.xlsx", col_names = TRUE)
raw_2019 = read_excel("2018-2019 Training Base.xlsx", col_names = TRUE)
raw_2020 = read_excel("2019-2020 Test Base.xlsx", col_names = TRUE)

dt = rbind(raw_2018, raw_2019, raw_2020)
```

Once I have created my combined data set for analysis purposes, I'll create a new data set and remove the Results column from my test set - as this will be the variable we will be predicting. This will be necessary since I want to keep my Results variable in my original test set so that I can compare my machine learning results later on while still being able to test my model without the answer sheet. I will also remove the date and team names from my training and test set - as I will not need these variables when I make my predictions. I will, however, keep these variables in my combined data set for the analysis. Since characters won't help me in this kind of analysis, I'll change my non-numeric variables (result, team names, and game length) to a factor:

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

Before conducting analysis, it's always a good idea to view the data and get familiar with its' contents. Since I'm using R, it's also a good idea to attach your data set so that variables are much easier to call. Once this has been completed, one of the first commands you should run with a new data frame is the structure and summary functions. It's also very important to deal with any missing values. I did already fix all of my missing values in Part 1, but it still can't hurt to check just in case:

```
 table(is.na(dt))

 FALSE 
130464  
```


Once your data is clear of any missing values, here are the corresponding structure and summary commands:

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

Some interesting observations/notes from the 3 seasons worth of data: <p><p/>
1. Not all of the teams played the same amount of home and away games due to the quarantine stoppage.
2. The highest amount of goals scored in a NHL game over the last three seasons are 10 (home team) and 9 (away team)
3. The highest corsi % we have seen in any game is 74.8%, while the highest amount of shooting % in any game is a whopping 41%.
4. 87 Penalty minutes and 67 hits is the most we've seen in any game over the last 3 years.
5. 15% of NHL games end in Overtime and almost 8% of games go to a Shootout.
6. Home team advantage? Not in the NHL - only 54.5% of wins belong to a home team.

Now, I'll look at a few different variables to see how much of an impact they have on the outcome when the variable is maximized and minimized. The trend that I'm expecting here (assuming a positive relationship) is that when the variable is maximized, we'd expect to see wins and when the variable is minimized, we'd expect losses. Take goals for example - the teams that scored 10 and 9 goals won their games, and the teams that scored 0 goals lost <u> almost </u> every game (I believe there were 1 or 2 shootout victories at 0-0, but I think you get the point I'm trying to make).

For the following commands, I set up a quick function to return the name of the team, date, and result of the max-min value:

```
home_grab = function(x) {
    gameinfo = list(paste(dt$Home[x]), paste(dt$Date[x]), paste(dt$Result[x]))
    return(gameinfo)
}


away_grab = function(x) {
    gameinfo = list(paste(dt$Away[x]), paste(dt$Date[x]), paste(dt$Result[x]))
    return(gameinfo)
}
```

<b>Shots</b>

```
> summary(df$Home_S)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  13.00   28.00   32.00   32.27   37.00   58.00 
> home_grab(df$Home_S==58) 
[[1]]
[1] "Washington Capitals"

[[2]]
[1] "2019-03-20"

[[3]]
[1] "L"

> summary(df$Away_S)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  13.00   26.00   30.00   30.96   35.00   60.00 
> away_grab(df$Away_S==60)
[[1]]
[1] "Carolina Hurricanes"

[[2]]
[1] "2017-11-02"

[[3]]
[1] "W"

> home_grab(df$Home_S==13) 
[[1]]
[1] "Anaheim Ducks"      "New York Islanders"

[[2]]
[1] "2019-01-23" "2019-03-19"

[[3]]
[1] "L" "L"

> away_grab(df$Away_S==13) 
[[1]]
[1] "Tampa Bay Lightning"

[[2]]
[1] "2019-10-06"

[[3]]
[1] "W"
```


Aside from goals, shots <i>can</i> be considered the next greatest factor that can influence the result of a game. To have an impact on the outcome, we would expect that the teams with the maximum shot count would win their games and the teams with the lowest shot counts would lose their games. However, the results above indicate that at the highest shot totals, the teams generating those shots lost all the games (note that "W" indicates a home win and "L" indicates an away win). The Carolina Hurricanes recorded the most shots in single game - putting up 60 shots in November 2017. However, they lost that game, as did the Washington Capitals when they put up 58 shots as the home team in March 2019. The teams that put up the lowest shot counts (13) all lost their games as well. Shots doesn't match the min-max trend that I would expect - but let's see if other variables can match the trend I'm looking for. 


<b>Hits</b>

```
> summary(df$Home_H)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   4.00   17.00   22.00   22.35   27.00   54.00 
   
> home_grab(df$Home_H==54) 
[[1]]
[1] "Pittsburgh Penguins"

[[2]]
[1] "2019-12-06"

[[3]]
[1] "W"

> summary(df$Away_H)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   2.00   16.00   21.00   21.41   26.00   67.00 
   
> away_grab(df$Away_H==67) #Away team WON
[[1]]
[1] "Montreal Canadiens"

[[2]]
[1] "2019-12-10"

[[3]]
[1] "L"


> home_grab(df$Home_H==4)
[[1]]
[1] "Toronto Maple Leafs" "Chicago Blackhawks"  "Toronto Maple Leafs"
[4] "Nashville Predators"

[[2]]
[1] "2018-11-24" "2019-02-18" "2019-12-23" "2020-02-22"

[[3]]
[1] "W" "W" "W" "W"

> away_grab(df$Away_H==2) 
[[1]]
[1] "Chicago Blackhawks" "Chicago Blackhawks"

[[2]]
[1] "2017-12-14" "2018-03-17"

[[3]]
[1] "L" "W"
```

For hits, it can go either way. Either the maximum amount of hits per game can cause a team to dominate their opponent physically, or too many hits in a game can distract the players from what is truly important - scoring goals and winning the game. As we can see at the maximum amount of hits (54 for the home team and 67 for the away team), the teams with those hit amounts won their games. With that said, would we expect to see teams with much less hits lose their games? Well, that doesn't seem to be the case here with hits. When we look at the lowest hit totals (4 for the home team and 2 for the away team), the home team actually won all of their games at 4 hits while the away teams split their games (1 win, 1 loss) at 2 hits. With this mixed bag of results, it's safe to say that hits doesn't match the trend I'm looking for either.

<b> Blocked Shots </b>
 
 ```
 > summary(df$Home_BS)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   2.00   11.00   14.00   13.99   17.00   33.00 
   
> home_grab(df$Home_BS==33) 
[[1]]
[1] "Vegas Golden Knights" "New York Islanders"  

[[2]]
[1] "2018-11-14" "2020-01-06"

[[3]]
[1] "W" "W"


> summary(df$Away_BS)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   2.00   11.00   14.00   14.52   18.00   33.00 
   
> away_grab(df$Away_BS==33)
[[1]]
[1] "Colorado Avalanche"  "Philadelphia Flyers" "New York Rangers"   
[4] "New York Islanders" 

[[2]]
[1] "2018-03-20" "2018-03-28" "2019-03-23" "2019-10-19"

[[3]]
[1] "L" "L" "L" "L"


> home_grab(df$Home_BS==2)
[[1]]
[1] "Buffalo Sabres"    "St. Louis Blues"   "New Jersey Devils"
[4] "Boston Bruins"    

[[2]]
[1] "2018-10-11" "2019-10-19" "2019-11-30" "2020-01-02"

[[3]]
[1] "L" "L" "L" "L"

> away_grab(df$Away_BS==2) #Away team LOST
[[1]]
[1] "Boston Bruins"

[[2]]
[1] "2018-02-17"

[[3]]
[1] "W"
```

Have we finally got some meaningful results when looking at the max and min values? Blocked shots have always been an underrated hockey factor in my mind and teams don't get enough credit when they do a good job of getting bodies in front of pucks. When we look at the max values (33 for both home and away), the teams won all their games. When we look at the min values (2 for home and away), the teams lost of all their games. This finally matches the trend I've been looking for. Based on these facts, I'll assume that the amount of blocked shots can, in fact, influence the outcome of a hockey game.

<b> Corsi </b>

```
summary(df$Home_Corsi)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  25.90   46.20   51.30   51.18   56.42   74.80 
  
> home_grab(df$Home_Corsi==74.8) 
[[1]]
[1] "Carolina Hurricanes"

[[2]]
[1] "2019-10-06"

[[3]]
[1] "W"

> summary(df$Away_Corsi)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  25.20   43.58   48.70   48.82   53.80   74.10 
  
> away_grab(df$Away_Corsi==74.10) 
[[1]]
[1] "Carolina Hurricanes"

[[2]]
[1] "2019-03-15"

[[3]]
[1] "W"

> home_grab(df$Home_Corsi==25.9) 
[[1]]
[1] "Columbus Blue Jackets"

[[2]]
[1] "2019-03-15"

[[3]]
[1] "W"

> away_grab(df$Away_Corsi==25.2)
[[1]]
[1] "Tampa Bay Lightning"

[[2]]
[1] "2019-10-06"

[[3]]
[1] "W"
```

Corsi has been a key stat in the NHL over the last decade when looking at team possession. However, at the max and min values, Corsi is a split decision. When the Carolina Hurrincanes dominated at a 74.8% Corsi, they won while the defending team (Tampa Lightning - 25.2%) lost that game. However, when Carolina also put up a League best Corsi of 74.1% as the Away team, they lost to the Columbus Blue Jackets - who only held a Corsi % of 25.9%. When we look at the next lowest values for the home (74.6%) and the away (71.2%) team, the teams at those high corsi percentages *lost* all of their games. With that said, it's safe to say that there are more important factors that determine a winner than shots.

<b> Offensive Zone Start Percentage </b>

```
summary(df$Home_OFS)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  18.00   44.80   52.10   51.81   59.00   86.50 
> home_grab(df$Home_OFS==86.5) #Home team LOST
[[1]]
[1] "Columbus Blue Jackets"

[[2]]
[1] "2018-02-18"

[[3]]
[1] "L"

> 
> summary(df$Away_OFS)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   13.9    42.6    49.4    49.8    56.9    82.8 
> away_grab(df$Away_OFS==82.8) #Away team LOST
[[1]]
[1] "Nashville Predators"

[[2]]
[1] "2019-03-12"

[[3]]
[1] "W"

> 
> # Exploring Game stats - min OFS
> home_grab(df$Home_OFS==18) #Home team WON
[[1]]
[1] "Anaheim Ducks"

[[2]]
[1] "2019-03-12"

[[3]]
[1] "W"

> away_grab(df$Away_OFS==13.9) #Away team WO
[[1]]
[1] "Pittsburgh Penguins"

[[2]]
[1] "2018-02-18"

[[3]]
[1] "L"
```

Finally, I'll conclude this max-min comparison with the often-overlooked Offensive Zone Start %. While we would expect this to have a positive impact on the outcome of a game, would this actually have a negative impact (crazy, I know, but I don’t make up the data) instead? The highest Offensive Zone start % on the last 3 years was 86.5% for any home team and 82.8% for any away team. The teams that put up these numbers, the Columbus Blue Jackets and Nashville Predators, lost both of these games to the teams who put up the lowest Offensive Start %s over the last 3 years. Since there can only be a 100% share of offensive zone starts in a game, this means that the lowest offensive zone start %s (as shown in the data above) simultaneously won their games. If I look at the next largest values for the home (82.6%) and away (81.6%) teams, those teams also lost their games - similarly to the Corsi variable. While both corsi and offensive zone % play a role in each game, they are clearly not the most important statistics when determining a winner and loser.

# Analysis - Aggregated Data

Here we are moving away from our raw data to our aggregated data - with each game representing the team's average stats for each variable <i>prior</i> to that game being played. This is the same data I'll use for the rest of this analysis portion and will utilize for my machine learning models.

To begin, I'll start running a few different tests to compare the home and away data. It's worth exploring since I want to know if home ice truly has any sort of advantage (or has a weight when it comes to predicting). In data science terms, I want to see if there is a statistically significant difference between home and away variables.

<b> Goals </b>

The first command I'll run is to look at the distribution of the variables, followed by a shapiro test in order to confirm normality:


```
hist(Home_Goals)
abline(v = mean(Home_Goals), col = "blue")
```
<img src = "https://user-images.githubusercontent.com/39016197/87982251-c0b03c80-ca93-11ea-8d59-377a655b01a6.png" width = 430 height = 250>

```
hist(Away_Goals)
abline(v = mean(Away_Goals), col = "blue")
```
<img src = "https://user-images.githubusercontent.com/39016197/87982343-ea696380-ca93-11ea-8a17-bfadf75f47f4.png" width = 430 height = 250>

```
> shapiro.test(Home_Goals)

	Shapiro-Wilk normality test

data:  Home_Goals
W = 0.90674, p-value < 2.2e-16

> shapiro.test(Away_Goals)

	Shapiro-Wilk normality test

data:  Away_Goals
W = 0.94241, p-value < 2.2e-16
```

As we can see from the above graphs, the goal variable doesn't have a normal distribution - as the data doesn't have that bell shape spread away from the mean. This is also confirmed with the shapiro test, which our null hypothesis would be that the data is normally distributed. With the p-values at the lowest possible value in R (2.2 x e^-16), we can easily reject the null hypothesis that goal data is normally distributed.

I'll also run a quick correlation test to make sure there isn't a strong correlation between the two variables. To test if there is a statistically significant difference between the home and away goals, I'll run a T-test to make sure the means between the two variables are statistically different. However, to do this, one of the T-test assumptions is that both variables already have a normal distribution. Below is my function to normalize this data, as well as the other R commands:


```
# Normalize Function
normalize = function(x) {
     (x-min(x))/(max(x) - min(x))
 }
 
> cor(Home_Goals, Away_Goals)
[1] 0.0285731

> dt1 = dt[,c(2:3)]
> dt1 = lapply(dt1, normalize)
> t.test(dt1$Home_Goals, dt1$Away_Goals, paired = FALSE)

	Welch Two Sample t-test

data:  dt1$Home_Goals and dt1$Away_Goals
t = -73.427, df = 6956.7, p-value < 2.2e-16
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -0.10351067 -0.09812745
sample estimates:
mean of x mean of y 
0.2995634 0.4003824 
```

Based on the p-value of the T-test, we can reject the null hypothesis that there is no difference in the means between home and away goals. There is a statistically significant difference between the two variables. I'll also note that the T-test was not paired since the data comes from different participants.


<b> Corsi </b>

Next, I'll do the same commands as I did with the Goal variables but with the Corsi variable. Looking ahead at correlations for the Result variable, you'll understand later on why this variable was chosen to be looked at.

```
hist(Home_Corsi)
abline(v = mean(Home_Corsi), col = "blue")
```
<img src = "https://user-images.githubusercontent.com/39016197/87985992-c1e46800-ca99-11ea-9b7e-845fd48393c0.png" width = 430 height = 250>

```
hist((Away_Corsi))
abline(v = mean(Away_Corsi), col = "blue")
```
<img src = "https://user-images.githubusercontent.com/39016197/87986182-fc4e0500-ca99-11ea-8819-7aefb549e811.png" width = 430 height = 250>

```
shapiro.test(Home_Corsi)

	Shapiro-Wilk normality test

data:  Home_Corsi
W = 0.97762, p-value < 2.2e-16

shapiro.test(Away_Corsi)

	Shapiro-Wilk normality test

data:  Away_Corsi
W = 0.96966, p-value < 2.2e-16
```

Very similar to goals (and pretty much all of the other variables), we see that there is not a normal distribution with the corsi data. Now, I'll test for significance:
```
> cor(Home_Corsi, Away_Corsi)
[1] -0.02187122

> dt1 = dt[,c(8:9)]

> dt1 = lapply(dt1, normalize)

> t.test(dt1$Home_Corsi, dt1$Away_Corsi, paired = FALSE) 

	Welch Two Sample t-test

data:  dt1$Home_Corsi and dt1$Away_Corsi
t = 17.975, df = 7345.5, p-value <
2.2e-16
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 0.02924196 0.03640055
sample estimates:
mean of x mean of y 
0.5164798 0.4836585 
```

I'll note here that if one team dominates the corsi %, the other will have a much lower corsi% - which is why the negative relationship makes sense.  I can reject the null hypothesis that there is no difference between the means of the home and away corsi variables.

To wrap up this section, I'll confirm that the home and away variables are completely independent of each other (as they should be, based on how the data was manipulated) by running chi-square tests. I'll also run a chi-square test on our Results variable:

```
> chisq.test(table(dt[,c("Result", "Home_Goals")]))

	Pearson's Chi-squared test

data:  table(dt[, c("Result", "Home_Goals")])
X-squared = 1146.2, df = 1118, p-value = 0.2723

> chisq.test(table(dt[,c("Home_Goals", "Away_Goals")]))

	Pearson's Chi-squared test

data:  table(dt[, c("Home_Goals", "Away_Goals")])
X-squared = 1355962, df = 1243216, p-value < 2.2e-16
```

For the Chi-Square Independence test, the null hypothesis is that the variables are independent of each other. For the Result and Home Goals variables, we would obviously expect that there is an association between the two – or that they are not independent of each other. However, let’s keep in mind that I’m no longer using my raw data. The goal column now only reflects the average goal scored per game, and this doesn’t exactly tell us any stats regarding the actual outcome of the game. Since the p-value is greater than 0.05, we <i>fail</i> to reject the null hypothesis that these variables are independent of each other. It’s important to note here that while we’re not rejecting the null hypothesis, we’re not accepting it either. It just means that given the nature of the data, it’s uncertain where the variables are independent or not. It’s a bit different when we look at the home and away goals. Regardless if I’m using the raw or aggregated data, these metrics are typically independent of each other – especially in my aggregated data since this only tells us the average stat for each team prior to the outcome. Unfortunately, the p-value is below our expected threshold and would force me to reject the null hypothesis that these two stats are independent. I'll give the test the benefit of the doubt here since eventually, the two teams would have to play each other, and each team's averaged stats would have to impact each other somehow. I also understand that high scoring games are usually not one-sided – which means there can be a relationship between these two variables.



# Correlations and Regressions

Before I get to the machine learning aspect of my project, I want look at the variable correlations, as well as running regressions, on a few different variables. First, I want to run a correlation command that will show me the strongest positive and negative correlations on my predictor variable - Results. Since my Results variable is currently as factor in my data set, I'll first copy the data frame and create a numeric, binary variable to represent a "win" in a game's outcome:

<b> Results </b>
```
> dt1 = dt

> attach(dt1)

> dt1$target = ifelse(dt1$Result=="W", 1,0)

> numeric_columns = setdiff(names(dt1), 'Result')

> target_cor = cor(dt1[,numeric_columns])['target',]

> target_cor =  target_cor[order(target_cor, decreasing = TRUE)]

> head(target_cor)

    target Home_Corsi   Home_DIF     Home_P     Home_W 
1.00000000 0.06400173 0.05612663 0.05371894 0.04817111 
   Away_GA 
0.04458491 

> tail(target_cor)

   Away_OFS  Away_Corsi  Away_Goals      Away_W 
-0.05758304 -0.06531523 -0.07366870 -0.07451280 
   Away_DIF      Away_P 
-0.07481934 -0.08382848 

# Absolute Value for correlation strength only

> abs_cor =  abs(cor(dt1[,numeric_columns])['target',])

> abs_cor =  abs_cor[order(abs_cor, decreasing = TRUE)]

> abs_cor
      target       Away_P     Away_DIF       Away_W 
 1.000000000  0.083828483  0.074819337  0.074512798 
  Away_Goals   Away_Corsi   Home_Corsi     Away_OFS 
 0.073668700  0.065315229  0.064001728  0.057583035 
     Home_GA     Home_DIF      Away_SH       Home_P 
 0.056858792  0.056126633  0.054837159  0.053718941 
     Home_SA     Away_PDO       Home_W      Away_GA 
 0.052310661  0.048523831  0.048171108  0.044584913 
   Away_BS_A      Home_EN      Away_BS      Away_SA 
 0.042659400  0.039949709  0.037266626  0.035835527 
  Home_Goals   Home_Shots Home_Corsi_A      Home_SV 
 0.034851479  0.033865106  0.033164523  0.032130353 
    Home_PDO     Home_OFS    Away_EN_A   Away_Shots 
 0.031452668  0.029883373  0.029563092  0.026827002 
Away_Corsi_A   Away_OFS_A    Home_BS_A      Home_BS 
 0.025997852  0.023825416  0.023398096  0.022219481 
  Home_OFS_A      Away_SV   Away_PIM_A     Home_PIM 
 0.019505274  0.018930211  0.016135687  0.015297126 
     Away_EN      Home_SH   Home_PIM_A     Away_PIM 
 0.013809102  0.013698184  0.012456492  0.012252871 
   Home_EN_A    Away_HT_A    Home_HT_A      Away_HT 
 0.011148183  0.006317129  0.005281937  0.003529820 
     Home_HT 
 0.003073305 
```

Let's take a moment to digest this information. With the number of variables and factors that go into a hockey game, it doesn't necessarily surprise me that the strongest correlation of a game result is only .0838, but I was hoping for a variable to at least hit 0.10. When we look at the positive variables that affect the outcome, the top variables are the Home team's Corsi %, season differential (+/- based on goals scored and goals against), points, wins, and how bad the other team's defensive/goaltending is. I'll note that the classic favorite variables, goals and shots, are nowhere near the top correlation variables for the home team. In fact, when we look at correlation strength alone (abs_cor, which ignores the positive or negative relationship), home goals is ranked 21st while shots for both home and away teams fall even further. However, according to correlation strength, the average goals scored by the away team is much more important. Perhaps this is due to good team's ability to win away from home ice, or that home ice is such an advantage that it doesn't matter if a high scoring or a low scoring team is the home team - both are valid points in my opinion. So based on correlation strength alone, the outcome of a game depends on the away team and how they fare based on average point, differential, win, goals, and Corsi per game - which, I'll interpret as a team's ability to take away the home ice advantage from their opponent. Surprisingly, the away team's offensive zone start % is a high enough correlation to be near the top, but after my previous analysis with the min-max values, it'll be interesting to see how much of a factor this will really have. Despite that same min-max analysis where Corsi had a mixed result, this appears to be the top variable for both the home and away team. I'll also pay a bit more attention to the points, differential, and win variables for the remainder of this project - but it is no surprise that these variables are among top correlations (good teams win, put up points, and will usually have a positive differential).

<b> Away Points </b>

Since this was the strongest correlation to the outcome of a game, we'll start looking at correlations with this variable - as well as a regression.

```
> head(away_points_cor)

  Home_DIF   Home_PDO     Home_P     Home_W Home_Goals 
 1.0000000  0.8593444  0.8580106  0.7965284  0.7469333 
   Home_SH 
 0.6835826 
 
 
 > head(away_points_cor, n= 25L)

     Home_DIF      Home_PDO        Home_P        Home_W    Home_Goals 
 1.000000e+00  8.593444e-01  8.580106e-01  7.965284e-01  7.469333e-01 
      Home_SH       Home_SV       Home_EN    Home_Corsi    Home_Shots 
 6.835826e-01  5.982487e-01  3.687187e-01  1.712608e-01  1.695435e-01 
   Home_PIM_A     Home_HT_A     Home_BS_A      Home_OFS       Away_SV 
 1.331332e-01  1.181475e-01  5.826288e-02  5.422902e-02  1.742194e-02 
      Home_BS      Away_DIF      Away_PDO      Away_PIM       Away_BS 
 1.349992e-02  1.000112e-02  6.271051e-03  5.944267e-03  5.177181e-03 
    Away_BS_A       Away_SA    Away_PIM_A       Away_HT    Away_Goals 
 5.132681e-03  3.900925e-03  3.230196e-03  2.442725e-03 -9.518124e-05 
 ```
 
This is where these correlation results doesn't quite make sense. The top variables that have the strongest correlation with the average away points are the home team's PDO, points, wins, goals and shooting percentage - which, have nothing to do with the opposing team's points average. I guess I'm not surprised here, since the chi-square tests above thinks that home and away stats have a relationship, but this is just flat out wrong.  I don't start to see any away variables until I get into the top 15 (top 25 shown for additional away variables). In this case, the top away variables would be save %, differential, PDO, PIM, and blocked shots. Since I know these results are not correct, I'm going to use my clone data frame to include *only* the away variables:
 
 ```
dt_away = dt[, -1]
dt_away= dt_away[, rep(c(FALSE, TRUE), 22)]

away_points_cor = cor(dt_away[, unlist(lapply(dt_away, is.numeric))])
away_points_cor = away_points_cor[,18]
away_points_cor = away_points_cor[order(away_points_cor, decreasing = TRUE)]

head(away_points_cor)

    Away_P     Away_W   Away_DIF   Away_PDO Away_Goals    Away_SH 
 1.0000000  0.9122581  0.8692082  0.7610000  0.6687041  0.6053759 
 
 ```
I'll note that I had to remove the "Results" variable since I needed an even 44 columns to run the above command. Looking at the new results, this makes much, much more sense. Winning games obviously has a strong correlation with points. Differential usually tells a story - positive teams are usually winning more, however, does this stat does often get inflated by winning games by a lot of goals, but losing more games by few goals.

Next, we'll look at the regression model:

```
> model = lm(Away_P ~ Away_W + Away_DIF + Away_PDO + Away_SH)

> summary(model)

Call:
lm(formula = Away_P ~ Away_W + Away_DIF + Away_PDO + Away_SH)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.63698 -0.03295 -0.00558  0.02490  1.41566 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)
(Intercept)  0.36244    0.14181   2.556   0.0106
Away_W       1.07412    0.01726  62.246   <2e-16
Away_DIF     0.13149    0.00487  26.999   <2e-16
Away_PDO     0.19622    0.14951   1.312   0.1894
Away_SH      0.24698    0.15433   1.600   0.1096
               
(Intercept) *  
Away_W      ***
Away_DIF    ***
Away_PDO       
Away_SH        
---
Signif. codes:  
0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.09116 on 3671 degrees of freedom
Multiple R-squared:  0.8818,	Adjusted R-squared:  0.8816 
F-statistic:  6844 on 4 and 3671 DF,  p-value: < 2.2e-16
```

Since PDO and shooting percentage is not significant, I'll go ahead and run the model without those variables:

```
> model = lm(Away_P ~ Away_W + Away_DIF)

> summary(model)

Call:
lm(formula = Away_P ~ Away_W + Away_DIF)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.64154 -0.03316 -0.00537  0.02462  1.41843 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)
(Intercept) 0.581568   0.008643   67.28   <2e-16
Away_W      1.076995   0.017246   62.45   <2e-16
Away_DIF    0.140425   0.003597   39.04   <2e-16
               
(Intercept) ***
Away_W      ***
Away_DIF    ***
---
Signif. codes:  
0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.09126 on 3673 degrees of freedom
Multiple R-squared:  0.8814,	Adjusted R-squared:  0.8814 
F-statistic: 1.365e+04 on 2 and 3673 DF,  p-value: < 2.2e-16


> plot(Away_P ~ Away_W + Away_DIF)

> abline(model, col = "red")
```

<img src = "https://user-images.githubusercontent.com/39016197/88122490-fb929d00-cb85-11ea-8b02-0c53d69ded02.png" width = 410 height = 250>
<img src = "https://user-images.githubusercontent.com/39016197/88122511-0f3e0380-cb86-11ea-9ba5-1b90c641f3c6.png" width = 410 height = 250>

According to the regression model, a one unit increase in wins increasing points by (1.0769) and a one unit increase in differential increases points by 0.14. It's good to see both variables having a positive relationship with points, and both make sense.

Finally, when we look at the differential and win rate together in a plot, it makes sense that there would be a correlation with the points variable:

```
dt_away %>%
    mutate(High_Points = ifelse(Away_P > mean(Away_P), "Yes", "No")) %>%
    ggplot(aes(x = Away_W, y = Away_DIF, color = High_Points)) + 
    geom_point() + 
    labs(x = "Average Win Rate", y = "Average Differential", title = "Above Average Points")
```
<img src = "https://user-images.githubusercontent.com/39016197/88123138-7f00be00-cb87-11ea-9928-53dd307fbece.png" width = 510 height = 350>

<b> Away Save % </b>

To wrap up this section, I'll run the same commands as I did for points, but for save %.

```
> away_save_cor = cor(dt_away[, unlist(lapply(dt_away, is.numeric))])

> away_save_cor = away_save_cor[,20]

> away_save_cor = away_save_cor[order(away_save_cor, decreasing = TRUE)]

> head(away_save_cor)

  Away_SV  Away_PDO  Away_DIF    Away_P    Away_W 
1.0000000 0.7452563 0.6512525 0.5767216 0.5394995 
  Away_SA 
0.2715680 

> tail(away_save_cor)

  Away_BS_A  Away_Shots   Away_EN_A    Away_OFS 
-0.04996938 -0.17028450 -0.17593604 -0.19558513 
 Away_Corsi     Away_GA 
-0.26007001 -0.85985466 
```

I'll note that the strongest correlation here is the negative relation with goals against - which makes complete sense. PDO doesn't surprise me since save % is part of that calculation, differential is related since a save % is more likely to have a positive differential, points and wins are also straight forward. I'm surprised that shots against didn't have a stronger correlation, but I'm even more surprised that there isn't a strong correlation with blocked shots. Blocked shots can prevent goals, but also prevents shots on goal that would count toward a save, so I guess the lower correlation (0.169) isn't all that far off.

For my regression, I'll include the strong correlation variables and will also initially include blocked shots:


```
> model = lm(Away_SV ~ Away_PDO + Away_P + Away_W + Away_SA + Away_GA + Away_BS)

> summary(model)

Call:
lm(formula = Away_SV ~ Away_PDO + Away_P + Away_W + Away_SA + 
    Away_GA + Away_BS)

Residuals:
      Min        1Q    Median        3Q       Max 
-0.081196 -0.001026  0.000082  0.001327  0.019322 

Coefficients:
              Estimate Std. Error  t value
(Intercept)  8.266e-01  3.996e-03  206.868
Away_PDO     9.629e-02  4.540e-03   21.207
Away_P      -1.259e-02  5.595e-04  -22.504
Away_W       9.536e-03  8.889e-04   10.728
Away_SA      2.602e-03  3.182e-05   81.760
Away_GA     -2.900e-02  1.705e-04 -170.123
Away_BS     -4.511e-06  3.827e-05   -0.118
            Pr(>|t|)    
(Intercept)   <2e-16 ***
Away_PDO      <2e-16 ***
Away_P        <2e-16 ***
Away_W        <2e-16 ***
Away_SA       <2e-16 ***
Away_GA       <2e-16 ***
Away_BS        0.906    
---
Signif. codes:  
0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.003267 on 3669 degrees of freedom
Multiple R-squared:  0.9502,	Adjusted R-squared:  0.9501 
F-statistic: 1.167e+04 on 6 and 3669 DF,  p-value: < 2.2e-16

> model = lm(Away_SV ~ Away_PDO + Away_P + Away_W + Away_SA + Away_GA)

> summary(model)

Call:
lm(formula = Away_SV ~ Away_PDO + Away_P + Away_W + Away_SA + 
    Away_GA)

Residuals:
      Min        1Q    Median        3Q       Max 
-0.081214 -0.001026  0.000085  0.001325  0.019315 

Coefficients:
              Estimate Std. Error t value
(Intercept)  8.266e-01  3.992e-03  207.06
Away_PDO     9.622e-02  4.504e-03   21.36
Away_P      -1.259e-02  5.581e-04  -22.55
Away_W       9.537e-03  8.888e-04   10.73
Away_SA      2.601e-03  3.122e-05   83.31
Away_GA     -2.900e-02  1.704e-04 -170.23
            Pr(>|t|)    
(Intercept)   <2e-16 ***
Away_PDO      <2e-16 ***
Away_P        <2e-16 ***
Away_W        <2e-16 ***
Away_SA       <2e-16 ***
Away_GA       <2e-16 ***
---
Signif. codes:  
0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.003266 on 3670 degrees of freedom
Multiple R-squared:  0.9502,	Adjusted R-squared:  0.9501 
F-statistic: 1.401e+04 on 5 and 3670 DF,  p-value: < 2.2e-16
```

Well, that didn't last long. Blocked shots had an enormous p-value, and did not have any statistically significant weight compared to the other stats. You can see that this variable was dropped and the model was re-run. While Goals against isn't surprising when looking at the negative relationships, I am shocked that points has a negative relationship. For every one unit increase in points, save % is expected to drop by 0.01259 - which isn't a big change, but still interesting to see. You would expect to see good goaltending save % that results in more points.

```
dt_away %>%
    mutate(High_Save = ifelse(Away_SV > mean(Away_SV), "Yes", "No")) %>%
    ggplot(aes(x = Away_W, y = Away_P, color = High_Save)) + geom_point() + labs(x = "Average Win Rate", y = "Average Points per Game", title = "Above Average Saves") +   geom_smooth(method = "lm")
```
<img src = "https://user-images.githubusercontent.com/39016197/88235330-e118fc00-cc37-11ea-95ab-b6ded30a78da.png" width = 510 height = 350>

Finally, you can see that a high save % is usually found at the higher end of points per game (despite the negative correlation) and average win per game. I'll note that the linear regression fit line has a steeper slope for the save % values below the mean - which does *not* mean that winning with a lower save %, on average, results in more points on average. The regression line is simply a best fitting line based on the data - which can be skewed from the few outliers that we see in the plot. In fact, if we look back at the raw data, we'll see that a higher save % certainly results in more points: 

```
> sum(dt$Home_P[dt$Home_SV>mean(dt$Home_SV)])
[1] 3200

> sum(dt$Home_P[dt$Home_SV<mean(dt$Home_SV)])
[1] 1160

> sum(dt$Away_P[dt$Away_SV>mean(dt$Away_SV)])
[1] 2886

> sum(dt$Away_P[dt$Away_SV<mean(dt$Away_SV)])
[1] 819
```
# Predicting

Finally, the section you've been waiting for! Is it possible to reasonably predict NHL games based on the data I've collected? Well, we're about to find out! First, I'll begin with a basic logistic regression - I'll note that I don't have high hopes for a good accuracy on this model. I'll then look at some of the more complex models, such as SVM and ANN, to try to get my accuracy as high as possible.

# Logistic Regression 

I'll recall from the above that I'm using "Dtrain" as my training set and "Dtest" for my testing set. For a logistic regression, there's a few assumptions I'll need to familiarize myself with:

1. There needs to be a linear relationship between the logit of the outcome and each predictor variables.
2. There should not be any outliers in any continuous variables
3. No high multicollinearity among the predictor variables

I already know that there is high multicollinearity among some of these variables (take PDO for example, which is the direct calculation of save % and shooting %), so I'll need to make sure that some of these variables are removed. However, before I get to that step, I'll first run a logistic regression with all of my variables for comparison:

```
> attach(dtrain)

> logistic = glm(Result ~ ., data = dtrain, family = "binomial")

> summary(logistic)

Call:
glm(formula = Result ~ ., family = "binomial", data = dtrain)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.5712  -1.2021   0.8712   1.0801   2.1024  

Coefficients:
               Estimate Std. Error z value Pr(>|z|)   
(Intercept)   -2.209040  28.895598  -0.076  0.93906   
Home_Goals     1.835300   0.800152   2.294  0.02181 * 
Away_Goals     0.011210   0.736458   0.015  0.98786   
Home_PIM       0.069016   0.033449   2.063  0.03908 * 
Away_PIM      -0.086199   0.036179  -2.383  0.01719 * 
Home_Shots    -0.128591   0.071666  -1.794  0.07276 . 
Away_Shots     0.172258   0.052614   3.274  0.00106 **
Home_Corsi     0.275448   0.185829   1.482  0.13827   
Away_Corsi    -0.058952   0.199119  -0.296  0.76718   
Home_OFS      -0.144572   0.168812  -0.856  0.39177   
Away_OFS      -0.126106   0.173015  -0.729  0.46608   
Home_HT        0.012384   0.015618   0.793  0.42782   
Away_HT       -0.021243   0.015791  -1.345  0.17854   
Home_BS        0.065205   0.050873   1.282  0.19994   
Away_BS       -0.068005   0.048713  -1.396  0.16271   
Home_EN        1.191431   0.549968   2.166  0.03028 * 
Away_EN        0.190133   0.505907   0.376  0.70705   
Home_GA       -1.398821   0.825524  -1.694  0.09018 . 
Away_GA       -0.320881   0.761943  -0.421  0.67366   
Home_PIM_A    -0.031708   0.038698  -0.819  0.41257   
Away_PIM_A     0.083368   0.043394   1.921  0.05471 . 
Home_SA        0.060267   0.080031   0.753  0.45142   
Away_SA       -0.067543   0.058373  -1.157  0.24723   
Home_Corsi_A   0.080533   0.170865   0.471  0.63741   
Away_Corsi_A   0.137977   0.184159   0.749  0.45372   
Home_OFS_A    -0.115216   0.168394  -0.684  0.49385   
Away_OFS_A    -0.100760   0.173810  -0.580  0.56211   
Home_HT_A     -0.004641   0.017868  -0.260  0.79505   
Away_HT_A      0.029602   0.017699   1.672  0.09443 . 
Home_BS_A     -0.043187   0.044698  -0.966  0.33394   
Away_BS_A      0.031780   0.041716   0.762  0.44617   
Home_EN_A      0.443807   0.767816   0.578  0.56326   
Away_EN_A     -0.409631   0.638851  -0.641  0.52139   
Home_W         0.099034   1.088459   0.091  0.92750   
Away_W         0.320636   1.055531   0.304  0.76130   
Home_P        -0.145406   0.631942  -0.230  0.81802   
Away_P        -0.426628   0.637420  -0.669  0.50330   
Home_DIF      -1.163728   0.791786  -1.470  0.14163   
Away_DIF      -0.538459   0.707696  -0.761  0.44674   
Home_SV      -39.613386  64.559139  -0.614  0.53948   
Away_SV      -71.785119  67.764179  -1.059  0.28945   
Home_SH      -53.930775  65.732908  -0.820  0.41196   
Away_SH      -70.194137  67.897705  -1.034  0.30122   
Home_PDO      39.485752  65.455078   0.603  0.54634   
Away_PDO      76.743247  68.065464   1.127  0.25953   
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 3571.1  on 2593  degrees of freedom
Residual deviance: 3464.7  on 2549  degrees of freedom
AIC: 3554.7

Number of Fisher Scoring iterations: 5
```


The family needs to be binomial since I'm predicting a categorical variable with only 2 options. Based on the above, there's only a handful of variables that are statistically significant enough to even consider keeping in my model. But as variables get removed, the p-values will begin to change for different variables. Normally, you would want to remove all of the statistically insignificant variables out of the model until you only have p-values less than the confidence interval picked. However, I believe that I'll need more variables than the few that are significant above to get much success out of this regression.

For comparison purposes, I'll still predict using the full variables set:


```
> pred = predict(logistic, newdata = test, type = "response")

> pred = round(pred)

> results = table(dtest$Result, pred)

> results

   pred
      0   1
  L 171 334
  W 139 438
  

> recal = results[4]/(results[4] + results[2])

> prec = results[4]/(results[4] + results[3])

> acc = (results[1] + results[4])/nrow(dtest)

> recal

[1] 0.7590988

> prec
[1] 0.5673575

> acc
[1] 0.5628466
```

I set up a prediction variable 'pred', that was set up as a response type since I wanted to see the fitted values (since the residual values wouldn't help me narrow this down into 2 categories). Then, I needed to round the fitted variables so that I have either a 0 or 1. While I  could change my results variable to a binary, numeric variable, I left it as it is since I know that 0 is "L" and 1 is "W".  I'll also note that I predicted the model on my test set instead of dtest, since I'll leave dtest how it is so that I have a way to compare the results (recall that test is dtest without the labels). Once everything has been set up correctly, I ran a table to check my results. First, I looked at the recall ('recal') to see that it is 0.759 - which is not bad. However, as most data scientists understand, having a good recall doesn't mean that the model is any good. On the flip side of that, the recall for the "L" category is only 0.339. Looking at the precision (again for wins), we only see 0.567. Even worse, our accuracy for the entire model is only 56.3%. Let's hope we can do better than that with at least one of the logistic regressions below.

<b> Goals only </b>

Below is a logistic model when looking *only* at the goal variables for each team:


```
> logistic = glm(Result ~ Home_Goals + Away_Goals, data = dtrain, family = "binomial")

> summary(logistic)

Call:
glm(formula = Result ~ Home_Goals + Away_Goals, family = "binomial", 
    data = dtrain)

Deviance Residuals: 
   Min      1Q  Median      3Q     Max  
-1.742  -1.250   1.025   1.094   1.472  

Coefficients:
            Estimate Std. Error z value Pr(>|z|)
(Intercept)  0.22571    0.32992   0.684  0.49390
Home_Goals   0.21247    0.07864   2.702  0.00689
Away_Goals  -0.22156    0.08263  -2.681  0.00733
              
(Intercept)   
Home_Goals  **
Away_Goals  **
---
Signif. codes:  
0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 3571.1  on 2593  degrees of freedom
Residual deviance: 3557.3  on 2591  degrees of freedom
AIC: 3563.3

Number of Fisher Scoring iterations: 4


> pred = predict(logistic, newdata = test, type = "response")

> pred = round(pred)

> results = table(dtest$Result, pred)

> results

   pred
      0   1
  L  49 456
  W  48 529

> recal = results[4]/(results[4] + results[2])

> prec = results[4]/(results[4] + results[3])

> acc = (results[1] + results[4])/nrow(dtest)

> recal
[1] 0.9168111

> prec
[1] 0.5370558

> acc
[1] 0.5341959
```

Let's look at the above model. My first impression is that the goal coefficients are correct - home team goals increase the log odds (since the result is based on the home team) while away goals hurt the chances of a home win. While away goals were not statistically significant when every variable was included, it is now significant since there are only 2 variables included in this model. Once the model was run, I first looked at the recall - which is a whopping 91.68% when we look at the number of "W"s predicted correctly! Of course, having a high recall for "W"s means that I have an extremely low recall for "L" (about 9.7%).  When looking at the precision of "W"s, it was only 53.7%. When looking at the whole accuracy for the model, I only was able to correctly predict 53.42% (I can guess games much better than that rate).

<b> Best Linear Model? </b>

After experimenting with which variables I think should be in the regression and which variables should not be, I believe I came up with the best possible linear model with the data I have. I'll note that most of the core stats was able to stay in the model - which includes goals, shots, corsi, hits, blocked shots, differential, wins, points, save %, and shooting %. The surprise variables that also stayed include empty net goals and penalty minutes - which makes me glad that I was able to decipher out the empty net goals with my web scraper and create a separate column. PDO, which is save % + shooting %, was able to stay in the model as long as it could - but ultimately, it was the last variable removed to get the best accuracy possible. This made sense due to the high colinearity between PDO and the variables it is based off. I'll also note that all of my "against" variables, such as goals against, shots against, didn't have much weight in the logistic regression, but I'm still hoping these will come in handy when I move on to the SVM and ANN models. 

Here is the best linear logistic model I can create with this data set:


```
> summary(logistic)

Call:
glm(formula = Result ~ Home_Goals + Away_Goals + Home_Shots + 
    Away_Shots + Home_PIM + Away_PIM + Home_Corsi + Away_Corsi + 
    Home_HT + Away_HT + Home_EN + Away_EN + Home_BS + Away_BS + 
    Home_SH + Away_SH + Home_DIF + Away_DIF + Home_W + Away_W + 
    Home_P + Away_P + Home_SV + Away_SV, family = "binomial", 
    data = dtrain)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.5132  -1.2169   0.9001   1.0820   1.8356  

Coefficients:
              Estimate Std. Error z value
(Intercept)  -2.697449  14.403414  -0.187
Home_Goals    0.929243   0.480477   1.934
Away_Goals   -0.463775   0.410064  -1.131
Home_Shots   -0.125061   0.053002  -2.360
Away_Shots    0.123895   0.044644   2.775
Home_PIM      0.051619   0.023720   2.176
Away_PIM     -0.045629   0.023459  -1.945
Home_Corsi    0.117232   0.036860   3.180
Away_Corsi   -0.139971   0.037718  -3.711
Home_HT       0.013684   0.013517   1.012
Away_HT      -0.008546   0.013307  -0.642
Home_EN       1.136173   0.508977   2.232
Away_EN       0.287071   0.458700   0.626
Home_BS       0.027398   0.039117   0.700
Away_BS      -0.027131   0.037710  -0.719
Home_SH     -18.518776  16.703249  -1.109
Away_SH       1.874765  12.043321   0.156
Home_DIF     -0.212497   0.259394  -0.819
Away_DIF      0.218623   0.273273   0.800
Home_W       -0.329124   0.707424  -0.465
Away_W        0.332814   0.790311   0.421
Home_P        0.285682   0.452774   0.631
Away_P       -0.570194   0.497623  -1.146
Home_SV      11.854076   8.466455   1.400
Away_SV      -7.250930   8.395518  -0.864
            Pr(>|z|)    
(Intercept) 0.851442    
Home_Goals  0.053113 .  
Away_Goals  0.258062    
Home_Shots  0.018297 *  
Away_Shots  0.005517 ** 
Home_PIM    0.029541 *  
Away_PIM    0.051763 .  
Home_Corsi  0.001471 ** 
Away_Corsi  0.000206 ***
Home_HT     0.311390    
Away_HT     0.520734    
Home_EN     0.025597 *  
Away_EN     0.531423    
Home_BS     0.483678    
Away_BS     0.471857    
Home_SH     0.267563    
Away_SH     0.876294    
Home_DIF    0.412668    
Away_DIF    0.423701    
Home_W      0.641757    
Away_W      0.673669    
Home_P      0.528067    
Away_P      0.251864    
Home_SV     0.161477    
Away_SV     0.387771    
---
Signif. codes:  
0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 3571.1  on 2593  degrees of freedom
Residual deviance: 3490.1  on 2569  degrees of freedom
AIC: 3540.1

Number of Fisher Scoring iterations: 4


> pred = predict(logistic, newdata = test, type = "response")

> pred = round(pred)

> results = table(dtest$Result, pred)

> results

   pred
      0   1
  L 187 318
  W 144 433
  

> recal = results[4]/(results[4] + results[2])

> prec = results[4]/(results[4] + results[3])

> acc = (results[1] + results[4])/nrow(dtest)

> recal
[1] 0.7504333

> prec
[1] 0.5765646

> acc
[1] 0.5730129
```

The best accuracy I could muster was a 57.3% - only roughly 1% higher than if I simply kept all of my variables into the model. I'm also sharing below one of many graphs that can be completed with the logistic regression. As we can see, the predictions were all over the place - the model could have done almost just as well by flipping a coin for every game.

```
pred = predict(logistic, newdata = test, type = "response") # removed the rounding to see real value
pred.data = data.frame(winning.chances = pred, results=dtest$Result)
pred.data = pred.data[order(pred.data$winning.chances, decreasing = FALSE),]
pred.data$rank = 1:nrow(pred.data)

ggplot(data = pred.data, aes(x = rank, y = winning.chances)) + geom_point(aes (color = dtest$Result), alpha=1, shape = 2, stroke=2) + labs(x = "Game Number", y = "Predicted Wins/Losses", title = "Logistic Regression Results")
```
<img src = "https://user-images.githubusercontent.com/39016197/88246870-16821180-cc59-11ea-919a-487553c351da.png" width = 510 height = 350>

<b> Best Logistic Model </b>

While not much better than the linear model, I was able to slightly improve the accuracy using a polynomial regression as opposed to a linear. In order to see which variables have a linear relationship with the dependent variable and which variables don't, here is a useful command:

```
mydata = test[, c(6,8,39,41)] %>%
    dplyr::select_if(is.numeric)

predictors = colnames(mydata)

mydata = mydata %>%
    mutate(logit = log(pred/(1-pred))) %>%
    gather(key = "predictors", value = "predictor.value", -logit)
    
ggplot(mydata, aes(logit, predictor.value)) + geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess") + 
	theme_bw() + theme_bw() + facet_wrap(~predictors, scales = "free_y")
```

<img src = "https://user-images.githubusercontent.com/39016197/88611171-b4a11d80-d045-11ea-830c-79873d980c3e.png" width = 450 height = 320>

After reviewing the variables, it did appear that other variables also had a polynomial shaped curve, but ultimately, changing those variables as polynomial made the model worse except for the 4 shown above. By adding a 2nd degree polynomial curve to home shots, penalty minutes, and away wins, as well as adding a 3rd degree to away points, I get the best possible accuracy of 57.95%.  

```
> poly = glm(Result ~ Home_Goals + Away_Goals + poly(Home_Shots, deg=2) + Away_Shots + poly(Home_PIM, deg=2) + Away_PIM + Home_Corsi + Away_Corsi + Home_HT + Away_HT + Home_EN + Away_EN + Home_BS + Away_BS + Home_SH + Away_SH + Home_DIF + Away_DIF + Home_W + poly(Away_W, deg=2) + Home_P + poly(Away_P, deg=3) + Home_SV + Away_SV, dtrain, family = "binomial")

> pred = predict(poly, newdata = test, type = "response")

> pred = round(pred)

> pred = as.factor(pred)

> dtest$Result = ifelse(dtest$Result=="W", 1, 0)

> dtest$Result = as.factor(dtest$Result)

> confusionMatrix(dtest$Result, pred)
Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 193 312
         1 143 434
                                          
               Accuracy : 0.5795          
                 95% CI : (0.5494, 0.6091)
    No Information Rate : 0.6895          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.1372          
                                          
 Mcnemar's Test P-Value : 3.381e-15       
                                          
            Sensitivity : 0.5744          
            Specificity : 0.5818          
         Pos Pred Value : 0.3822          
         Neg Pred Value : 0.7522          
             Prevalence : 0.3105          
         Detection Rate : 0.1784          
   Detection Prevalence : 0.4667          
      Balanced Accuracy : 0.5781          
                                          
       'Positive' Class : 0 
```

I'll also note the concerning low kappa coefficient - which is in the range of 'poor.' For those who are not familiar, the kappa coefficient adjusts accuracy by accounting for the change of the model simply guessing correctly. The R^2 is also extremely low 0.023 for my best model and only 0.298 when all 45 (only the ones used for machine learning) variables are present. This may hint that either the logistic model is the complete wrong model to use, or there are just simply to many factors in hockey to predict games accurately. Below is an easy way to calculate R^2 on a logistic regression. Let's hope that I have much better success with the Support Vector Machine and the Neural Network. On to the next model.

```
> log_likelihood_null = poly$null.deviance/-2

> log_likelihood_proposed = poly$deviance/-2

> (log_likelihood_null - log_likelihood_proposed)/log_likelihood_null
[1] 0.02319687 
```


# Support Vector Machine

While the Support Vector Machine is a more complex model than the logistic regression and often has more prediction power, I'll note earlier than I do not believe the SVM will predict better than my neural network. Why? Well, if you understand how a SVM uses a hyperplane to separate groups, you'll understand why this may not be the greatest model for my almost-randomized data as we've previously have seen in other plots:
```
plot(dtrain$Home_Goals, dtrain$Away_Goals, col = dtrain$Result)
```

<img src = "https://user-images.githubusercontent.com/39016197/90063356-7d1dac80-dca6-11ea-825c-ca338b22339c.png" width = 450 height = 320>

```
#SVM with only goal variables

> svm_model = ksvm(Result ~ Home_Goals + Away_Goals, data = dtrain, C = 0.5)

> pred = predict(svm_model, test)

> results = table(dtest$Result, pred)

> results
   pred
      L   W
  L 147 358
  W 122 455

> recal = results[4]/(results[4] + results[2])

> prec = results[4]/(results[4] + results[3])

> acc = (results[1] + results[4])/nrow(dtest)

> recal
[1] 0.7885615
> prec
[1] 0.5596556
> acc
[1] 0.5563771
```

It's important to note here that many believe that your data should be normalized/scaled when using SVM - but I do not believe it is necessary for the SVM. In fact, when using scaled data, the above model dips down to 54% as opposed to the 55.6% that we see above. 

Overall, the results were not kind to me with the SVM. Even after trying different variable combinations, different kernel methods, adjusting cost constraints and epsilon values, below is the best model I could muster:


```
# Best model

> svm_model = ksvm(Result ~ Home_Goals + Away_Goals + Home_Shots + Away_Shots + Home_Corsi + Away_Corsi + Home_Corsi_A + Away_Corsi_A + Home_SA + Away_SA + Home_PIM_A + Away_PIM_A, data = dtrain, C = 0.1, kpar=list(sigma=0.1))


> svm_model
Support Vector Machine object of class "ksvm" 

SV type: C-svc  (classification) 
 parameter : cost C = 0.1 

Gaussian Radial Basis kernel function. 
 Hyperparameter : sigma =  0.1 

Number of Support Vectors : 2370 

Objective Function Value : -229.3267 
Training error : 0.427525 

> pred = predict(svm_model, test)

> confusionMatrix(table(pred, dtest$Result), positive = "W")
Confusion Matrix and Statistics

    
pred   L   W
   L 155 110
   W 350 467
                                          
               Accuracy : 0.5749          
                 95% CI : (0.5448, 0.6045)
    No Information Rate : 0.5333          
    P-Value [Acc > NIR] : 0.00329         
                                          
                  Kappa : 0.1198          
                                          
 Mcnemar's Test P-Value : < 2e-16         
                                          
            Sensitivity : 0.8094          
            Specificity : 0.3069          
         Pos Pred Value : 0.5716          
         Neg Pred Value : 0.5849          
             Prevalence : 0.5333          
         Detection Rate : 0.4316          
   Detection Prevalence : 0.7551          
      Balanced Accuracy : 0.5581          
                                          
       'Positive' Class : W
```

<img src = "https://user-images.githubusercontent.com/39016197/88880951-4eee9600-d1eb-11ea-8b6a-0833311c804e.png" width = 450 height = 320>

You may notice that the variables are a little different than the logistic regression. While the goals, shots and corsi variables are all the same, this model actually improved by adding a few of the "against" statistics - such as corsi against, shots against, and penalty minutes against. To me, these are variables that should be included in the model (as well as the other 47 variables I went through the trouble of collecting) since the first two variables are defensive weaknesses while the penalty minutes against opens up more power play opportunities for scoring. Nonetheless, these are the variables that work best with this model. After many attempts, I couldn't get the SVM to even beat the logistic regression. On to the Neural Network.

# Neural Network

To start, I'll run a basic neural network model, with all of the variables, using the neural net package. I don't expect good results for this model since I prefer running neural networks using the keras/tensorflow packages -  many of the activation & optimizer functions, as well as the input/output layers, are easily modified with keras compared to the neuralnet package model. The only real reason to run this model is to show a plot example of the network neurons and layers, since this is not currently an option for the keras/tf packages. Unlike the SVM, I definitely want to normalize my data before running a neural network model:
```
dtrain_norm = as.data.frame(lapply(dtrain[,2:45], normalize))

dtrain_norm = cbind(dtrain$Result, dtrain_norm)
names(dtrain_norm)[1] = "Result"

dtest_norm = as.data.frame(lapply(dtest[,2:45], normalize))

attach(dtrain_norm)

set.seed(18)

ann_model = neuralnet(Result ~ ., data = dtrain_norm, hidden = c(50,10), threshold = 7.0, stepmax = 1e5, err.fct = "ce", act.fct = "logistic", linear.output = FALSE, lifesign = 'full', learningrate = NULL, rep = 1)
hidden: 50, 10    thresh: 7    rep: 1/1    steps:     519	error: 97.93533	time: 13.04 secs

net.predict = compute(ann_model, dtest_norm)$net.result

net.prediction = c("W", "L")[apply(net.predict, 1, which.max)]

predict.table = table(dtest$Result, net.prediction)

predict.table
   net.prediction
      L   W
  L 263 242
  W 311 266

net.prediction = as.factor(net.prediction)

confusionMatrix(net.prediction, dtest$Result, positive = "W")
Confusion Matrix and Statistics

          Reference
Prediction   L   W
         L 263 311
         W 242 266
                                          
               Accuracy : 0.4889          
                 95% CI : (0.4587, 0.5192)
    No Information Rate : 0.5333          
    P-Value [Acc > NIR] : 0.998421        
                                          
                  Kappa : -0.018          
                                          
 Mcnemar's Test P-Value : 0.003832        
                                          
            Sensitivity : 0.4610          
            Specificity : 0.5208          
         Pos Pred Value : 0.5236          
         Neg Pred Value : 0.4582          
             Prevalence : 0.5333          
         Detection Rate : 0.2458          
   Detection Prevalence : 0.4695          
      Balanced Accuracy : 0.4909          
                                          
       'Positive' Class : W               
                              

plot(ann_model, col.hidden = 'darkgreen', col.hidden.synapse = 'darkgreen', col.intercept = "red", col.out = "blue", show.weights = F, information = F, fill = 'lightblue')
```

<img src = "https://user-images.githubusercontent.com/39016197/89111358-22689300-d412-11ea-983a-c0db622d3e02.png" width = 600 height = 500>

Despite how powerful the neural network is, these results aren't good. In my opinion, in order to run this correctly, I'll need to utilize the keras/tensorflow packages in R. Not only will this speed up the model time (I set the loss threshold high enough for the model to run only once since it not only takes a while to run, but it constantly gets stuck at the same loss threshold - which causes the model to be invalid), but the ability to customize the neural network will be much more beneficial for getting good results. Using the Keras package also has great visualization tools to compare the loss and accuracy of your training data to the loss and accuracy of your validation data.

```
library(keras)
library(tensorflow)
library(deepviz)
library(magrittr)
```
After installing the necessary packages, I can proceed with this 'upgraded' neural network model. I'll note that using the keras model with tensorflow will require a different set up than what I currently have. Essentially, I'll need to convert my data into a 2D tensor (also called a matrix) and one-hot encode my predictor variable. I'll then separate out my training and test labels.

```
# Set up Model
dtrain$Result = as.numeric(dtrain$Result)
dtest$Result = as.numeric(dtest$Result)

train = as.matrix(dtrain)
test = as.matrix(dtest)

dimnames(train) = NULL
dimnames(test) = NULL

train[,2:45] = scal(train[,2:45])
test[,2:45] = scal(test[,2:45])

train[,1] = train[,1] - 1
test[,1] = test[,1] - 1

traintarget = train[,1]
testtarget = test[,1]

train = train[, 2:45]
test = test[, 2:45]

trainLabels = to_categorical(traintarget) 
testLabels = to_categorical(testtarget)

head(testLabels)
     [,1] [,2]
[1,]    0    1
[2,]    0    1
[3,]    0    1
[4,]    1    0
[5,]    0    1
[6,]    1    0
```
As opposed to my last neural network model, I'll note that I scaled my data this time instead - as this seems to yield more promising results than normalized data (believe me, I've checked).  Next, I'll build my new neural network using API functional:

```
model = NULL

model <- local({
  input = layer_input(shape = c(44), name = 'main_input')
  
  layer1 = input %>%
    layer_dense(units = 30, activation = "relu")
  
  layer2 = input %>%
    layer_dense(units = 30, activation = "relu")
  
  output = layer_concatenate(c(layer1, layer2)) %>%
    layer_dropout(0.5) %>%
    layer_batch_normalization() %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = 2, activation = "sigmoid", name = 'main_output')
  
  keras_model(inputs = input, outputs = output)
})
```

I'll note that instead of 50 hidden layers followed by 10, I have 30 node layers split into two paths before concatenating. This is then followed by a dropout and batch normalization layer before wrapping up with 10 more hidden nodes before the output layer. More importantly, I have more control over my input and output layers.

```
summary(model)
Model: "model_4"
________________________________________________________________________________________________________________________________________
Layer (type)                                Output Shape                  Param #          Connected to                                 
========================================================================================================================================
main_input (InputLayer)                     [(None, 44)]                  0                                                             
________________________________________________________________________________________________________________________________________
dense_16 (Dense)                            (None, 30)                    1350             main_input[0][0]                             
________________________________________________________________________________________________________________________________________
dense_17 (Dense)                            (None, 30)                    1350             main_input[0][0]                             
________________________________________________________________________________________________________________________________________
concatenate_4 (Concatenate)                 (None, 60)                    0                dense_16[0][0]                               
                                                                                           dense_17[0][0]                               
________________________________________________________________________________________________________________________________________
dropout_4 (Dropout)                         (None, 60)                    0                concatenate_4[0][0]                          
________________________________________________________________________________________________________________________________________
batch_normalization_4 (BatchNormalization)  (None, 60)                    240              dropout_4[0][0]                              
________________________________________________________________________________________________________________________________________
dense_18 (Dense)                            (None, 10)                    610              batch_normalization_4[0][0]                  
________________________________________________________________________________________________________________________________________
main_outout (Dense)                         (None, 2)                     22               dense_18[0][0]                               
========================================================================================================================================
Total params: 3,572
Trainable params: 3,452
Non-trainable params: 120
________________________________________________________________________________________________________________________________________
```

Model architecture:

```
model %>% plot_model()
```
<img src = "https://user-images.githubusercontent.com/39016197/89111757-d704b380-d416-11ea-82c7-c83908d6c0ea.png" width = 380 height = 420>

Notice that the architecture is quite different than what the first model looked like (excluding node count). I'll also get to choose my loss, optimizer (both available in the first model, but not as effective) and metrics. I'll run 40 epochs with batch sample sizes of 32, and a validation set at 20% of my training data (which does reduce the overall size of my training data).

```
> model %>%
+   compile(loss = "binary_crossentropy",
+           optimizer = 'adam',
+           metrics = "accuracy")
> 
> history = model %>%
+   fit(train,
+       trainLabels,
+       epoch = 40,
+       batch_size = 32,
+       validation_split = 0.2)
Train on 2075 samples, validate on 519 samples
Epoch 1/40
2075/2075 [==============================] - 1s 420us/sample - loss: 0.7822 - accuracy: 0.5031 - val_loss: 0.6887 - val_accuracy: 0.5356
Epoch 2/40
2075/2075 [==============================] - 0s 131us/sample - loss: 0.7317 - accuracy: 0.5169 - val_loss: 0.6874 - val_accuracy: 0.5405
Epoch 3/40
2075/2075 [==============================] - 0s 128us/sample - loss: 0.7109 - accuracy: 0.5154 - val_loss: 0.6877 - val_accuracy: 0.5520
Epoch 4/40
2075/2075 [==============================] - 0s 137us/sample - loss: 0.7033 - accuracy: 0.5292 - val_loss: 0.6869 - val_accuracy: 0.5472
Epoch 5/40
2075/2075 [==============================] - 0s 141us/sample - loss: 0.6927 - accuracy: 0.5557 - val_loss: 0.6864 - val_accuracy: 0.5424
Epoch 6/40
2075/2075 [==============================] - 0s 164us/sample - loss: 0.6862 - accuracy: 0.5670 - val_loss: 0.6860 - val_accuracy: 0.5405
Epoch 7/40
2075/2075 [==============================] - 0s 165us/sample - loss: 0.6827 - accuracy: 0.5737 - val_loss: 0.6848 - val_accuracy: 0.5462
Epoch 8/40
2075/2075 [==============================] - 0s 174us/sample - loss: 0.6829 - accuracy: 0.5641 - val_loss: 0.6843 - val_accuracy: 0.5511
Epoch 9/40
2075/2075 [==============================] - 0s 186us/sample - loss: 0.6822 - accuracy: 0.5610 - val_loss: 0.6834 - val_accuracy: 0.5462
Epoch 10/40
2075/2075 [==============================] - 0s 180us/sample - loss: 0.6830 - accuracy: 0.5655 - val_loss: 0.6832 - val_accuracy: 0.5520
Epoch 11/40
2075/2075 [==============================] - 0s 178us/sample - loss: 0.6833 - accuracy: 0.5788 - val_loss: 0.6834 - val_accuracy: 0.5453
Epoch 12/40
2075/2075 [==============================] - 0s 184us/sample - loss: 0.6824 - accuracy: 0.5663 - val_loss: 0.6828 - val_accuracy: 0.5491
Epoch 13/40
2075/2075 [==============================] - 0s 185us/sample - loss: 0.6788 - accuracy: 0.5723 - val_loss: 0.6815 - val_accuracy: 0.5520
Epoch 14/40
2075/2075 [==============================] - 0s 188us/sample - loss: 0.6762 - accuracy: 0.5725 - val_loss: 0.6808 - val_accuracy: 0.5511
Epoch 15/40
2075/2075 [==============================] - 0s 192us/sample - loss: 0.6767 - accuracy: 0.5783 - val_loss: 0.6807 - val_accuracy: 0.5491
Epoch 16/40
2075/2075 [==============================] - 0s 177us/sample - loss: 0.6788 - accuracy: 0.5677 - val_loss: 0.6806 - val_accuracy: 0.5568
Epoch 17/40
2075/2075 [==============================] - 0s 182us/sample - loss: 0.6756 - accuracy: 0.5870 - val_loss: 0.6802 - val_accuracy: 0.5539
Epoch 18/40
2075/2075 [==============================] - 0s 179us/sample - loss: 0.6709 - accuracy: 0.5752 - val_loss: 0.6794 - val_accuracy: 0.5511
Epoch 19/40
2075/2075 [==============================] - 0s 174us/sample - loss: 0.6741 - accuracy: 0.5781 - val_loss: 0.6796 - val_accuracy: 0.5511
Epoch 20/40
2075/2075 [==============================] - 0s 180us/sample - loss: 0.6720 - accuracy: 0.5807 - val_loss: 0.6797 - val_accuracy: 0.5434
Epoch 21/40
2075/2075 [==============================] - 0s 171us/sample - loss: 0.6668 - accuracy: 0.6012 - val_loss: 0.6789 - val_accuracy: 0.5491
Epoch 22/40
2075/2075 [==============================] - 0s 179us/sample - loss: 0.6763 - accuracy: 0.5761 - val_loss: 0.6792 - val_accuracy: 0.5462
Epoch 23/40
2075/2075 [==============================] - 0s 191us/sample - loss: 0.6678 - accuracy: 0.5940 - val_loss: 0.6792 - val_accuracy: 0.5434
Epoch 24/40
2075/2075 [==============================] - 0s 186us/sample - loss: 0.6667 - accuracy: 0.5990 - val_loss: 0.6784 - val_accuracy: 0.5539
Epoch 25/40
2075/2075 [==============================] - 0s 179us/sample - loss: 0.6656 - accuracy: 0.5973 - val_loss: 0.6779 - val_accuracy: 0.5626
Epoch 26/40
2075/2075 [==============================] - 1s 345us/sample - loss: 0.6656 - accuracy: 0.5947 - val_loss: 0.6776 - val_accuracy: 0.5655
Epoch 27/40
2075/2075 [==============================] - 0s 178us/sample - loss: 0.6682 - accuracy: 0.5896 - val_loss: 0.6772 - val_accuracy: 0.5626
Epoch 28/40
2075/2075 [==============================] - 0s 176us/sample - loss: 0.6640 - accuracy: 0.6029 - val_loss: 0.6772 - val_accuracy: 0.5578
Epoch 29/40
2075/2075 [==============================] - 0s 174us/sample - loss: 0.6638 - accuracy: 0.5930 - val_loss: 0.6762 - val_accuracy: 0.5578
Epoch 30/40
2075/2075 [==============================] - 0s 181us/sample - loss: 0.6645 - accuracy: 0.6017 - val_loss: 0.6760 - val_accuracy: 0.5636
Epoch 31/40
2075/2075 [==============================] - 0s 178us/sample - loss: 0.6642 - accuracy: 0.5906 - val_loss: 0.6760 - val_accuracy: 0.5645
Epoch 32/40
2075/2075 [==============================] - 0s 173us/sample - loss: 0.6618 - accuracy: 0.6031 - val_loss: 0.6747 - val_accuracy: 0.5742
Epoch 33/40
2075/2075 [==============================] - 0s 178us/sample - loss: 0.6638 - accuracy: 0.5995 - val_loss: 0.6753 - val_accuracy: 0.5703
Epoch 34/40
2075/2075 [==============================] - 0s 175us/sample - loss: 0.6653 - accuracy: 0.6002 - val_loss: 0.6754 - val_accuracy: 0.5674
Epoch 35/40
2075/2075 [==============================] - 0s 177us/sample - loss: 0.6600 - accuracy: 0.5973 - val_loss: 0.6756 - val_accuracy: 0.5684
Epoch 36/40
2075/2075 [==============================] - 0s 181us/sample - loss: 0.6543 - accuracy: 0.6118 - val_loss: 0.6753 - val_accuracy: 0.5665
Epoch 37/40
2075/2075 [==============================] - 0s 185us/sample - loss: 0.6610 - accuracy: 0.6029 - val_loss: 0.6753 - val_accuracy: 0.5578
Epoch 38/40
2075/2075 [==============================] - 0s 182us/sample - loss: 0.6561 - accuracy: 0.6094 - val_loss: 0.6752 - val_accuracy: 0.5665
Epoch 39/40
2075/2075 [==============================] - 0s 179us/sample - loss: 0.6596 - accuracy: 0.6067 - val_loss: 0.6749 - val_accuracy: 0.5626
Epoch 40/40
2075/2075 [==============================] - 0s 183us/sample - loss: 0.6588 - accuracy: 0.5981 - val_loss: 0.6746 - val_accuracy: 0.5597
```

<img src = "https://user-images.githubusercontent.com/39016197/89111812-6b6f1600-d417-11ea-946d-0b04b8f73c5d.png" width = 600 height = 500>

When looking at just the validation accuracy:

<img src = "https://user-images.githubusercontent.com/39016197/89111821-95283d00-d417-11ea-8989-225798e4d477.png" width = 600 height = 500>

I can also see which weights were the strongest by viewing the history varaible data and running a few quick commands:
```
> max_val_acc  = order(history$metrics$val_accuracy, decreasing = TRUE)
> epoch = max_val_acc[1] 
> epoch
[1] 32
> history$metrics$val_accuracy[epoch]
[1] 0.5741811
```

By using those weights, I can evaluate my model:

```
model %>%
+   evaluate(test, testLabels)
1082/1082 [==============================] - 0s 82us/sample - loss: 0.6924 - accuracy: 0.5721
$loss
[1] 0.6923977

$accuracy
[1] 0.5720887


> prob = model %>%
+   predict_on_batch(test)

> pred = ifelse(model$predict(test)[,1]>model$predict(test)[,2], 0,1)

> table(Predicted = pred, Actual = testtarget)
         Actual
Predicted   0   1
        0 205 164
        1 300 413


> Results = ifelse(pred==1, "W", "L")

> Results = as.factor(Results)

> confusionMatrix(ttest$Result, Results, positive = "W")
Confusion Matrix and Statistics

          Reference
Prediction   L   W
         L 205 300
         W 164 413
                                          
               Accuracy : 0.5712          
                 95% CI : (0.5411, 0.6009)
    No Information Rate : 0.659           
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.1238          
                                          
 Mcnemar's Test P-Value : 3.676e-10       
                                          
            Sensitivity : 0.5792          
            Specificity : 0.5556          
         Pos Pred Value : 0.7158          
         Neg Pred Value : 0.4059          
             Prevalence : 0.6590          
         Detection Rate : 0.3817          
   Detection Prevalence : 0.5333          
      Balanced Accuracy : 0.5674          
                                          
       'Positive' Class : W               
```
Obviously, this is much better than the first nueural network model, but still room for plenty of improvement.

<b> Optimizing the model </b>

There are many techniques to optimize a neural network. To start, it's not a bad idea to add/remove neurons, as well as experimenting with a different number of hidden layers. From there, try out different optimizers and learning rates if you're not sure what will work best for your model. While optimziing my model, I found that it was best to also train the model using the SGD optimizer (stochastic gradient descent) - to help push the accuracy as much as possible. 

K-Fold Cross validation is another technique that was used to push the accuracy to its' peak. I'll note that at this point in my project, I added two additional seasons of training data - bringing my training data total up to 4 seasons. Using a 4-Fold cross validation, I manually partitioned my data so that each season is used as a validation set:

```
train_2015 = train[1:1211,]
train_2016 = train[1212:2423,]
train_2017 = train[2424:3746,]
train_2018 = train[3747:5017,]

label_2015 = trainLabels[1:1211,]
label_2016 = trainLabels[1212:2423,]
label_2017 = trainLabels[2424:3746,]
label_2018 = trainLabels[3747:5017,]


#1st fold
train_1 = rbind(train_2018,train_2017,train_2016,train_2015)
label_1 = rbind(label_2018,label_2017,label_2016,label_2015)

history = model %>%
  fit(train_1,
      label_1,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.24)

#2nd fold
train_2 = rbind(train_2015,train_2017,train_2016,train_2018)
label_2 = rbind(label_2015,label_2017,label_2016,label_2018)

history = model %>%
  fit(train_2,
      label_2,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.25)

#3rd fold
train_3 = rbind(train_2016,train_2018,train_2015,train_2017)
label_3 = rbind(label_2016,label_2018,label_2015,label_2017)

history = model %>%
  fit(train_3,
      label_3,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.26)

#4th fold
train_4 = rbind(train_2017,train_2015,train_2018,train_2016)
label_4 = rbind(label_2017,label_2015,label_2018,label_2016)

history = model %>%
  fit(train_4,
      label_4,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.24)
```

Some of the training sets are different lengths, which explains the different validation splits. If you can recall how my data is formatted, I needed to manually enter stats for the 1st games of season so that I have data to predict game 2+. Since this was a time-consuming approach, I didn't complete this every season - especially not my new training data.

Combining the cross validation with the other optimizing methods allowed me to get the best accuracy possible for my neural network:

```
model = NULL

model <- local({
    input = layer_input(shape = c(44), name = 'main_input')
    
    layer1 = input %>%
        layer_dense(units = 30, activation = "relu")
    
    layer2 = input %>%
        layer_dense(units = 30, activation = "relu")
    
    output = layer_concatenate(c(layer1, layer2)) %>%
        layer_dropout(0.5) %>%
        layer_dense(units = 10, activation = "relu", kernel_regularizer = regularizer_l1(0.001)) %>%
        layer_dense(units = 2, activation = "sigmoid", name = 'main_outout')
    
    keras_model(inputs = input, outputs = output)
})

model %>%
  compile(loss = "binary_crossentropy",
          optimizer = optimizer_adam(lr=.0001),
          metrics = "accuracy")

history = model %>%
  fit(train,
      trainLabels,
      epoch = 300,
      batch_size = 120,
      validation_split = 0.2)
```
<img src = "https://user-images.githubusercontent.com/39016197/90069153-1ea8fc00-dcaf-11ea-90d8-c991b1177b00.png" width = 600 height = 500>
<img src = "https://user-images.githubusercontent.com/39016197/90069193-31bbcc00-dcaf-11ea-9163-6468000f78b1.png" width = 600 height = 500>

```
pred = ifelse(model$predict(test)[,1]>model$predict(test)[,2], 0,1)

> table(Predicted = pred, Actual = testtarget)
         Actual
Predicted   0   1
        0 215 144
        1 290 433

> Results = ifelse(pred==1, "W", "L")

> Results = as.factor(Results)

> confusionMatrix(ttest$Result, Results, positive = "W")
Confusion Matrix and Statistics

          Reference
Prediction   L   W
         L 215 290
         W 144 433
                                         
               Accuracy : 0.5989         
                 95% CI : (0.569, 0.6283)
    No Information Rate : 0.6682         
    P-Value [Acc > NIR] : 1              
                                         
                  Kappa : 0.1794         
                                         
 Mcnemar's Test P-Value : 3.397e-12      
                                         
            Sensitivity : 0.5989         
            Specificity : 0.5989         
         Pos Pred Value : 0.7504         
         Neg Pred Value : 0.4257         
             Prevalence : 0.6682         
         Detection Rate : 0.4002         
   Detection Prevalence : 0.5333         
      Balanced Accuracy : 0.5989         
                                         
       'Positive' Class : W
```

Once the model was initially run, I ran the model again with the different folds and modified the learning rate and batch sizes. Based on the final accuracy, this is my best model yet with this data. Despite exhaustive efforts to push past the 60% threshold, it seems I'm either stuck at a local or global minimum - meaning any changes to my weights results in a lesser accuracy. If it was truly a local minimum, I may be able to increase my accuracy with a higher learning rate but after several attempts with different learning rates, this seems to be the best I can do with my data.

# Troubleshoot

With the above results, what exactly went wrong here? Did I not have the correct variables? Did I not have enough data? Is predicting using a cumulative mean a poor way to approach these predictions? Well, there's one way I can at least narrow down my options - and that's running the predictions with my raw data.

I know what you're thinking, and believe me, I share your concern. Using my raw data means that I'm using real stats from each game - including goals scored! It would be a no-brainer for my model to get 100% accuracy. But what if I removed the obvious stats that would give away the outcome of the game, such as: Goals, Empty Netters, Goals Against, Differential, Points, Wins, etc. In fact, I'll only keep 14 variables that are unrelated, more or less, to the result:


<img src = "https://user-images.githubusercontent.com/39016197/90059220-0b426480-dca0-11ea-8609-9eedcfceaa9d.png" width = 1600 height = 350>

I'll follow the same process as before - scaling and hot-coding my data for my neural network. Using the same API neural network as my optimized model above, but with less neurons, I'm finally able to achieve some significant results:

```
model = NULL

model <- local({
    input = layer_input(shape = c(14), name = 'main_input')
    
    layer1 = input %>%
        layer_dense(units = 10, activation = "relu")
    
    layer2 = input %>%
        layer_dense(units = 10, activation = "relu")
    
    output = layer_concatenate(c(layer1, layer2)) %>%
        layer_dropout(0.5) %>%
        layer_batch_normalization() %>%
        layer_dense(units = 10, activation = "relu") %>%
        layer_dense(units = 2, activation = "sigmoid", name = 'main_outout')
    
    keras_model(inputs = input, outputs = output)
})

# compile
model %>%
  compile(loss = "binary_crossentropy",
          optimizer = optimizer_adam(lr=.0001),
          metrics = "accuracy")

history = model %>%
  fit(train,
      trainLabels,
      epoch = 165,
      batch_size = 100,
      validation_split = 0.2)
```

<img src = "https://user-images.githubusercontent.com/39016197/90059346-40e74d80-dca0-11ea-8b79-abece096a089.png" width = 600 height = 500>

```
 model %>%
+   evaluate(test, testLabels)
1082/1082 [==============================] - 0s 23us/sample - loss: 0.1710 - accuracy: 0.9524
$loss
[1] 0.1709769

$accuracy
[1] 0.9524029

> max_val_acc  = order(history$metrics$val_accuracy, decreasing = TRUE)

> epoch = max_val_acc[1] 

> epoch
[1] 157

> history$metrics$val_accuracy[epoch]
[1] 0.9547206

> prob = model %>%
+   predict_on_batch(test)

> pred = ifelse(model$predict(test)[,1]>model$predict(test)[,2], 0,1)

> table(Predicted = pred, Actual = testtarget)
         Actual
Predicted   0   1
        0 480  24
        1  25 553

> Results = ifelse(pred==1, "W", "L")

> Results = as.factor(Results)

> confusionMatrix(ttest$Result, Results, positive = "W")
Confusion Matrix and Statistics

          Reference
Prediction   L   W
         L 480  25
         W  24 553
                                          
               Accuracy : 0.9547          
                 95% CI : (0.9406, 0.9663)
    No Information Rate : 0.5342          
    P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.909           
                                          
 Mcnemar's Test P-Value : 1               
                                          
            Sensitivity : 0.9567          
            Specificity : 0.9524          
         Pos Pred Value : 0.9584          
         Neg Pred Value : 0.9505          
             Prevalence : 0.5342          
         Detection Rate : 0.5111          
   Detection Prevalence : 0.5333          
      Balanced Accuracy : 0.9546          
                                          
       'Positive' Class : W   
```

As we can see, knowing only 14 of the variable's true values yields a very high accuracy - and this is without even optimizing my model. Based on the validation loss, running additional epochs with more/less neurons and expiermenting with hidden layers can likely get this accuracy into the 98-99% range. 

So what does this mean? My takeaway is that using a cumulative mean to predict each game will only yield roughly a 60% accuracy. When predicting using my method, there needs to be more research involved to accurately predict each variable's values before predicting the outcome of each game. What this means is that, perhaps, there is a prior step required (such as first predicting variable values) when predicting NHL games. While it will be no easy task predicting accurate variable values, such as shots and hits, I've proven that predicting the correct variable values for each game can, in fact, predict the outcome of the game at a reasonable accuracy.

# Conclusion

There can be many approaches to predicting NHL games. With my method, I used team’s cumulative averaged stats to try to predict the outcome of the games. The overall results were not promising, as most hockey fanatics can pick game winners better than a 60% rate. However, when using the right variable values per game, outcomes have been proven to be predicted at a more than reasonable accuracy. The question then remains, how we get to the point where we can predict game variables accurately enough to hit that 90% accuracy threshold when predicting outcomes. At the end of this project, here are some questions that still linger:

1) Will results improve with more training data? 
	- Normally, the answer would be yes. It's always beneficial to have more training data. But when it comes to hockey, I'm not so sure. The NHL is in a constant change of flux - the game is always evolving. NHL players do not play the same style that was played back in the 90s. More training data means reaching back further into NHL history, and once we start pulling data that is old enough, those game stats could cause even more confusion to the model. With that said, I don’t think it would necessarily hurt to add an extra season or two (putting my training total to 6 seasons for predicting one test season), but any more than that will be unnecessary. I’ll note that adding those last 2 seasons (2015-2017) only slightly improved my model – maybe 1% better at the most.

2) What other variables can be added to improve accuracy?
	- It highly depends on how accurate we can first predict the variable values. With the correct variable values, you only need 14 variables (as shown above) to predict at an accuracy better than 95%. However, it may be extremely unlikely that anyone can predict those variable values well enough to get these types of results. For my research, I pulled every variable that was made available at www.hockey-reference.com. If I didn't have time constraints with this project, I would have built another web scraper to pull stats from a different site that would include: Face Off %, Power Play %, Penalty Kill %, Give Aways, Take Aways, etc. I was actually shocked that hockey-reference did not keep track of power plays/penalty kills even though penalties were tracked in the box scored (a more complex formula could have been step up to eventually pull these stats for each game, but at the time my web scraper was built, I didn't think I needed any more variables). I also would have loved to add a few subjective variables, such as health. Since I would need this variable summarized in usable value for game’s prediction (perhaps between 0-1), this would take an incredible amount of research and time  just to add this one variable (knowing which player is out for every team (31) for every one of their games (82) for 5 different seasons and subjectively putting a value on that variable depending on which player is out).

3) Is the neural network the best model to use? 
	- With my data, yes, the neural network seems to be the best model for my data. However, that doesn't mean other models won't be successful at predictions. With the research I've done prior to this project, a few others have tried to predict sport games and have produced similar accuracy to my best model with SVM, etc. I'm not saying their method or data is an apples to apples comparison with my own, but I'm sure there are models that can be just as good, if not better, with the right data. Out of the 3 models I tried, I was most disappointed with the SVM. While I wasn't expecting anything spectaculur with the Support Vector Machine, I was hoping to at least beat the logistic regression. That was not the case, but perhaps that are better methods to predicting SVM than the kernlab package I tried (similar to the issue I had with the neuralnet package when switching over to Keras). I always had high hopes for the neural network since I knew that I would be able to use all of my variables (which I still believe are essential to predicting) and the ability for the neural network to be modified and re-trained.
	
Overall, this was a great learning project for both sports data and machine learning. With my analysis, I was able to find patterns in certain variables, such as the min-max trend I found with blocked shots, that gave me a new perspective on how important some variables can be. Finding out that the strongest correlations on my result variable were points, differential, wins, and goals weren't exactly a surprise, but it was fascinating that the Away team stats in those categories seemed to have impacted the results more than the home team (despite the result being a direct product of the home team). The analysis also helped me put less stock into certain variables, such as corsi and offensive zone %, that are more highly regarded in the stats community than I'm now willing to give them credit for. In regards to the machine learning section, I had a blast experimenting and trying to find ways to optimize my models - it certainly does help when you're also interested with the subject at hand (sport predictions). While I didn't get the results that I necessarily wanted, this opened the door for future improvement and I can't wait to build onto what I've learned. By the time you're reading this, I'm hoping you've learned something too and that my model accuracy has much improved!
