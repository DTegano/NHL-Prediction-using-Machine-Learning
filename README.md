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

Some interesting observations/notes from the 3 seasons worth of data: <p><p/>
1. Not all of the teams played the same amount of home and away games due to the quarantine stoppage.
2. The highest amount of goals scored in a NHL game are 10 (home team) and 9 (away team)
3. The highest corsi % we have seen in any game is 74.8%, while the highest amount of shooting % in any game is a whopping 41%.
4. 87 Penalty minutes and 67 hits is the most we've seen in any game over the last 3 years.
5. Neither the home or away team has recorded more than 33 blocked shots in a single game.

Now, I'll look at a few different variables to see how much of an impact they have of the outcome of the game when the variable is maximized and minimized. I'll spoil goals for you - the teams that scored 10 and 9 goals won their games, and the teams that scored 0 goals <u> almost </u> lost every game (I believe there were 1 or 2 shootout victories at 0-0).

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


Aside from goals, shots can be considered the next greatest factor that can influence the result of a game. To have an impact on the outcome, we would expect that the teams with the maximum shot count would win their games and the teams with the lowest shot counts would lose their games. However, the results above indicate that at the highest and lowest shot totals, the teams generating those shots lost all of the games (note that "W" indicates a home win and "L" indicates an away win). The Carolina Hurrincanes recorded the most shots in single game - putting up 60 shots in November 2017. However, they lost that game - as did the Washington Capitals when they put up 58 shots as the home team in March 2019.


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

For hits, it can go either way. Either the maximum amount of hits per game can cause a team to dominate their opponent physically, or too many hits in a game can distract the players from what's important - scoring goals and winning the game. As we can see at the maximum amount of hits (54 for the home team and 67 for the away team), the teams with those hit amounts won their games. With that said, would we expect to see teams with much less hits lose their games? Well, that doesn't seem to be the case here with hits. When we look at the lowest hit totals (4 for the home team and 2 for the away team), the home team actually won all of their games at 4 hits while the away teams split their games (1 win, 1 loss) at 2 hits. With this mixed bag of results, we can't yet see the impact that hits have on the result.

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

 Have we finally got some meaningful results when looking at the max and min values? Blocked shots has always been an underrated hockey factor in my mind and teams don't get enough credit when they do a good job of getting bodies in front of pucks. When we look at the max values (33 for both home and away), the teams won all of their games. When we look at the min values (2 for home and away), the teams lost of all of their games. Based on these facts, I'll assume that the amount of blocked shots can, in fact, influence the outcome of a hockey game.

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

Corsi has been a key stat in the NHL over the last decade when looking at team possession. However, at the max and min values, Corsi is a split decision. When the Carolina Hurrincanes dominated at a 74.8% Corsi, they won while the defending team (Tampa Lightning - 25.2%) lost that game. However, when Carolina also put up a League best 74.1% as the Away team, they lost to the Columbus Blue Jackets - who only held a Corsi % of 25.9%. With these results, we cannot yet confirm the impact of corsi on the results.

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

Finally, I'll conclude this max-min comparison with the often overlooked Offensive Zone Start %. While we would expect this to have a positive impact on the outcome of a game, would this actually have a negative impact (crazy, I know, but run with me for a second)? The highest Offensize Zone start % on the last 3 years was 86.5% for any home team and 82.8% for any away team. The teams that put  up these numbers, the Columbus Blue Jackets and Nashville Predators, lost both of these games to the teams who put up the lowest Offensize Start %s over the last 3 years.
