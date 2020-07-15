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

# Re-Structuring the Data to reflect Cumulative means

I'll be the first to admit when I'm wrong about something. If you looked at part 1 of this project, you'll notice that most of my data was structured to use a cumulative sum - which meant that stats were aggregating for each team as the season went on. Unfortunately, it didn't take my long in my analysis to notice the problem with that. So before I jump into anything else, let me show you the issue I noticed:

``` 
> cor(ttrain$Home_Goals, ttrain$Home_BS)
[1] 0.9595307
> plot(ttrain$Home_Goals ~ ttrain$Home_BS)
```

<img src = "https://user-images.githubusercontent.com/39016197/87491348-240a1c80-c605-11ea-8040-f6be85cdcbf7.png" width = 470 height = 250>

The plot above shows the plot relationship between home goals and blocked shots. Notice anything? Typically, we wouldn't expect to see much of a relationship between these two variables - goals is more known to have a relationship with shots on goal. You can't score if you don't shoot the puck. However, since I used a cumulative sum for these stats, there is now a huge linear relationship between all of my predictor variables (since all of my stats are increasing with each other for every game). In the data science world, this is known as collinearity -  and a high amount of that in this case.

To fix this problem, I had to go back to my raw data and apply the same scripting methods that I applied the first time around. However, this time I used a cumulative mean for all of my variables. With the new formatted data, we finally get the results that we would expect to see between these two variables: 

```
> cor(dtrain$Home_Goals, dtrain$Home_BS)
[1] 0.06437274
> plot(dtrain$Home_Goals ~ dtrain$Home_BS)
```

<img src = "https://user-images.githubusercontent.com/39016197/87491760-3173d680-c606-11ea-8902-cd9be499187a.png" width = 470 height = 250>

A correlation value of 0.064 makes much more sense than 0.959 between goals and blocked shots. We're finally ready to move on to the analysis. 

# Importing the Data

As we can see from the above, my machine learning data set will be imported into R as "Dtain" and "Dtest" (I've always gotten into the habit of using dt for 'data table' and because my initials happen to be dt). However, I also think there is a lot of value of also looking at the raw data - which is the data that shows each game's stats and results before manipulated into an aggregated data set. For this raw data, i'll import both my training and test set and combine both sets into one data frame:

```
dtrain = read_excel("Training Final CM.xlsx", col_names = TRUE)
dtest = read_excel("Test Final CM.xlsx", col_names = TRUE)

raw_2018 = read_excel("2017-2018 Training Base.xlsx", col_names = TRUE)
raw_2019 = read_excel("2018-2019 Training Base.xlsx", col_names = TRUE)
raw_2020 = read_excel("2019-2020 Test Base.xlsx", col_names = TRUE)

df = rbind(raw_2018, raw_2019, raw_2020)
```

Once I have combined my training set with my entire test set, I'll create new set and remove the Results column from my test set - as this will be the variable we will be predicting. I want to keep my Results variable in my original test set so that I can compare my machine learning results later on. I'll also remove the date and team names from my training and test set - as I will not need these variables when I make my predicitons. Finally, I need to make sure that my Prediction variable is set as a factor:

```
test = dtest[,-4]


dtrain = dtrain[, -c(1:3)]
dtest = dtest[, -c(1:3)]

dtrain$Result = as.factor(dtrain$Result)
dtest$Result = as.factor(dtest$Result)
```

For 


# Analysis

Before conducting analysis, it's always a good idea to view the data and get familiar with its' contents before. Since I'm using R, it's also a good idea to attach your data set so that variables are much easier to call. Once this has been completed, one of the first commands you should run with a new data frame is the structure and summary functions. Since my teams and results loaded in as characters, I'll first change these variables to factors. It's also very important to deal with any missing values. I did already fix all of my missing values in Part 1, but it still can't hurt to check just in case:

```
table(is.na(dt))

 FALSE 
181200 
```



Once the change has been made, here are the corresponding strucutre and summary commands:

```
> str(dt)

Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	3624 obs. of  50 variables:
 $ Date        : POSIXct, format: "2017-10-04" ...
 $ Home        : Factor w/ 31 levels "Anaheim Ducks",..: 12 24 23 31 1 3 4 7 20 11 ...
 $ Away        : Factor w/ 31 levels "Anaheim Ducks",..: 5 22 25 27 2 17 16 23 8 15 ...
 $ Result      : Factor w/ 2 levels "L","W": 2 1 1 1 2 2 1 2 1 2 ...
 $ Home_Goals  : num  3 2 3 3 2 3 2 3 3 2 ...
 $ Away_Goals  : num  2 2 3 3 2 3 3 4 2 3 ...
 $ Home_PIM    : num  9 8 8 10 11 10 9 7 7 8 ...
 $ Away_PIM    : num  11 10 9 10 10 10 9 8 9 8 ...
 $ Home_Shots  : num  31 30 33 30 30 33 30 30 30 28 ...
 $ Away_Shots  : num  29 32 28 32 28 31 30 33 29 31 ...
 $ Home_Corsi  : num  50 51.1 50.1 49.3 49.7 54.7 47.2 50.4 48 48.2 ...
 $ Away_Corsi  : num  50.5 51.1 50.2 50.4 45 51.4 52.5 56.2 48.5 49.3 ...
 $ Home_OFS    : num  50.9 52.3 52.9 49.3 48.4 53.3 47.4 51.8 48.7 50.9 ...
 $ Away_OFS    : num  47.7 52.8 48.3 49.1 43.5 50.1 48.4 55.1 48.5 49.7 ...
 $ Home_HT     : num  25 18 25 23 26 21 22 14 20 20 ...
 $ Away_HT     : num  19 25 21 23 25 20 22 27 20 15 ...
 $ Home_BS     : num  16 16 16 14 14 13 13 15 15 13 ...
 $ Away_BS     : num  13 16 14 14 14 13 14 10 13 14 ...
 $ Home_EN     : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_EN     : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Home_GA     : num  2 2 3 3 2 2 3 2 2 3 ...
 $ Away_GA     : num  2 3 2 3 3 3 2 5 3 2 ...
 $ Home_PIM_A  : num  5 5 5 5 5 5 4 5 5 5 ...
 $ Away_PIM_A  : num  5 5 5 5 4 5 4 10 5 5 ...
 $ Home_SA     : num  29 28 32 31 30 27 34 31 30 31 ...
 $ Away_SA     : num  29 28 28 33 34 30 30 34 31 30 ...
 $ Home_Corsi_A: num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_Corsi_A: num  0 0 0 0 0 0 0 43.8 0 0 ...
 $ Home_OFS_A  : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_OFS_A  : num  0 0 0 0 0 0 0 46.7 0 0 ...
 $ Home_HT_A   : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_HT_A   : num  0 0 0 0 0 0 0 36 0 0 ...
 $ Home_BS_A   : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_BS_A   : num  0 0 0 0 0 0 0 19 0 0 ...
 $ Home_EN_A   : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_EN_A   : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Home_GP     : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_GP     : num  0 0 0 0 0 0 0 1 0 0 ...
 $ Home_W      : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_W      : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Home_P      : num  1 1 2 1 1 1 1 2 1 1 ...
 $ Away_P      : num  1 1 1 1 0 1 1 1 0 1 ...
 $ Home_DIF    : num  0 0 0 0 0 0 0 0 0 0 ...
 $ Away_DIF    : num  0 0 0 0 0 0 0 -1 0 0 ...
 $ Home_SV     : num  0.914 0.912 0.914 0.9 0.919 0.905 0.918 0.918 0.912 0.903 ...
 $ Away_SV     : num  0.907 0.901 0.907 0.912 0.908 0.911 0.918 0.853 0.894 0.916 ...
 $ Home_SH     : num  0.095 0.089 0.101 0.1 0.091 0.085 0.08 0.096 0.104 0.085 ...
 $ Away_SH     : num  0.093 0.082 0.1 0.096 0.084 0.093 0.091 0.121 0.072 0.104 ...
 $ Home_PDO    : num  1.009 1.003 1.01 0.999 1.007 ...
 $ Away_PDO    : num  0.997 0.985 1.007 1.005 0.998 ...
 
> summary(dt)

      Date                                       Home     
 Min.   :2017-10-04 00:00:00   Detroit Red Wings   : 119  
 1st Qu.:2018-02-18 00:00:00   Montreal Canadiens  : 119  
 Median :2018-12-20 12:00:00   Ottawa Senators     : 119  
 Mean   :2018-12-13 11:09:08   Vegas Golden Knights: 119  
 3rd Qu.:2019-10-27 06:00:00   Winnipeg Jets       : 119  
 Max.   :2020-03-11 00:00:00   Anaheim Ducks       : 118  
                               (Other)             :2911  
                 Away      Result     Home_Goals   
 Arizona Coyotes   : 119   L:1649   Min.   :  0.0  
 Calgary Flames    : 119   W:1975   1st Qu.: 56.0  
 Colorado Avalanche: 119            Median :113.0  
 Edmonton Oilers   : 119            Mean   :115.1  
 Chicago Blackhawks: 118            3rd Qu.:170.0  
 Los Angeles Kings : 118            Max.   :300.0  
 (Other)           :2912                           
   Away_Goals       Home_PIM        Away_PIM       Home_Shots  
 Min.   :  0.0   Min.   :  2.0   Min.   :  4.0   Min.   :  20  
 1st Qu.: 57.0   1st Qu.:167.8   1st Qu.:169.0   1st Qu.: 608  
 Median :113.0   Median :334.0   Median :334.5   Median :1210  
 Mean   :115.4   Mean   :338.7   Mean   :338.4   Mean   :1223  
 3rd Qu.:171.0   3rd Qu.:494.2   3rd Qu.:495.0   3rd Qu.:1828  
 Max.   :313.0   Max.   :914.0   Max.   :904.0   Max.   :2800  
                                                               
   Away_Shots       Home_Corsi      Away_Corsi   
 Min.   :  15.0   Min.   :30.50   Min.   :31.80  
 1st Qu.: 591.8   1st Qu.:48.20   1st Qu.:48.25  
 Median :1210.5   Median :49.83   Median :49.90  
 Mean   :1223.7   Mean   :49.97   Mean   :50.03  
 3rd Qu.:1827.5   3rd Qu.:51.67   3rd Qu.:51.77  
 Max.   :2796.0   Max.   :68.20   Max.   :69.50  
                                                 
    Home_OFS        Away_OFS        Home_HT      
 Min.   :27.70   Min.   :30.60   Min.   :  12.0  
 1st Qu.:48.68   1st Qu.:48.73   1st Qu.: 393.0  
 Median :50.72   Median :50.73   Median : 806.0  
 Mean   :50.81   Mean   :50.85   Mean   : 839.6  
 3rd Qu.:52.70   3rd Qu.:52.65   3rd Qu.:1238.2  
 Max.   :68.20   Max.   :72.10   Max.   :2339.0  
                                                 
    Away_HT          Home_BS          Away_BS      
 Min.   :  12.0   Min.   :   6.0   Min.   :   6.0  
 1st Qu.: 400.8   1st Qu.: 269.8   1st Qu.: 269.0  
 Median : 817.5   Median : 542.0   Median : 540.5  
 Mean   : 840.9   Mean   : 552.5   Mean   : 553.0  
 3rd Qu.:1241.0   3rd Qu.: 821.0   3rd Qu.: 820.0  
 Max.   :2294.0   Max.   :1375.0   Max.   :1359.0  
                                                   
    Home_EN          Away_EN          Home_GA         Away_GA   
 Min.   : 0.000   Min.   : 0.000   Min.   :  0.0   Min.   :  0  
 1st Qu.: 2.000   1st Qu.: 3.000   1st Qu.: 57.0   1st Qu.: 57  
 Median : 5.000   Median : 5.000   Median :114.0   Median :112  
 Mean   : 5.831   Mean   : 5.912   Mean   :115.5   Mean   :115  
 3rd Qu.: 9.000   3rd Qu.: 9.000   3rd Qu.:171.0   3rd Qu.:171  
 Max.   :21.000   Max.   :22.000   Max.   :295.0   Max.   :290  
                                                                
   Home_PIM_A      Away_PIM_A       Home_SA      
 Min.   :  2.0   Min.   :  4.0   Min.   :  19.0  
 1st Qu.:170.0   1st Qu.:171.0   1st Qu.: 596.8  
 Median :335.0   Median :337.0   Median :1212.0  
 Mean   :338.5   Mean   :338.6   Mean   :1223.4  
 3rd Qu.:498.0   3rd Qu.:494.2   3rd Qu.:1822.5  
 Max.   :949.0   Max.   :955.0   Max.   :2904.0  
                                                 
    Away_SA        Home_Corsi_A    Away_Corsi_A  
 Min.   :  15.0   Min.   : 0.00   Min.   : 0.00  
 1st Qu.: 602.8   1st Qu.:48.24   1st Qu.:48.17  
 Median :1214.0   Median :50.11   Median :50.05  
 Mean   :1223.1   Mean   :49.33   Mean   :49.39  
 3rd Qu.:1823.0   3rd Qu.:51.77   3rd Qu.:51.72  
 Max.   :2879.0   Max.   :69.50   Max.   :68.20  
                                                 
   Home_OFS_A      Away_OFS_A      Home_HT_A     
 Min.   : 0.00   Min.   : 0.00   Min.   :   0.0  
 1st Qu.:48.89   1st Qu.:48.91   1st Qu.: 396.0  
 Median :50.84   Median :50.85   Median : 814.5  
 Mean   :50.15   Mean   :50.24   Mean   : 838.8  
 3rd Qu.:53.01   3rd Qu.:52.96   3rd Qu.:1240.0  
 Max.   :73.40   Max.   :72.10   Max.   :2408.0  
                                                 
   Away_HT_A        Home_BS_A        Away_BS_A     
 Min.   :   0.0   Min.   :   0.0   Min.   :   0.0  
 1st Qu.: 398.8   1st Qu.: 266.0   1st Qu.: 265.0  
 Median : 816.0   Median : 538.5   Median : 541.5  
 Mean   : 840.8   Mean   : 552.0   Mean   : 552.7  
 3rd Qu.:1247.2   3rd Qu.: 820.2   3rd Qu.: 818.0  
 Max.   :2367.0   Max.   :1550.0   Max.   :1569.0  
                                                   
   Home_EN_A        Away_EN_A         Home_GP     
 Min.   : 0.000   Min.   : 0.000   Min.   : 0.00  
 1st Qu.: 3.000   1st Qu.: 3.000   1st Qu.:19.00  
 Median : 5.000   Median : 5.000   Median :38.50  
 Mean   : 5.883   Mean   : 5.869   Mean   :38.68  
 3rd Qu.: 9.000   3rd Qu.: 9.000   3rd Qu.:58.00  
 Max.   :19.000   Max.   :21.000   Max.   :81.00  
                                                  
    Away_GP          Home_W          Away_W         Home_P      
 Min.   : 0.00   Min.   : 0.00   Min.   : 0.0   Min.   :  0.00  
 1st Qu.:19.00   1st Qu.: 9.00   1st Qu.: 9.0   1st Qu.: 21.00  
 Median :38.00   Median :18.00   Median :19.0   Median : 42.00  
 Mean   :38.68   Mean   :19.29   Mean   :19.4   Mean   : 43.06  
 3rd Qu.:58.00   3rd Qu.:28.00   3rd Qu.:28.0   3rd Qu.: 63.00  
 Max.   :81.00   Max.   :59.00   Max.   :61.0   Max.   :122.00  
                                                                
     Away_P          Home_DIF            Away_DIF        
 Min.   :  0.00   Min.   :-121.0000   Min.   :-120.0000  
 1st Qu.: 21.00   1st Qu.: -11.0000   1st Qu.: -11.0000  
 Median : 42.00   Median :   1.0000   Median :   1.0000  
 Mean   : 43.29   Mean   :  -0.3226   Mean   :   0.3455  
 3rd Qu.: 63.00   3rd Qu.:  11.0000   3rd Qu.:  12.0000  
 Max.   :126.00   Max.   :  95.0000   Max.   :  95.0000  
                                                         
    Home_SV          Away_SV          Home_SH       
 Min.   :0.8000   Min.   :0.7140   Min.   :0.00000  
 1st Qu.:0.9020   1st Qu.:0.9020   1st Qu.:0.08600  
 Median :0.9090   Median :0.9100   Median :0.09400  
 Mean   :0.9094   Mean   :0.9096   Mean   :0.09483  
 3rd Qu.:0.9170   3rd Qu.:0.9170   3rd Qu.:0.10400  
 Max.   :1.0000   Max.   :1.0000   Max.   :0.22700  
                                                    
    Away_SH           Home_PDO        Away_PDO    
 Min.   :0.00000   Min.   :0.824   Min.   :0.775  
 1st Qu.:0.08600   1st Qu.:0.993   1st Qu.:0.993  
 Median :0.09400   Median :1.005   Median :1.005  
 Mean   :0.09522   Mean   :1.004   Mean   :1.005  
 3rd Qu.:0.10400   3rd Qu.:1.017   3rd Qu.:1.018  
 Max.   :0.33300   Max.   :1.193   Max.   :1.273 
```
As we can see....

