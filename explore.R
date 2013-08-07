library(Hmisc)
setwd('C:\\Users\\alez\\Desktop\\octave\\kaggle\\yelp')

train = read.csv('some_stem_fea.csv')
train$date = as.Date.character(train$date, '%Y-%m-%d')
train$since = as.Date.character('2013-01-19', '%Y-%m-%d') - train$date
train$since = as.numeric(train$since)
train = read.csv('review_.csv')

names(train)

lm1 = lm(votes_useful ~ review_len+stem_len+stem_unique_len+stem_unique_len_ratio+
         like+know+even+think+review+               
         well+thing+peopl+make+chees+                
         open+time+right+could+take+since, data=train  )

pred = lm1$fitted.values
sqrt( mean((log(train$votes_useful+1)-log(pred+1))^2) )

step(lm1)

lm1 = glm(formula = votes_useful ~ log(stem_len+6) + log(stem_unique_len+5) + like + 
     know + even + think + review + well + peopl + make + chees + 
     open + time + right + take + since, data = train, family = "poisson")
lm1 = glm(formula = votes_useful ~ log(stem_len+6) +  log(stem_unique_len+5) + like + 
     know + think + review + well + chees + open + time + right + 
     take + since, data = train, family = "poisson")

pred = lm1$fitted.values
sqrt( mean((log(train$votes_useful+1)-log(pred+1))^2) )


lm1 = glm(votes_useful ~ review_len + stem_len
          + stem_unique_len + stem_unique_len_ratio
          + date
          , data = train, family = "poisson")
pred = lm1$fitted.values
sqrt( mean((log(train$votes_useful+1)-log(pred+1))^2) )

