import pandas as pd
import numpy as np
import time
import datetime

## CONFIG
process_test = True

## BUSINESS DATA
business = pd.read_csv('Data\\yelp_training_set_business.csv').set_index('business_id')
business = business.drop(['city','full_address', #'latitude','longitude', # keep for location grouping
                      'name','neighborhoods','state','type'],axis=1)
if process_test:
    business_test = pd.read_csv('Data\\yelp_test_set_business.csv').set_index('business_id')
    business_test = business_test.drop(['city','full_address', #'latitude','longitude', # keep for location grouping
                      'name','neighborhoods','state','type'],axis=1)

## FEATURE EXTRACTION for CATEGORIES
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(tokenizer=lambda text: text.split(','))

cat_fea = vect.fit_transform(business['categories'].fillna(''))
cat_fea = cat_fea.todense()
idx_max_1 = cat_fea > 1
cat_fea[idx_max_1] = 1
if process_test:    ## PROCESS TEST SET ON TRAINED CountVectorizer
    cat_fea_test = vect.transform(business_test['categories'].fillna(''))
    cat_fea_test = cat_fea_test.todense()
    idx_max_1 = cat_fea_test > 1
    cat_fea_test[idx_max_1] = 1


from sklearn.cluster import MiniBatchKMeans
## CATEGORY CLUSTERS
#  Based on the category extracted before, the idea is to create a n clusters to
#  aggregate set of similar categories
for esti in (20,35,50,60,70,80,90,100,110,125):
    km = MiniBatchKMeans(n_clusters=esti, random_state=1377, init_size=esti*10)

    print "fitting "+str(esti)+" clusters - category"
    init_time = time.time()
    km.fit(cat_fea)
    print (time.time()-init_time)/60

    business['cat_clust_'+str(esti)] = km.predict(cat_fea)
    if process_test:
        business_test['cat_clust_'+str(esti)] = km.predict(cat_fea_test)
    

## LOCATION CLUSTERS
#  Location cluster, even if only in Phoenix area this might spot interesting patterns         
for esti in (5,10,15,20,25,30,40):
    km = MiniBatchKMeans(n_clusters=esti, random_state=1377, init_size=esti*100)

    print "fitting "+str(esti)+" clusters - location"
    init_time = time.time()
    km.fit(business.ix[:,['latitude','longitude']])
    print (time.time()-init_time)/60

    business['loc_clust_'+str(esti)] = km.predict(business.ix[:,['latitude','longitude']])
    if process_test:
        business_test['loc_clust_'+str(esti)] = km.predict(business_test.ix[:,['latitude','longitude']])


## LOADING CHECKIN DATA     
checkin = pd.read_csv('Data\\yelp_training_set_checkin.csv').set_index('business_id')
checkin = checkin.drop(['type'],axis=1).fillna(0)
if process_test:
    checkin_test = pd.read_csv('Data\\yelp_test_set_checkin.csv').set_index('business_id')
    checkin_test = checkin_test.drop(['type'],axis=1).fillna(0)

for i in range(7):
    checkin['check_d'+str(i)] = checkin.ix[:,i*24:(i+1)*24].apply(sum,axis=1)
    if process_test:
        checkin_test['check_d'+str(i)] = checkin_test.ix[:,i*24:(i+1)*24].apply(sum,axis=1)

checkin = checkin.ix[:,168:]
checkin['check_all'] = checkin.ix[:,:].apply(sum,axis=1)
if process_test:
    checkin_test = checkin_test.ix[:,168:]
    checkin_test['check_all'] = checkin_test.ix[:,:].apply(sum,axis=1)

## APPENDING CHECKING DATA to BUSINESSES
business = business.join(checkin).fillna(0)

print "printing file"
business.to_csv('DataProcessed\\business_fea.csv')
if process_test:
    business_test = business_test.join(checkin_test).fillna(0)
    business.append(business_test).to_csv('DataProcessed\\test_and_train_business_fea.csv')
    
    
    
    
