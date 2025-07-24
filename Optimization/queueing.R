#Clear environment
rm(list=ls())

#load mission critical packages
library(tidyverse)
library(reshape2)
library(RMySQL)
library(lubridate)
library(queueing)

db <- RMySQL::dbConnect( RMySQL::MySQL(), dbname='pharm_db', username='root', password='root')
dbListTables(db)

res<-dbSendQuery(db,'select * from rxdata')

data<-fetch(res, n = -1)

dbDisconnect(db)

#--modify data-------------------------------------------------------------------------------------------------
#data$temp <- data$DrugCode
# change the type 
data$Date <- mdy(data$Date)
data$Method <- as.factor(data$Method)
data$Priority <- as.factor(data$Priority)
data$CommonDrugName <- as.factor(data$CommonDrugName)
data$Drug <- as.factor(data$Drug)
data$Unit <- as.factor(data$Unit)
data$Hospital <- as.factor(data$Hospital)
data$GenericNameCode <- as.factor(data$GenericNameCode)
data$`DEA Status` <- as.factor(data$`DEA Status`)
data$DrugCode <- as.factor(data$DrugCode)
data$tVerify <- as.numeric(data$tVerify)
data$DrugClass <- as.factor(data$DrugClass)
data$MajorDrugClass <- as.factor(data$MajorDrugClass)
data$OrderComplete <- ymd_hms(data$OrderComplete)

# Create "Hour" and "minutes" column
data$Time <- as.numeric(data$Time)
data %>% mutate(Hour = floor(Time/100)) -> data
data %>% mutate(Minutes = Time%%100) -> data

# Create "dayNight"
data <- data %>%
        mutate(dayNight = ifelse(730 <= Time & Time <= 1830, "Day", "Night")) #Night hours selected based on drop down to lowest staff hours

# Create "day" and "wday"
data %>% mutate(day=weekdays(Date)) %>%
        mutate(wday = ifelse(day %in% c("Saturday", "Sunday"), "weekend", "weekday")) -> data
data$day <- as.factor(data$day)
data$wday <- as.factor(data$wday)
data %>% select(-AllInfo) -> data

# Remove auto verified orders
ori_data <- data
data <- data[!is.na(data$tVerify),]

#--Explore------------------------------------------------------------------
nrow(data)
#[1] 3520471
length(unique(data$Drug))
#[1] 23397
length(unique(data$CommonDrugName))
#[1] 1859

data %>%
        filter(Date >= as.Date('2019-05-01')) %>%
        filter(Hospital == 'RMC') -> data_RMC
sd(data_RMC$tVerify)
#[1] 24.71035
hist(data_RMC$tVerify,breaks=seq(0,10000,1),xlim = c(0,100),freq = TRUE)

# extract STAT orders and remove outliers
data_RMC %>% filter(Priority == 'STAT') -> data_RMCSTAT
sd(data_RMCSTAT$tVerify)
#[1] 22.60527
#nrow(subset(data_RMCSTAT,data_RMCSTAT$tVerify>(mean(data_RMCSTAT$tVerify) + 6*sd(data_RMCSTAT$tVerify))))
#[1] 206
#nrow(subset(data_RMCSTAT,data_RMCSTAT$tVerify>(mean(data_RMCSTAT$tVerify) + 6*sd(data_RMCSTAT$tVerify))))/nrow(data_RMCSTAT)
#[1] 0.002795533
data_RMCSTAT %>%
        filter(tVerify <= (mean(tVerify) + 6*sd(tVerify)) & tVerify >= (mean(tVerify) - 6*sd(tVerify))) ->data_RMCSTAT
sd(data_RMCSTAT$tVerify)
#[1] 9.018625

# extract Routine orders and remove outliers
data_RMC %>% filter(Priority == 'Routine') -> data_RMCRoutine
sd(data_RMCRoutine$tVerify)
#[1] 24.70922
#nrow(subset(data_RMCRoutine,data_RMCRoutine$tVerify>(mean(data_RMCRoutine$tVerify) + 6*sd(data_RMCRoutine$tVerify))))
#[1] 4649
#nrow(subset(data_RMCRoutine,data_RMCRoutine$tVerify>(mean(data_RMCRoutine$tVerify) + 6*sd(data_RMCRoutine$tVerify))))/nrow(data_RMCRoutine)
#[1] 0.003642195
data_RMCRoutine %>%
        filter(tVerify <= (mean(tVerify) + 6*sd(tVerify)) & tVerify >= (mean(tVerify) - 6*sd(tVerify))) ->data_RMCRoutine
sd(data_RMCRoutine$tVerify)
#[1] 15.09898

# STAT: filter by Unit
data_RMCSTAT %>% filter(Unit=='RMC EMERGENCY ROOM') -> data_RMCSTAT_e
sd(data_RMCSTAT_e$tVerify)
#[1] 9.680943
data_RMCSTAT %>% filter(Unit=='RMC OPERATING ROOM') -> data_RMCSTAT_op
sd(data_RMCSTAT_op$tVerify)
#[1] 5.594
data_RMCSTAT %>% filter(Unit!='RMC OPERATING ROOM') %>% filter(Unit!='RMC EMERGENCY ROOM') -> data_RMCSTAT_other
sd(data_RMCSTAT_other$tVerify)
#[1] 7.495409

# STAT: filter by "VANCOCIN"
nrow(subset(data_RMCSTAT,data_RMCSTAT$CommonDrugName == 'VANCOCIN'))
#[1] 2264
#RMC EMERGENCY ROOM: 2189
#RMC OPERATING ROOM:4
data_RMCSTAT_e %>% filter(CommonDrugName == 'VANCOCIN') -> data_RMCSTAT_e_VAN
sd(data_RMCSTAT_e_VAN$tVerify)
#[1] 20.15995
data_RMCSTAT_e %>% filter(CommonDrugName != 'VANCOCIN') -> data_RMCSTAT_e_noVAN
sd(data_RMCSTAT_e_noVAN$tVerify)
#[1] 8.441751
#nrow(data_RMCSTAT_e_noVAN)
#[1] 38241

# data_RMCSTAT_e_noVAN %>% filter(wday == 'weekday') -> data_RMCSTAT_e_wday
# sd(data_RMCSTAT_e_wday$tVerify)
# #[1] 8.37536
# data_RMCSTAT_e_noVAN %>% filter(wday == 'weekend') -> data_RMCSTAT_e_wend
# sd(data_RMCSTAT_e_wend$tVerify)
# #[1] 8.616501
# 
# data_RMCSTAT_e_noVAN %>% filter(day == 'Sunday') -> data_RMCSTAT_e_wday
# sd(data_RMCSTAT_e_wday$tVerify)
# #[1] 8.20978
# data_RMCSTAT_e_noVAN %>% filter(day != 'Sunday') -> data_RMCSTAT_e_wend
# sd(data_RMCSTAT_e_wend$tVerify)
# #[1] 8.477324

# data_RMCSTAT_e_noVAN %>% filter(dayNight == 'Night') -> data_RMCSTAT_e_noVAN_n
# sd(data_RMCSTAT_e_noVAN_n$tVerify)
# #[1] 8.884003
# data_RMCSTAT_e_noVAN %>% filter(dayNight == 'Day') -> data_RMCSTAT_e_noVAN_d
# sd(data_RMCSTAT_e_noVAN_d$tVerify)
# #[1] 7.977101

# Routine: filter by Unit
data_RMCRoutine %>% filter(Unit=='RMC EMERGENCY ROOM') -> data_RMCRoutine_e
sd(data_RMCRoutine_e$tVerify)
#[1] 21.11246
data_RMCRoutine %>% filter(Unit=='RMC OPERATING ROOM') -> data_RMCRoutine_op
sd(data_RMCRoutine_op$tVerify)
#[1] 28.14687
data_RMCRoutine %>% filter(Unit!='RMC OPERATING ROOM') %>% filter(Unit!='RMC EMERGENCY ROOM') -> data_RMCRoutine_other
sd(data_RMCRoutine_other$tVerify)
#[1] 14.74264

# data_RMCRoutine_e %>% filter(wday == 'weekday') -> data_RMCRoutine_e_wday
# sd(data_RMCRoutine_e_wday$tVerify)
# #[1] 21.75725
# data_RMCRoutine_e %>% filter(wday == 'weekend') -> data_RMCRoutine_e_wend
# sd(data_RMCRoutine_e_wend$tVerify)
# #[1] 18.1559
# 
# data_RMCRoutine_e %>% filter(day == 'Monday' | day == 'Tuesday') -> data_RMCRoutine_e_MT
# sd(data_RMCRoutine_e_MT$tVerify)
# #[1] 21.66924
# data_RMCRoutine_e %>% filter(day != 'Monday' & day != 'Tuesday') -> data_RMCRoutine_e_nMT
# sd(data_RMCRoutine_e_nMT$tVerify)
# #[1] 20.73798
# 
# data_RMCRoutine_e %>% filter(CommonDrugName == 'LOVENOX' | day == 'GLUTOSE') -> data_RMCRoutine_e_LG
# sd(data_RMCRoutine_e_LG$tVerify)
# #[1] 23.59041
# data_RMCRoutine_e %>% filter(CommonDrugName != 'LOVENOX' & day != 'GLUTOSE') -> data_RMCRoutine_e_nLG
# sd(data_RMCRoutine_e_nLG$tVerify)
# #[1] 22.02856

# data_RMCRoutine_op %>% filter(wday == 'weekday') -> data_RMCRoutine_op_wday
# sd(data_RMCRoutine_op_wday$tVerify)
# #[1] 28.6025
# data_RMCRoutine_op %>% filter(wday == 'weekend') -> data_RMCRoutine_op_wend
# sd(data_RMCRoutine_op_wend$tVerify)
# #[1] 19.69969
# 
# data_RMCRoutine_other %>% filter(wday == 'weekday') -> data_RMCRoutine_other_wday
# sd(data_RMCRoutine_other_wday$tVerify)
# #[1] 14.96456
# data_RMCRoutine_other %>% filter(wday == 'weekend') -> data_RMCRoutine_other_wend
# sd(data_RMCRoutine_other_wend$tVerify)
# #[1] 13.85438


#---Calculate lambda--------------------------------------------------------------------------------------------
date <- sort(unique(data_RMCRoutine$Date))
lambda_R <- as.data.frame(date)
for(i in 0:23){
        for(k in 0:1){
                n <- c()
                for(j in 1:nrow(lambda_R)){
                        if(k==0){
                                data_RMCRoutine %>% filter(Hour==i) %>%
                                        filter(Minutes < (k+1)*30) %>%
                                        filter(Date==lambda_R$date[j]) -> this_day
                        }else{
                                data_RMCRoutine %>% filter(Hour==i) %>%
                                        filter(Minutes < k*60 & Minutes >= k*30) %>%
                                        filter(Date==lambda_R$date[j]) -> this_day
                        }
                        n <- c(n,nrow(this_day))
                }
                lambda_R <- data.frame(lambda_R,n)
        }
        
} 
#colnames(lambda_R) <- c('date','h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23')

lambda_R %>% mutate(day=weekdays(date)) %>%
        mutate(wday = ifelse(day %in% c("Saturday", "Sunday"), "weekend", "weekday")) -> lambda_R

lambda_R %>% filter(wday=="weekend") -> weekend
lambda_R %>% filter(wday=="weekday") -> weekday
#hist(weekday$h5)

lower <-  c()
upper <-  c()
for( i in 2:49){
        l <- mean(weekday[,i])-1.96*sd(weekday[,i])/sqrt(nrow(lambda_R))
        lower <- c(lower,l)
        
        u <- mean(weekday[,i])+1.96*sd(weekday[,i])/sqrt(nrow(lambda_R))
        upper <- c(upper,u)
}
plot(colMeans(weekday[2:49]))
points(upper, pch="-")
points(lower, pch="-")

lower <-  c()
upper <-  c()
for( i in 2:49){
        l <- mean(weekend[,i])-1.96*sd(weekend[,i])/sqrt(nrow(lambda_R))
        lower <- c(lower,l)
        
        u <- mean(weekend[,i])+1.96*sd(weekend[,i])/sqrt(nrow(lambda_R))
        upper <- c(upper,u)
}
plot(colMeans(weekend[2:49]))
points(upper, pch="-")
points(lower, pch="-")

# According to the plots, Day = 7:30-18:30 regardles of weekday/weekend


#lambda
lambda_R_summary <- rbind(colMeans(weekday[2:49]),colMeans(weekend[2:49]))    
l_day <- rowMeans(lambda_R_summary[,16:38])
l_night <- rowMeans(lambda_R_summary[,-(16:38)])
lambda_R_summary <- cbind(l_day,l_night)
rownames(lambda_R_summary) <- c("weekday", "weekend")

#---Calculate mu--------------------------------------------------------------------------------------------
mean_tVerify <- as.data.frame(sort(unique(data_RMCRoutine$Date)))
colnames(mean_tVerify) <- c('date')
for(i in 0:23){
        for(k in 0:1){
                n <- c()
                for(j in 1:nrow(mean_tVerify)){
                        if(k==0){
                                data_RMCRoutine %>% filter(Hour==i) %>%
                                        filter(Minutes < (k+1)*30) %>%
                                        filter(Date==mean_tVerify$date[j]) -> this_day
                        }else{
                                data_RMCRoutine %>% filter(Hour==i) %>%
                                        filter(Minutes < k*60 & Minutes >= k*30) %>%
                                        filter(Date==mean_tVerify$date[j]) -> this_day
                        }
                        n <- c(n,mean(this_day$tVerify))
                }
                mean_tVerify <- data.frame(mean_tVerify,n)
        }
        
} 

mean_tVerify %>% mutate(day=weekdays(date)) %>%
        mutate(wday = ifelse(day %in% c("Saturday", "Sunday"), "weekend", "weekday")) -> mean_tVerify

mean_tVerify %>% filter(wday=="weekend") -> mean_tVerify_weekend
mean_tVerify %>% filter(wday=="weekday") -> mean_tVerify_weekday

tV_lower <-  c()
tV_upper <-  c()
for( i in 2:49){
        l <- mean(mean_tVerify_weekday[,i])-1.96*sd(mean_tVerify_weekday[,i])/sqrt(nrow(mean_tVerify))
        tV_lower <- c(tV_lower,l)
        
        u <- mean(mean_tVerify_weekday[,i])+1.96*sd(mean_tVerify_weekday[,i])/sqrt(nrow(mean_tVerify))
        tV_upper <- c(tV_upper,u)
}
plot(colMeans(mean_tVerify_weekday[2:49],na.rm = TRUE))
points(tV_upper, pch="-")
points(tV_lower, pch="-")

tV_lower <-  c()
tV_upper <-  c()
for( i in 2:49){
        l <- mean(mean_tVerify_weekend[,i])-1.96*sd(mean_tVerify_weekend[,i])/sqrt(nrow(mean_tVerify))
        tV_lower <- c(tV_lower,l)
        
        u <- mean(mean_tVerify_weekend[,i])+1.96*sd(mean_tVerify_weekend[,i])/sqrt(nrow(mean_tVerify))
        tV_upper <- c(tV_upper,u)
}
plot(colMeans(mean_tVerify_weekend[2:49],na.rm = TRUE))
points(tV_upper, pch="-")
points(tV_lower, pch="-")

# According to the plots, Day = 7:30-18:30 regardles of weekday/weekend


#mu
mu_R_summary <- rbind(colMeans(mean_tVerify_weekday[2:49],na.rm = TRUE),colMeans(mean_tVerify_weekend[2:49],na.rm = TRUE))    
m_day <- rowMeans(mu_R_summary[,16:38],na.rm = TRUE)
m_night <- rowMeans(mu_R_summary[,-(16:38)],na.rm = TRUE)
mu_R_summary <- cbind(m_day,m_night)
rownames(mu_R_summary) <- c("weekday", "weekend")
mu_R_summary <- 30/mu_R_summary

#---Create Model--------------------------------------------------------------------------------------------

#RMC/Routine/Weekday/day/10AM
ff<-QueueingModel(NewInput.MM1(lambda=lambda_R_summary['weekday','l_day'],mu=mu_R_summary['weekday','m_day'],n=0))  #n Number of customers in the system
summary(ff)
#lambda       mu c  k  m        RO        P0         Lq          Wq        X         L          W        Wqq      Lqq
#1 4.587786 33.05066 1 NA NA 0.1388107 0.8611893 0.02237419 0.004876903 4.587786 0.1611849 0.03513348 0.03513348 1.161185

# lambda= Inter-Arrival time
# mu= Inter-Service time
# c= Number of Servers
# k = System capacity (Infinite MM1)
# m = The size of the customer population (Infinite MM1)
# R0 = Traffic Intensity (overall system utilization, 1??????????????????)
# P0 = Probability that no customers are in the restaurant.
# Lq = Expected number of cars in the queue
# Wq = Expected waiting time in queue
# X = lambda
# L = Expected number of cars in the system
# W = Expected time in the system
# Wqq = Expected time in queue when queue exists
# Lqq= Expected number of customers in queue when queue exists