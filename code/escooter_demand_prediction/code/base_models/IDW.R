library("dplyr")
library("fields")
library("ggplot2")
library("gstat") # -> idw function
library("RColorBrewer")
library("sp")
library("spacetime")
library("STRbook")

# INVERSE DISTANCE WEIGHTING (IDW)

# set working directory
setwd('C:/Users/luisa/Desktop/thesis_project/code/base_models/')

# read data
df <- read.csv(file = 'idw-data_daily_all.csv')
df = subset(df, select = -c(X) )

# filter most recent data
# subset data -> 1 APRIL 2022 
df <- filter(df, 
            month == 4 & 
            year == 2022) 

# construct 3D spatio-temporal grid using expand.grid
# 4 x 8 x 6 grid of lat x long x day
# --> all territory, for 6 days
pred_grid <- expand.grid(lon = seq(min(df$lon), max(df$lon), length = 4),
                         lat = seq(min(df$lat), max(df$lat), length = 8),
                         day = seq(4, 29, length = 6)) # pick 6 days


# run IDW for trips_count
# omitting data for the last day to interpolate
df_no_29 <- filter(df, !(day == 29)) # remove 29 april
df_29april_idw <- idw(formula = trips_count ~ 1, # dep. variable to interpolate
                   locations = ~ lon + lat + day, # spatial and temporal variables
                   data = df_no_29, 
                   newdata = pred_grid, # prediction grid
                   idp = 6) 

# NOTE: idp aka inv. dist. power (the alpha) to optimize via CV;
# the larger the alpha, the less the smoothing

df_29april_idw # lon, lat, day, and var1.pred (the interpolation)


# plot data 29 april - 5 may 2022
ggplot(df_29april_idw) +
  geom_tile(aes(x = lon, y = lat,
                fill = var1.pred)) +
  fill_scale(name = "num. trips") + # attach color scale
  xlab("Longitude") + # x-axis label
  ylab("Latitude") + # y-axis label
  facet_wrap(~ day, ncol = 3) + # facet by day
  coord_fixed(xlim = c(min(df$lon), max(df$lon)), 
              ylim = c(min(df$lat), max(df$lat))) + 
  theme_bw() # B&W theme


# by weekday
df <- read.csv(file = 'idw-data_daily_all.csv')
pred_grid <- expand.grid(lon = seq(min(df$lon), max(df$lon), length = 50),
                         lat = seq(min(df$lat), max(df$lat), length = 50),
                         weekday = seq(0, 6, length = 7)) # pick 6 days
df_idw <- idw(formula = trips_count ~ 1, # dep. variable to interpolate
                   locations = ~ lon + lat + weekday, # spatial and temporal variables
                   data = df, # full period
                   newdata = pred_grid, # prediction grid
                   idp = 6) 
# plot 
ggplot(df_idw) +
  geom_tile(aes(x = lon, y = lat,
                fill = var1.pred)) +
  fill_scale(name = "num. trips") + # attach color scale
  xlab("Longitude") + # x-axis label
  ylab("Latitude") + # y-axis label
  facet_wrap(~ weekday, ncol = 4) + # facet by day
  coord_fixed(xlim = c(min(df$lon), max(df$lon)), 
              ylim = c(min(df$lat), max(df$lat))) + 
  theme_bw() # B&W theme
