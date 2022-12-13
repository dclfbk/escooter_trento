# space-time stats packages:
library("ape") # for testing spatial or spatio-temporal independence with Moran's I statistic
library("FRK") # Fixed Rank Kriging -> auto_basis function
library("gstat") # -> idw function
library("sp") # for defining points/polygons

# packages for data manipulation and handling of 
# spatial/spatio-temporal objects:
library("spacetime")
library("tidyr")
library("dplyr") 

# for plotting:
library("ggplot2") 
library("RColorBrewer")

# set working directory
setwd('C:/Users/luisa/Desktop/thesis_project/code/base_models/data/')




# CONTENT:
# 1. GENERALIZED LINEAR MODEL
# 2. SPATIAL MODEL w/ FRK --> Spatial Random Effects (SRE) model
# 3. SPATIO-TEMPORAL MODEL --> SRE

# DOCS:
# https://www.rdocumentation.org/packages/FRK/versions/0.2.1/topics/FRK



# ------------------------------------------------------------------------------

# 1. GENERALIZED LINEAR MODEL

# read data
df <- read.csv(file = 'idw-data_daily_all.csv')
df = subset(df, select = -c(X, time, date, visibility, windspeedKmph) )
df = df[df$trips_count > 0,]
# df = df[df$year == 2022,]
df = subset(df, select = c(trips_count, lat, lon, month))



# basis function 
G <- auto_basis(data = df[,c("lon","lat")] %>%
                SpatialPoints(), # spatial object
                nres = 1, # one resolution
                type = "Gaussian") # Gaussian basis function

# evaluate basis function
S <- eval_basis(basis = G, # basis functions
                s = df[,c("lon","lat")] %>%
                as.matrix()) %>% # to matrix
  as.matrix() # convert result to matrix
colnames(S) <- paste0("basis", 1:ncol(S)) # assign column names

# attach additional columns with
# the basis-function covariate information to the df
# final dataframe:
df <- cbind(df, S) 

df[1:3, 1:16] 


# Possible exp. fam. and link functions:
# binomial(link = "logit")
# gaussian(link = "identity")
# Gamma(link = "inverse")
# inverse.gaussian(link = "1/mu^2")
# poisson(link = "log")
# quasi(link = "identity", variance = "constant")
# quasibinomial(link = "logit")
# quasipoisson(link = "log")


# GLM -> as lm but also requires exp. fam. and link function
glm1 <- glm(trips_count ~ (lon + lat + month) + ., # formula
            family = Gamma(link = "inverse"), # Poisson + log link (canonical)
            data = df)

# check for over-dispertion,
# aka if variance in the data is greater than 
# that suggested by model
glm1$deviance / glm1$df.residual # if > 1 -> over-dispersion
## 1.884891 # -> over-dispersion

summary(glm1)


# under the null-hp of no over-dispersion,
# the deviance is approximately chi-squared distributed with
# degrees of freedom equal to m-p-1
glm1$df.residual
## 15049
glm1$deviance
## 28365.72
1 - pchisq(q = glm1$deviance, df = glm1$df.residual) # p-value
## 0
# the probability of observing such a large or larger deviance 
# under the null hypothesis of no over-dispersion (p-value)
# is zero, thus reject the null-hp of no over-dispersion
# at significance level 10%, 5% and 1%

# negative-binomial (nb) family can deal with over-dispersion
# try -> generalized additive models, gam function in package mgcv


# PREDICTION
# generate grid over which to predict
# space-time grid --> lat x lon x month
pred_grid <- expand.grid(lon = seq(
  min(df$lon) - 0.2,
  max(df$lon) + 0.2,
  length.out = 80),
  lat = seq(
    min(df$lat) - 0.2,
    max(df$lat) + 0.2,
    length.out = 80),
  month = 1:12)

# evaluate basis function
# at the prediction locations
S_pred <- eval_basis(basis = G,
                     s = pred_grid[,c("lon","lat")] %>% 
                     as.matrix()) %>% 
  as.matrix() 
colnames(S_pred) <- paste0("basis", 1:ncol(S_pred)) 
pred_grid <- cbind(pred_grid,S_pred)

preds <- predict(glm1,
                 newdata = pred_grid,
                 type = "link",
                 se.fit = TRUE)


# attach preds and s.e. to grid for viz
pred_grid <- pred_grid %>%
  mutate(log_cnt = preds$fit,
         se = preds$se.fit)


# check deviance of residuals & correlations in errors
df$residuals <- residuals(glm1)

# plot residuals per month
residuals_glm1 <- ggplot(df) +
  geom_point(aes(lon, lat, colour = residuals)) +
  col_scale(name = "residuals") +
  facet_wrap(~month, ncol = 6) + theme_bw()
residuals_glm1
# there seems to be no strong spatial correlation
# as residuals close to each other are generally do NOT more similar
# than those further apart
# however, soma spatial influence is clearly visible


# MORAN'S I TEST
# on spatial residuals to check if corr.

# p-value < 0.05 --> statistically significant --> reject null hypothesis

# Moran's I null-hp --> randomly distributed in space (no spatial corr)

# p-value < 0.05 --> spatial correlation
# p-value > 0.05 --> no correlation 


P <- list() 
months <- c(1:12)

# for each month compute residuals
for(i in months) { 
  # select data from the i-th month
  monthly_df <- filter(df,
                       month == months[i]) 
  # take monthly data, select coords, compute distances, and convert to matrix
  obs_dists <- monthly_df %>% 
    select(lon,lat) %>% 
    dist() %>%
    as.matrix()
  # take the inverse of the matrix of distances
  obs_dists.inv <- 1/obs_dists # weight matrix
  # put zeros on the main diagonal
  diag(obs_dists.inv) <- 0 
  obs_dists.inv[which(!is.finite(obs_dists.inv))] <- 0 # replace inf w/ 0
  # run Moran's I on the monthly residuals
  P[[i]] <- Moran.I(monthly_df$residuals, # Moran's I
                    obs_dists.inv) %>% do.call("cbind", .)
}

do.call("rbind",P) %>% summary(digits = 2)
P
# at 5% significance level the null-hp of no spatial autocorr.
# in the residuals is cannot be rejected
# aka there seems to be SPATIAL CORRELATION

'''
# repeat for all df --> correlated
# take monthly data, select coords, compute distances, and convert to matrix
obs_dists <- df %>% 
  select(lon,lat) %>% 
  dist() %>%
  as.matrix()
# take the inverse of the matrix of distances
obs_dists.inv <- 1/obs_dists # weight matrix
# put zeros on the main diagonal
diag(obs_dists.inv) <- 0 
obs_dists.inv[which(!is.finite(obs_dists.inv))] <- 0 # replace inf w/ 0
# run Morans I on the monthly residuals
Moran.I(df$residuals, obs_dists.inv) %>% do.call("cbind", .)
'''

# temporal corr may also be present given the differences 
# in residuals over time (months)


# plot predictions
'''
ggplot(pred_grid) +
  geom_point(aes(lon, lat, colour = log_cnt)) +
  geom_tile(aes(x = lon, y = lat,
                fill = log_cnt)) +
  fill_scale() + 
  xlab("Latitude") + 
  ylab("Longitude") + 
  facet_wrap(~ month, nrow = 2) + 
  coord_fixed(xlim = c(min(pred_grid$lon), max(pred_grid$lon)),
              ylim = c(min(pred_grid$lat), max(pred_grid$lat))) +
  theme_bw() 
'''

# plot s.e.
# predictions are more precise toward the center rather than on the periphery 
ggplot(pred_grid) +
  geom_tile(aes(x = lon, y = lat,
                fill = se)) +
  fill_scale() + 
  xlab("Latitude") + 
  ylab("Longitude") + 
  facet_wrap(~ month, nrow = 2) + 
  coord_fixed(xlim = c(min(pred_grid$lon), max(pred_grid$lon)),
              ylim = c(min(pred_grid$lat), max(pred_grid$lat))) +
  theme_bw() 



# ------------------------------------------------------------------------------

# 2. SPATIAL FRK

# FRK needs all spatial objects to be of class 
# SpatialPointsDataFrame (if point-referenced) 
# or SpatialPolygonsDataFrame (if area-referenced)

# read data
df <- read.csv(file = 'idw_data-gh-prec6_meters.csv') # lat, lon in meters --> epsg 32632
df = subset(df, select = -c(X) )
df = df[df$year == 2022,]

# pt-referenced -> get spatial df by applying coordinates function
coordinates(df) = ~lon+lat 

# generate BAUs (Basic Areal Units)
# type can be either "grid" or "hex", indicating whether squared-grid or hexagonal 
set.seed(1)
GridBAUs1 <- auto_BAUs(manifold = plane(), # 2D plane
                       # SWITCH TO METERS COORDS??? -> cellsize = c(100,100)
                       cellsize = c(100,100), # BAU cellsize -> 2D, lat and lon (next 3D: lat, lon, time)
                       type = "grid", # grid
                       data = df, # data around which to create BAUs
                       convex=-0.05, # border buffer factor
                       nonconvex_hull=FALSE) # convex hull

# help(auto_BAUs)
# GridBAUs1 construct BAUs on the place, centered around the 

# important that this field is labelled 'fs' -> field fs is needed by SRE function
GridBAUs1$fs <- 1 # fine-scale variation at BAU level

plot(GridBAUs1) # plot BAU
plot(df) # plot data


# FRK decomposes the spatial process as a sum of basis functions
# these can be either user-specified or constructed via auto_basis
G <- auto_basis(manifold = plane(), # 2D plane
                data = df, # meuse data
                nres = 2, # number of resolutions
                type = "Gaussian", # type of basis function
                regular = 1) # place basis functions regularly in domain
# usually better results can be achieved by placing basis 
# functions IRREGULARLY on the domain
# when regular = 0 (irregular), the mesher package INLA is used 
# library(INLA)

# basis can be visualized with show_basis
show_basis(G) + # illustrate basis functions
  coord_fixed() + # fix aspect ratio
  xlab("Longitude (m) - epsg 32632") + # x-label
  ylab("Latitude (m) - epsg 32632") # y-label
# Basis functions automatically generated for the dataset with 2 resolutions. 
# The interpretation of the circles change with the domain and basis. 
# For Gaussian functions on the plane, each circle is
# centered at the basis function center, and has a radius equal to 1??. 
# Type help(auto basis) for details


# with BAU and BASIS FUNCTIONS
# we can construct the SRE model (Spatial Random Effects)
# https://www.rdocumentation.org/packages/FRK/versions/0.2.1/topics/FRK

# For fixed effects, we just use an intercept:   
# instead, we wish to use covariates, one must make sure that 
# they are also specified at the BAU level (and hence attributed to GridBAUs1)
f <- log(trips_count) ~ 1 # formula for Spatial Random Effects (SRE) model

# SRE model is now constructed from this function using sre function
# sre function bins the data in the BAUs, 
# construct all the matrices required for estimation, 
# and provides initial guesses for the quantities that need to be estimated
S <- SRE(f = f, # formula
         data = list(df), # list of datasets
         BAUs = GridBAUs1, # BAUs
         basis = G, # basis functions
         est_error = TRUE, # estimation measurement error  
         average_in_BAU = FALSE) # do not average data over BAUs -> if true summarize the data at a BAU level; good for large point-referenced datasets -> True = all data falling into the same bau are averaged

# fit model
# maximum likelihood carried out using the EM algorithm (expectation-maximization)
# which is assumed to have converged either when n_EM is exceeded, 
# or when the likelihood across subsequent steps does not change by more than tol
S <- SRE.fit(S, # SRE model
             n_EM = 10, # max. no. of EM iterations
             tol = 0.01, # tolerance at which EM is assumed to have converged
             print_lik=TRUE) # print log-likelihood at each iteration


# now we can predict all the BAUs with the fitted model
GridBAUs1 <- predict(S, obs_fs = FALSE) # obs_fs = F means we attribute the fine-scale variation to the process model (True means to the observation model (in which case it takes the role of systematicerror))
# GridBAUs1 now contains the predictions and the variance of the error
# in the fields mu and var

# plot mu and var using ggplot2
# to do this we need first to convert the Spatial object into a dataframe 
BAUs_df <- as(GridBAUs1,"data.frame")
# instead for SpatialPolygonsDataFrame to df takes as argument the BAUs and the variables we wish to extract from the BAUs. 

# plot FRK predictions:
g1 <- ggplot() + # Use a plain theme
  geom_tile(data=BAUs_df , # Draw BAUs
            aes(lon,lat,fill=mu), # Colour <-> Mean
            colour="light grey") + # Border is light grey
  scale_fill_distiller(palette="Spectral", # Spectral palette
                       name="log.trips") + # legend name
  geom_point(data=data.frame(df), # Plot data
             aes(lon,lat,fill=log(trips_count)), # Colour <-> log(trips_count)
             colour="black", # point outer colour
             pch=21, size=3) + # size of point
  coord_fixed() + # fix aspect ratio
  xlab("Longitude (m)") + ylab("Latitude (m)") + # axes labels
  labs(caption = "Tiles: predictions\nDots: real values") +
  theme_bw()

g1

# plot FRK standard error:
g2 <- ggplot() + # Similar to above but with standard error
  geom_tile(data=BAUs_df,
            aes(lon,lat,fill=sqrt(var)),
            colour="light grey") +
  scale_fill_distiller(palette="BrBG",
                       name = "s.e.",
                       guide = guide_legend(title="se")) +
  coord_fixed() +
  xlab("Longitude (m)") + ylab("Latitude (m)") +
  labs(caption = "Standard errors") + theme_bw()

g2


# REPEAT WITH LARGER REGIONS
# we wish to predict over regions encompassing several BAUs
# larger regionalisation:
Pred_regions <- auto_BAUs(manifold = plane(), # model on the 2D plane
                          cellsize = c(600,600), # choose a large grid size
                          type = "grid", # use a grid (not hex)
                          data = df, # the dataset on which to center cells
                          convex=-0.05, # border buffer factor
                          nonconvex_hull=FALSE) # convex hull
# predictions on larger polygons:
Pred_regions <- predict(S, newdata = Pred_regions) # prediction polygons

# plot
pred_df <- as(Pred_regions,"data.frame")
ggplot() + # Use a plain theme
  geom_tile(data=pred_df , # Draw BAUs
            aes(lon.x,lat.x,fill=mu), # Colour <-> Mean
            colour="light grey") + # Border is light grey
  scale_fill_distiller(palette="Spectral", # Spectral palette
                       name="log.trips") + # legend name
  geom_point(data=data.frame(df), # Plot data
             aes(lon,lat,fill=log(trips_count)), # Colour <-> log(trips_count)
             colour="black", # point outer colour
             pch=21, size=3) + # size of point
  coord_fixed() + # fix aspect ratio
  xlab("Longitude (m)") + ylab("Latitude (m)") + # axes labels
  labs(caption = "Tiles: predictions\nDots: real values") +
  theme_bw()

# S.E.
ggplot() + # Similar to above but with standard error
  geom_tile(data=pred_df,
            aes(lon.x,lat.x,fill=sqrt(var)),
            colour="light grey") +
  scale_fill_distiller(palette="BrBG",
                       name = "s.e.",
                       guide = guide_legend(title="se")) +
  coord_fixed() +
  xlab("Longitude (m)") + ylab("Latitude (m)") +
  labs(caption = "Standard errors") + theme_bw()

# -------------------------------------------------------------------------

# 3. SPATIO-TEMPORAL FRK
# w/ Spatial Random Effects (SRE)

# read data 
df <- read.csv(file = 'idw_data-gh-prec6.csv') # lat, lon in degrees 
df = subset(df, select = -c(X) )
df = df[df$year == 2022,]
df = df[df$month == 4,]

# build SPATIO-TEMPORAL object:

# first the temporal component must be defined as a DATE object
# "time" column as "Y-M-D" 
df <- within(df, {time = as.Date(paste(year,month,day,sep="-"))})

# point-referenced data:
# build spatio-temporal irregular data frame
# to do this we can use stConstruct from spacetime library
STObj <- stConstruct(x = df, # dataset
                     space = c("lon","lat"), # name of the columns with spatial coords
                     time="time", # name of column with time 
                     interval=TRUE) # TRUE if the data have  been recorded over the temporal interval, FALSE if specific instants

# Unlike for the spatial-only case, 
# the standard deviation of the measurement error 
# needs to be specified
# here we conservatively set it to be 2 trips
STObj$std <- 7

# construct BAUs in a space-time cube, centered around STObj, 
# with each BAU of size 
# ( 1 degree latitude x 1 deg longitude x 1 day )
# NOTE: tunit="days" 
# --> temporal intervals span equal to ONE DAY
grid_BAUs <- auto_BAUs(manifold=STplane(), # spatio-temporal process on the plane 
                       data=STObj, # data in ST obj
                       cellsize = c(0.001,0.001,1), # degree lat x degree lon x time -> BAU cell size
                       type="grid", # either grid or hex
                       convex=-0.1, # parameter for hull construction
                       tunit="days", # each BAU has temporal width of ONE DAY
                       nonconvex_hull=FALSE) # convex hull 
grid_BAUs$fs = 1 # fine-scale variation / homoscedastic fine-scale component

# first construct SPATIAL BASIS FUNCTIONS,
# then TEMPORAL BASIS FUNCTIONS
# and combine them by taking their TENSOR PRODUCT

# SPATIAL BASIS FUNCTION:
# project the spatio-temporal data onto the spatial domain (collapse out time)
# using as(STObj,"Spatial"), and then construct spatial basis function using auto basis
G_spatial <- auto_basis(manifold = plane(), # spatial functions on the plane
                        data=as(STObj,"Spatial"), # remove the temporal dimension
                        nres = 1, # three resolutions
                        type = "bisquare", # bisquare basis functions
                        regular = 1) # regular basis functions

# here we specify
# basis functions on the real line, 
# located  between t = 2 and t = 28 at an interval spacing of 4
# each location represents a temporal interval used in the construction of grid BAUs;
# for example, t=1 corresponds to 2020-11-30 (first date of data collection)
# cfr. print below to understand better
print(head(grid_BAUs@time)) # show time indices

G_temporal <- local_basis(manifold = real_line(), # functions on the real line
                          type = "Gaussian", # Gaussian functions
                          loc = matrix(seq(2,28,by=4)), # locations of functions
                          scale = rep(3,7)) # scales of functions

# VISUALIZE BASIS FUNCTIONS
basis_s_plot <- show_basis(G_spatial) + xlab("lon") + ylab("lat")
basis_t_plot <- show_basis(G_temporal) + xlab("time index") + ylab(expression(phi(t)))

basis_s_plot
basis_t_plot


# COMBINE SPATIAL AND TEMPORAL BAU WITH TENSOR PRODUCT:
# TensorP to combine spatial and temporal functions
G <- TensorP(G_spatial, G_temporal) # take the tensor product

# CONSTRUCT THE SRE MODEL
# using intercept and latitude as fixed effects
# STObj as the data
# G (tensor product of spatial and temporal basis functions) as set of basis functions
# grid_BAUs as the 

# Note: specify est_error = FALSE 
# cause this functionality is NOT IMPLEMENTED ATM for spatio-temporal data

# formula
# -> can only contain space or time as covariates when BAUs are not explicitly supplied with the covariate data
f <- trips_count ~ 1 + lat  # fixed effects part 
S <- SRE(f = f, # formula fixed effects
         data = list(STObj), # data (must be a list of data)
         basis = G, # basis functions
         BAUs = grid_BAUs, # BAUs
         est_error = FALSE) # do not estimate measurement-error variance

# FIT MODEL
S <- SRE.fit(S, # estimate parameters in the SRE model S
             n_EM = 10, # maximum no. of EM iterations
             tol = 0.1, # tolerance on log-likelihood
             print_lik=TRUE) # print log-likelihood trace

# PREDICT
grid_BAUs <- predict(S, obs_fs = FALSE)


# iterate to the time-points we wish to visualize
# extract a data frame for the BAUs on selected days (to plot) 
analyse_days <- c(4,6,11,13,20,29) # analyse only a few days
df_st <- lapply(analyse_days, # for each day
                function(i)
                  as(grid_BAUs[,i],"data.frame") %>%
                  cbind(day = i)) # add day number to df
df_st <- do.call("rbind",df_st) # append all dfs together

# df w/ real values for selected days only (for plot)
df_selected_days <- df[is.element(df$day, analyse_days),]


# plot ST predictions
df_st_plt <- df_st
df_st_plt$mu[which(df_st_plt$mu<0)]<-0
ggplot() + # Use a plain theme
  ggtitle('April 2022') +
  geom_tile(data=df_st_plt , # Draw BAUs
            aes(lon,lat,fill=mu), # Colour <-> Mean
            colour="light grey") + # Border is light grey
  scale_fill_distiller(palette="Spectral", # Spectral palette
                       name="num.trips") + 
  geom_point(data=data.frame(df_selected_days), # Plot data
             aes(lon,lat,fill=trips_count), # Colour <-> log(trips_count)
             colour="black", # point outer colour
             pch=21, size=3) + # size of point
  coord_fixed() + 
  facet_wrap(~ day, ncol = 9) +
  xlab("Longitude") + ylab("Latitude") + # axes labels
  labs(caption = "Tiles: predictions\nDots: real values") +
  theme_bw()

# s.e.
ggplot() + # Use a plain theme
  ggtitle('April 2022') +
  geom_tile(data=df_st , # Draw BAUs
            aes(lon,lat,fill=sqrt(var)), # Colour <-> Mean
            colour="light grey") + # Border is light grey
  scale_fill_distiller(palette="BrBG", 
                       name="s.e.",
                       guide = guide_legend(title="se")) + 
  coord_fixed() + # fix aspect ratio
  facet_wrap(~ day, ncol = 9) +
  xlab("Longitude") + ylab("Latitude") + # axes labels
  labs(caption = "Standard errors") + theme_bw()


# MAE (daily)
mean(sqrt(df_st$var))
## 2.836415

