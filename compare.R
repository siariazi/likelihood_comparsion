# EpiEstim demostration

library(EpiEstim)
library(ggplot2)
library(dplyr)
library(zoo)
setwd("/Users/siavashriazi/Desktop/SFU/Codes/simulations")

recovery = 0.6 # this is sum of gamma and psi
mult = 7 # multiplication factor 
## load data
all_data = read.csv("mathematica.csv")
colnames(all_data) = c("sim","treeSize","betaTrue","mpe","lowerCI","upperCI")

# Define your start date
start_date <- as.Date("2023-01-01")

for (iter in all_data$sim){
  try({  file_name = paste("sim",iter,".csv",sep="")
  data = read.csv(file_name)
  colnames(data) = c("time","I")
  
  # Convert time values to date values
  data$dates <- start_date + data$time*10
  data = data[,c(3,2)]
  data <- aggregate(I ~ dates, data = data, sum)
  
  res_parametric_si <- estimate_R(data$I, 
                                  method="parametric_si",
                                  config = make_config(list(
                                    mean_si = recovery*mult, 
                                    std_si = 0.2))
  )
  time_to_pick = 1
  EpiMpeValues = res_parametric_si$R$`Mean(R)`
  EpiMpeValue = EpiMpeValues[time_to_pick]
  EpiMpeValue = EpiMpeValue*recovery
  
  EpiUpValues = res_parametric_si$R$`Quantile.0.05(R)`
  EpiUpValue = EpiUpValues[time_to_pick]
  EpiUpValue = EpiUpValue*recovery
  
  EpiDownValues = res_parametric_si$R$`Quantile.0.95(R)`
  EpiDownValue = EpiDownValues[time_to_pick]
  EpiDownValue = EpiDownValue*recovery
  
  rows_to_insert <- all_data$sim == iter
  # Insert the value into the selected rows
  all_data$EpiMpe[rows_to_insert] <- EpiMpeValue
  all_data$EpiUp[rows_to_insert] <- EpiUpValue
  all_data$EpiDown[rows_to_insert] <- EpiDownValue})
}
  
# Remove rows with NA values
all_data <- na.omit(all_data)

# Remove rows with infinite or negative infinite values
all_data <- all_data %>%
  filter_all(all_vars(!is.infinite(.)))


my_blue = '#5B9BD5' 
my_green = '#70AD47'
my_purple = '#7030A0'

# Create the plot with labels and legend
ggplot(all_data) +
  geom_point(aes(x = betaTrue, y = mpe, color = "Case Count"), fill = my_blue) +
  geom_errorbar(aes(x = betaTrue, ymin = lowerCI, ymax = upperCI, color = "Case Count"), alpha = 0.5, width = 0.2) +  
  geom_point(aes(x = betaTrue, y = EpiMpe, color = "EpiEstim"), fill = my_green) +
  geom_errorbar(aes(x = betaTrue, ymin = EpiDown, ymax = EpiUp, color = "EpiEstim"), alpha = 0.5, width = 0.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(
    x = bquote(beta ~ "True"),
    y = expression(hat(beta)),
    color = "Methods"
  ) + scale_color_manual(values = c(my_blue, my_green,"red")) + ylim(0,3) +
  theme_bw()

cor(all_data$betaTrue,all_data$mpe)
cor(all_data$betaTrue,all_data$EpiMpe)


############################
iter = 4
all_data$betaTrue[all_data$sim==iter]
file_name = paste("sim",iter,".csv",sep="")
data = read.csv(file_name)
colnames(data) = c("time","I")
plot(data$time,data$I,main="tochastic time",xlab="time (stochastic)",ylab="Incidences")

# Define your start date
start_date <- as.Date("2023-01-01")
# Convert time values to date values
data$dates <- start_date + data$time*10
data = data[,c(3,2)]
plot(data$dates,data$I,main="date",xlab="time (date)",ylab="Incidences")

# aggregating same date data 
data <- aggregate(I ~ dates, data = data, sum)

# Create a sequence of dates covering the entire range of your data
all_dates <- seq(min(data$date), max(data$date), by = "1 day")

# Create a new dataframe with all dates and set daily_incidence to 0
data2 <- data.frame(dates = all_dates, Inc = 0)

# Merge the two dataframes based on the 'date' column
data3 <- merge(data2, data, by = "dates", all = TRUE)

# Convert your dataframe to a zoo object
zoo_obj <- zoo(data3$I, order.by = data3$dates)

# Interpolate missing values
zoo_obj_interp <- na.approx(zoo_obj)

# Convert the interpolated zoo object back to a dataframe
data4 <- data.frame(date = index(zoo_obj_interp), I = coredata(zoo_obj_interp))

# Calculate daily incidence using diff, first I take derivative and then insert missing data
data4$incidence <- c(0, diff(data4$I))

data4$incidence[data4$incidence<0] = 0

# Replace missing values in the 'daily_incidence' column with 0
merged_data$incidence[is.na(merged_data$incidence)] <- 0

# Calculate daily incidence using diff, first I take derivative and then insert missing data
merged_data$incidence <- c(0, diff(data$I))






data3 <- merged_data[,c(1,4)]

mult = 2
res_parametric_si <- estimate_R(data4$incidence, 
                                method="parametric_si",
                                config = make_config(list(
                                  mean_si = recovery*mult, 
                                  std_si = 0.2)))


plot(res_parametric_si, legend = FALSE)

time_to_pick = 1
EpiMpeValues = res_parametric_si$R$`Mean(R)`
EpiMpeValue = EpiMpeValues[time_to_pick]
EpiMpeValue = EpiMpeValue*recovery
EpiMpeValue

