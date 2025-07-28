# Install necessary libraries (only first time)
# install.packages("arrow")
# install.packages("tidyverse")
# install.packages("lubridate")
# install.packages("readr")
# install.packages("randomForest")

library(arrow)
library(dplyr)
library(lubridate)
library(readr)
library(randomForest)

# Load house data
house_data <- read_parquet("https://intro-datascience.s3.us-east-2.amazonaws.com/SC-data/static_house_info.parquet")

# Randomly select 500 houses
set.seed(42)  # for reproducibility
house_ids <- sample(house_data$bldg_id, 500)

# Check
length(house_ids)  # Should be 500

# Create empty list
all_data_list <- list()

for (id in house_ids) {
  sample_id <- as.character(id)
  sample_house <- house_data %>% filter(bldg_id == id)
  county_code <- sample_house$in.county[1]
  
  try({
    # Load energy
    energy_url <- paste0("https://intro-datascience.s3.us-east-2.amazonaws.com/SC-data/2023-houseData/", sample_id, ".parquet")
    energy_data <- read_parquet(energy_url) %>%
      filter(!is.na(time)) %>%
      rename(timestamp = time) %>%
      filter(month(timestamp) == 7)  # Only July
    
    # Load weather
    weather_url <- paste0("https://intro-datascience.s3.us-east-2.amazonaws.com/SC-data/weather/2023-weather-data/", county_code, ".csv")
    weather_data <- read_csv(weather_url) %>%
      filter(!is.na(date_time)) %>%
      rename(timestamp = date_time) %>%
      filter(month(timestamp) == 7) %>%
      filter(!is.na(`Dry Bulb Temperature [°C]`))
    
    # Add county info
    energy_data$in.county <- county_code
    weather_data$in.county <- county_code
    
    # Merge energy + weather
    combined <- energy_data %>%
      left_join(weather_data, by = c("timestamp", "in.county"))
    
    # Add static house info
    combined <- combined %>%
      mutate(bldg_id = id) %>%
      left_join(sample_house, by = "bldg_id")
    
    # Add total_kWh
    energy_cols <- combined %>% select(contains("energy_consumption"))
    combined$total_kWh <- rowSums(energy_cols, na.rm = TRUE)
    
    all_data_list[[sample_id]] <- combined
  }, silent = TRUE)
}

# Combine all house data
all_houses_data <- bind_rows(all_data_list)

# Clean
july_data <- all_houses_data %>%
  filter(!is.na(total_kWh), !is.na(`Dry Bulb Temperature [°C]`))
#colnames(july_data)[colnames(july_data) == "Dry Bulb Temperature [°C]"] <- "temp_c"
#colnames(july_data)[colnames(july_data) == "Relative Humidity [%]"] <- "humidity"
head(july_data)
saveRDS(july_data, "july_data.rds")
model_data <- july_data %>%
  select(
    total_kWh,
    `Dry Bulb Temperature [°C]`,
    `Relative Humidity [%]`,
    in.sqft,
    in.bedrooms,
    in.occupants,
    in.cooling_setpoint,
    in.heating_setpoint,
    in.ceiling_fan,
    in.cooling_setpoint_has_offset,
    in.heating_setpoint_has_offset,
    in.insulation_ceiling,
    in.insulation_wall,
    in.windows,
    in.water_heater_fuel,
    in.dishwasher,
    in.clothes_dryer,
    in.clothes_washer,
    in.lighting,
    in.hvac_cooling_efficiency,
    in.hvac_heating_efficiency,
    in.refrigerator,
    in.cooking_range,
    in.plug_loads,
    in.hot_water_distribution,
    in.interior_shading,
    in.door_area,
    in.geometry_attic_type,
    in.geometry_foundation_type,
    in.geometry_wall_type,
    in.solar_hot_water
  ) %>%
  na.omit()
model_data

colnames(model_data)[colnames(model_data) == "Dry Bulb Temperature [°C]"] <- "temp_c"
colnames(model_data)[colnames(model_data) == "Relative Humidity [%]"] <- "humidity"
model_data <- model_data %>% select(where(~n_distinct(.) > 1))
#colnames(model_data)
#nrow(model_data)
#write.csv(model_data, "model_data1.csv", row.names = FALSE)
#install.packages("writexl")  # (only first time)
#library(writexl)
#write_xlsx(model_data, "model_data1.xlsx")
#print("done")
saveRDS(model_data, "model_data.rds")
set.seed(42)
n <- nrow(model_data)
train_indices <- sample(1:n, 0.7 * n)  # 70% training

train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]
dim(train_data)
dim(test_data)



# Build Linear Regression
lm_model <- lm(total_kWh ~ ., data = train_data)

# Summary (optional to see)
summary(lm_model)

saveRDS(lm_model, "lm_model.rds")

# Predict on test data
lm_predictions <- predict(lm_model, newdata = test_data)

# Evaluate
lm_mse <- mean((lm_predictions - test_data$total_kWh)^2)
print(paste("Linear Regression MSE:", round(lm_mse, 4)))



# Load library
library(randomForest)

# Build Random Forest
rf_model <- randomForest(
  total_kWh ~ ., 
  data = train_data, 
  ntree = 100,
  importance = TRUE
)

saveRDS(rf_model, "rf_model.rds")

# Predict on test data
rf_predictions <- predict(rf_model, newdata = test_data)

# Evaluate
rf_mse <- mean((rf_predictions - test_data$total_kWh)^2)
print(paste("Random Forest MSE:", round(rf_mse, 4)))
importance(rf_model)
# Variable Importance Plot
varImpPlot(rf_model)
print("done")

library(ggplot2)

# Linear Regression Plot
ggplot(data = test_data, aes(x = total_kWh, y = lm_predictions)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Linear Regression: Actual vs Predicted",
       x = "Actual total_kWh", y = "Predicted total_kWh") +
  theme_minimal()

# Random Forest Plot
ggplot(data = test_data, aes(x = total_kWh, y = rf_predictions)) +
  geom_point(alpha = 0.5, color = "forestgreen") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Random Forest: Actual vs Predicted",
       x = "Actual total_kWh", y = "Predicted total_kWh") +
  theme_minimal()

library(lubridate)
library(dplyr)

july_warmer <- july_data %>%
  mutate(temp_c = `Dry Bulb Temperature [°C]` + 5,
         humidity = `Relative Humidity [%]`)

# Step 3: Predict using Random Forest model
july_warmer_predictions <- predict(rf_model, newdata = july_warmer)

# Step 4: Calculate total energy usage
original_total_energy <- sum(july_data$total_kWh)
warmer_total_energy <- sum(july_warmer_predictions)

# Step 5: Print Results
print(paste("Original July Energy Consumption:", round(original_total_energy, 2), "kWh"))
print(paste("Predicted July Energy Consumption (+5°C):", round(warmer_total_energy, 2), "kWh"))
print(paste("Increase in Energy:", round(warmer_total_energy - original_total_energy, 2), "kWh"))


#  Bar Chart - Before vs After +5°C
#  Bar Chart - Per House Energy (Before vs After +5°C)

library(ggplot2)

# Calculate average kWh per house
num_houses <- length(unique(july_data$bldg_id))

energy_df <- data.frame(
  Scenario = c("Original July", "July +5°C"),
  Avg_kWh_Per_House = c(original_total_energy / num_houses, warmer_total_energy / num_houses)
)

ggplot(energy_df, aes(x = Scenario, y = Avg_kWh_Per_House, fill = Scenario)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = round(Avg_kWh_Per_House, 2)), vjust = -0.5, size = 5) +
  labs(
    title = "Avg Predicted Energy Usage per House (Original vs +5°C)",
    y = "kWh per House",
    x = ""
  ) +
  theme_minimal() +
  theme(legend.position = "none")










# Energy Usage Analysis - Final Visuals

# Load libraries
library(dplyr)
library(ggplot2)
library(tidyr)

# Assuming you have the following ready:
# - july_data (original July data)
# - july_warmer (July data with +5°C adjustment)

# 1. Total Energy by Hour (Before vs After +5°C)
original_hourly <- july_data %>%
  mutate(hour = lubridate::hour(timestamp)) %>%
  group_by(hour) %>%
  summarise(total_energy = sum(total_kWh))

future_hourly <- july_warmer %>%
  mutate(hour = lubridate::hour(timestamp)) %>%
  group_by(hour) %>%
  summarise(total_energy = sum(total_kWh))

# Combine for plotting
hourly_combined <- original_hourly %>%
  rename(current_usage = total_energy) %>%
  left_join(future_hourly %>% rename(future_usage = total_energy), by = "hour") %>%
  pivot_longer(cols = c(current_usage, future_usage), names_to = "Scenario", values_to = "Energy")

# Plot
ggplot(hourly_combined, aes(x = hour, y = Energy, fill = Scenario)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Total Energy Usage by Hour (Before vs After +5°C)",
       x = "Hour", y = "Total Energy (kWh)") +
  theme_minimal()

# 2. Total Energy Consumption by Device (Bar Chart)

device_cols <- c(
  "out.electricity.clothes_dryer.energy_consumption",
  "out.electricity.refrigerator.energy_consumption",
  "out.electricity.cooling.energy_consumption",
  "out.electricity.lighting_interior.energy_consumption",
  "out.electricity.plug_loads.energy_consumption"
)

device_total <- july_data %>%
  select(all_of(device_cols)) %>%
  summarise(across(everything(), sum)) %>%
  pivot_longer(cols = everything(), names_to = "Device", values_to = "Energy") %>%
  arrange(desc(Energy))

# Plot
ggplot(device_total, aes(x = reorder(Device, -Energy), y = Energy, fill = Device)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Energy Consumption by Top 5 Devices", x = "Device", y = "Energy (kWh)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# 3. Device Contribution Pie Chart

ggplot(device_total, aes(x = "", y = Energy, fill = Device)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Device Contribution to Total Energy") +
  theme_void()


# 4. Hourly Energy Usage Trends for Top 5 Devices

device_hourly <- july_data %>%
  mutate(hour = lubridate::hour(timestamp)) %>%
  group_by(hour) %>%
  summarise(across(all_of(device_cols), sum)) %>%
  pivot_longer(cols = -hour, names_to = "Device", values_to = "Energy")

# Plot
ggplot(device_hourly, aes(x = hour, y = Energy, color = Device)) +
  geom_line(size = 1) +
  labs(title = "Hourly Energy Usage Trends for Top 5 Devices", x = "Hour", y = "Energy (kWh)") +
  theme_minimal()


print("All plots done")



