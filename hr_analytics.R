# HR analytics: predict next month's demand for hourly employees
demand <- c(187, 204, 221, 175, 175, 180, 185, 191, 205, 205, 197, 188,
            201, 230, 238, 181, 185, 190, 199, 207, 213, 215, 210, 205,
            220, 241, 245, 201, 196, 208, 211, 223, 230, 226, 221, 220)

# Monthly data from Oct 2023 to Sep 2026
ts_data <- ts(demand, start = c(2023, 10), frequency = 12)
plot(ts_data, main = "Monthly Demand for Hourly Employees",
     xlab = "Year", ylab = "Employees needed", col = "blue", lwd = 2)

# Regression with trend and monthly seasonality
period <- 1:36
month  <- factor(cycle(ts_data))
model  <- lm(demand ~ period + month)
print(summary(model))

# Forecast for next month (October 2026)
new_data <- data.frame(period = 37, month = factor(10, levels = 1:12))
forecast <- predict(model, newdata = new_data, interval = "prediction")
print(round(forecast, 1))

# Model accuracy
rmse <- sqrt(mean(residuals(model)^2))
cat("RMSE :", round(rmse, 2), "\n")

# Chart: actual vs fitted and forecast
plot(period, demand, type = "o", col = "blue", xlim = c(1, 38),
     xlab = "Month index", ylab = "Employees needed",
     main = "Actual, Fitted and Forecast")
lines(period, fitted(model), col = "red", lwd = 2)
points(37, forecast[1, "fit"], col = "darkgreen", pch = 19, cex = 1.5)
legend("topleft", c("Actual", "Fitted", "Forecast (Oct 2026)"),
       col = c("blue", "red", "darkgreen"), lty = c(1, 1, NA),
       pch = c(1, NA, 19))
