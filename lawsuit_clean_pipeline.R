# lawsuit_clean_pipeline.R
# Combined R pipeline for Lawsuit data analysis
# English comments, ggplot outputs, and clean CSV exports

# Load libraries
library(dplyr)
library(ggplot2)
library(scales)

# ========== STEP 1: Load and Clean Data ==========
setwd("/Users/zhangjianyu/Desktop/学习课件/NTU/AN6003 Analytics Strategy/AN6003 Course Materials/AN6003_GradedTeamAssignment")
if (!dir.exists("outputs")) dir.create("outputs")

df <- read.csv("Lawsuit.csv", stringsAsFactors = FALSE)

# Convert variables to factors/numeric as appropriate
df$Dept <- factor(df$Dept,
                  levels = 1:6,
                  labels = c("Biochem/MolBio", "Physiology", "Genetics",
                             "Pediatrics", "Medicine", "Surgery"))
df$Gender <- factor(df$Gender, levels = c(0, 1), labels = c("Female", "Male"))
df$Clin   <- factor(df$Clin,   levels = c(0, 1), labels = c("Research", "Clinical"))
df$Cert   <- factor(df$Cert,   levels = c(0, 1), labels = c("NotCert", "Cert"))
df$Rank   <- factor(df$Rank,   levels = c(1, 2, 3), labels = c("Assistant", "Associate", "Full"))

# Ensure numeric
df$Sal94  <- as.numeric(df$Sal94)
df$Prate  <- as.numeric(df$Prate)
df$Exper  <- as.numeric(df$Exper)

# Log transforms
df <- df %>%
  mutate(
    Log_Sal94  = log(Sal94),
    Log_Sal95  = log(Sal95),
    Log_Exper  = log1p(Exper),
    Log_Prate  = log1p(Prate)
  )

# ========== STEP 2: Train/Test Split ==========
set.seed(42)
n <- nrow(df)
test_idx <- sample.int(n, size = ceiling(0.2 * n))
train <- df[-test_idx, ]
test  <- df[test_idx, ]

# ========== STEP 3: Fit Models ==========
m0 <- lm(Sal95 ~ Gender, data = train)
m1 <- lm(Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin, data = train)
m2 <- lm(Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept + Rank, data = train)
m3 <- lm(Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept, data = train)

# ========== STEP 4: Metrics ==========
rmse_vec <- function(actual, pred) sqrt(mean((actual - pred)^2, na.rm = TRUE))
r2_vec   <- function(actual, pred) {
  sse <- sum((actual - pred)^2, na.rm = TRUE)
  sst <- sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
  1 - sse/sst
}

# Test predictions
yhat_m1 <- predict(m1, newdata = test)
met_m1 <- c(RMSE = rmse_vec(test$Sal95, yhat_m1), R2 = r2_vec(test$Sal95, yhat_m1))

yhat_m2_log <- predict(m2, newdata = test)
yhat_m3_log <- predict(m3, newdata = test)
smear2 <- mean(exp(residuals(m2)), na.rm = TRUE)
smear3 <- mean(exp(residuals(m3)), na.rm = TRUE)
yhat_m2_lvl <- smear2 * exp(yhat_m2_log)
yhat_m3_lvl <- smear3 * exp(yhat_m3_log)

met_m2_log <- c(RMSE = rmse_vec(test$Log_Sal94, yhat_m2_log), R2 = r2_vec(test$Log_Sal94, yhat_m2_log))
met_m3_log <- c(RMSE = rmse_vec(test$Log_Sal94, yhat_m3_log), R2 = r2_vec(test$Log_Sal94, yhat_m3_log))
met_m2_lvl <- c(RMSE = rmse_vec(test$Sal94, yhat_m2_lvl), R2 = r2_vec(test$Sal94, yhat_m2_lvl))
met_m3_lvl <- c(RMSE = rmse_vec(test$Sal94, yhat_m3_lvl), R2 = r2_vec(test$Sal94, yhat_m3_lvl))

test_metrics <- data.frame(
  model = c("m1 (level) - $", "m2 (log) - log", "m2 (log) - $", "m3 (log) - log", "m3 (log) - $"),
  RMSE  = c(met_m1["RMSE"], met_m2_log["RMSE"], met_m2_lvl["RMSE"], met_m3_log["RMSE"], met_m3_lvl["RMSE"]),
  R2    = c(met_m1["R2"],   met_m2_log["R2"],   met_m2_lvl["R2"],   met_m3_log["R2"],   met_m3_lvl["R2"])
)
write.csv(test_metrics, "outputs/test_metrics_oos.csv", row.names = FALSE)

# ========== STEP 5: Coefficients to $ Impacts (ggplot) ==========
base_salary <- median(train$Sal94, na.rm = TRUE)
coef_df <- summary(m2)$coefficients
coef_table <- data.frame(
  Term = rownames(coef_df),
  Estimate = coef_df[, "Estimate"],
  stringsAsFactors = FALSE
) %>%
  filter(Term != "(Intercept)") %>%
  mutate(
    Dollar_Impact = (exp(Estimate) - 1) * base_salary,
    Label = dollar_format()(Dollar_Impact)
  ) %>%
  arrange(desc(abs(Dollar_Impact)))

gg <- ggplot(coef_table, aes(x = reorder(Term, Dollar_Impact), y = Dollar_Impact)) +
  geom_col(fill = "lightblue") +
  coord_flip() +
  geom_text(aes(label = Label),
            hjust = ifelse(coef_table$Dollar_Impact >= 0, -0.1, 1.1),
            size = 3.6) +
  scale_y_continuous(labels = dollar_format(),
                     expand = expansion(mult = c(0.1, 0.25))) +
  labs(
    title = "Estimated Dollar Impact on Salary (Model m2)",
    subtitle = paste0("Baseline = ", dollar_format()(base_salary),
                      "; impacts = (exp(beta)-1) × baseline"),
    x = "Predictor",
    y = "Estimated $ Change"
  ) +
  theme_minimal(base_size = 12)

print(gg)
ggsave("outputs/m2_usd_barplot_gg.png", gg, width = 10, height = 7, dpi = 300)

cat("Analysis complete. Results saved in 'outputs/' and plot shown in RStudio Plots.\n")
