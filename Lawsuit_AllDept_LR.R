# Lawsuit - Linear Regression (Simplified + CSV Outputs)
# Scope: Base R only, within course slides (Unit 1–4).
# What this script does:
#   - Reads Lawsuit.csv
#   - Casts factors
#   - Prints basic summaries to console
#   - Saves summaries (dataset summary, gender counts, gender-by-dept) to CSV
#   - Builds three linear models (m0, m1, m3)
#   - Saves each model's coefficient table to CSV
#   - Saves model fit stats (R2, AdjR2) to CSV
#   - Draws simple boxplots and diagnostic plots (not saved as CSV)

setwd("/Users/zhangjianyu/Desktop/学习课件/NTU/AN6003 Analytics Strategy/AN6003 Course Materials/AN6003_GradedTeamAssignment")

# ---------------------------
# 0. Setup
# ---------------------------
if (!dir.exists("outputs")) dir.create("outputs")

# ---------------------------
# 1. Read data
# ---------------------------
df <- read.csv("Lawsuit.csv", stringsAsFactors = FALSE)

# ---------------------------
# 2. Clean and cast types
# ---------------------------
df$Dept <- factor(df$Dept,
                  levels = 1:6,
                  labels = c("Biochem/MolBio", "Physiology", "Genetics",
                             "Pediatrics", "Medicine", "Surgery"))
df$Gender <- factor(df$Gender, levels = c(0, 1), labels = c("Female", "Male"))
df$Clin   <- factor(df$Clin,   levels = c(0, 1), labels = c("Research", "Clinical"))
df$Cert   <- factor(df$Cert,   levels = c(0, 1), labels = c("NotCert", "Cert"))
df$Rank   <- factor(df$Rank,   levels = c(1, 2, 3), labels = c("Assistant", "Associate", "Full"))

# ---------------------------
# 3. Basic exploration (console + CSV)
# ---------------------------
cat("\n=== Basic Summary of Dataset ===\n")
print(summary(df))

# Numeric variable summary to CSV (base R)
is_num <- sapply(df, is.numeric)
num_names <- names(df)[is_num]
num_summary <- t(sapply(num_names, function(v) {
  x <- df[[v]]
  q <- quantile(x, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
  c(
    n = sum(!is.na(x)),
    mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE),
    q1 = q[[1]],
    median = q[[2]],
    q3 = q[[3]],
    max = max(x, na.rm = TRUE),
    na = sum(is.na(x))
  )
}))
num_summary_df <- data.frame(variable = rownames(num_summary), num_summary, row.names = NULL)
write.csv(num_summary_df, file = "outputs/data_summary_numeric.csv", row.names = FALSE)

# Gender counts CSV
gender_counts <- as.data.frame(table(df$Gender))
colnames(gender_counts) <- c("Gender", "Count")
write.csv(gender_counts, file = "outputs/gender_counts.csv", row.names = FALSE)

# Gender by Dept counts CSV (long format)
dept_gender_tbl <- as.data.frame(table(Dept = df$Dept, Gender = df$Gender))
write.csv(dept_gender_tbl, file = "outputs/gender_by_dept_counts.csv", row.names = FALSE)

# ---------------------------
# 4. Simple boxplots (Base R; optional to save as images)
# ---------------------------
boxplot(Sal94 ~ Gender, data = df,
        main = "Salary (1994) by Gender", xlab = "Gender", ylab = "Salary 1994",
        col = c("pink", "lightblue"))
boxplot(Sal95 ~ Gender, data = df,
        main = "Salary (1995) by Gender", xlab = "Gender", ylab = "Salary 1995",
        col = c("pink", "lightblue"))

# ---------------------------
# 5. Linear Regression Models (lm)
# ---------------------------
m0 <- lm(Sal95 ~ Gender, data = df)
m1 <- lm(Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin, data = df)
m3 <- lm(Sal94 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin, data = df)

# ---------------------------
# 6. Print results to console
# ---------------------------
cat("\n=== Model m0: Sal95 ~ Gender (Baseline) ===\n"); print(summary(m0))
cat("\n=== Model m1: Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin ===\n"); print(summary(m1))
cat("\n=== Model m3: Sal94 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin ===\n"); print(summary(m3))

# ---------------------------
# 7. Save model outputs to CSV (coefficients + fit stats)
# ---------------------------
save_model_coef <- function(model, filepath) {
  sm <- summary(model)$coefficients  # matrix: Estimate, Std. Error, t value, Pr(>|t|)
  dfc <- data.frame(term = rownames(sm), sm, row.names = NULL)
  write.csv(dfc, file = filepath, row.names = FALSE)
}

save_model_coef(m0, "outputs/model_m0_coefficients.csv")
save_model_coef(m1, "outputs/model_m1_coefficients.csv")
save_model_coef(m3, "outputs/model_m3_coefficients.csv")

fitstats <- data.frame(
  Model = c("m0: Sal95 ~ Gender", 
            "m1: Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin",
            "m3: Sal94 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin"),
  R2 = c(summary(m0)$r.squared, summary(m1)$r.squared, summary(m3)$r.squared),
  Adj_R2 = c(summary(m0)$adj.r.squared, summary(m1)$adj.r.squared, summary(m3)$adj.r.squared)
)
write.csv(fitstats, file = "outputs/model_fit_stats.csv", row.names = FALSE)

# ---------------------------
# 8. Diagnostic plots (Base R)
# ---------------------------
par(mfrow = c(2, 2)); plot(m1, col = "lightblue", pch = 19); par(mfrow = c(1, 1))
par(mfrow = c(2, 2)); plot(m2, col = "lightpink",  pch = 19); par(mfrow = c(1, 1))
par(mfrow = c(2, 2)); plot(m3, col = "red",  pch = 19); par(mfrow = c(1, 1))

cat("\nAnalysis complete. CSV files saved in 'outputs/' folder.\n")
