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

# Macbook Path
setwd("/Users/zhangjianyu/Desktop/学习课件/NTU/AN6003 Analytics Strategy/AN6003 Course Materials/AN6003_GradedTeamAssignment")
# Windows Path
#setwd("C:/Users/Zhang/OneDrive - Nanyang Technological University/桌面/NTU学习/AN6003 Course Materials/AN6003 Course Materials/Graded Team Assignment - Gender Discrimination Lawsuit/AN6003_GradedTeamAssignment")

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

# Log transforms (safe for zeros via log1p)
df$Log_Sal94  <- log(df$Sal94)           # Sal94 > 0, direct log is fine
df$Log_Sal95  <- log(df$Sal95)
df$Log_Exper  <- log1p(df$Exper)         # log(Exper + 1), avoids log(0)
df$Log_Prate  <- log1p(df$Prate)         # log(Prate + 1)

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
# m0: baseline (levels, 1995)
m0 <- lm(Sal95 ~ Gender, data = df)

# m1: full levels model (1995) with all main controls
m1 <- lm(Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin, data = df)

# m2: log-levels model (1994) with Rank (replicates slide with log terms)
m2 <- lm(Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept + Rank, data = df)

# m3: log-levels model (1994) dropping Rank to mitigate multicollinearity
#     (slide rationale: Rank correlates with Gender; remove as a robustness spec)
m3 <- lm(Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept, data = df)

# ---------------------------
# 6. Print results to console
# ---------------------------
cat("\n=== Model m0: Sal95 ~ Gender (baseline) ===\n");              print(summary(m0))
cat("\n=== Model m1: Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin ===\n"); print(summary(m1))
cat("\n=== Model m2: Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept + Rank ===\n"); print(summary(m2))
cat("\n=== Model m3: Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept (no Rank) ===\n"); print(summary(m3))

# ---------------------------
# 7. Save model outputs to CSV (coefficients + fit stats)
# ---------------------------
# 1) helper: significance stars
pstars <- function(p) ifelse(p < .001, "***",
                             ifelse(p < .01,  "**",
                                    ifelse(p < .05,  "*", "")))

# 2) helper: make a tidy coef table (term, est, se, t, p, star, cell)
tidy_from_lm <- function(model) {
  sm <- summary(model)$coefficients
  dfc <- data.frame(term = rownames(sm),
                    est  = sm[, 1],
                    se   = sm[, 2],
                    t    = sm[, 3],
                    p    = sm[, 4],
                    row.names = NULL,
                    check.names = FALSE)
  dfc$star <- pstars(dfc$p)
  dfc$cell <- sprintf("%.2f%s", dfc$est, dfc$star)  # e.g., 0.18***
  dfc
}

# 3) helper: pretty names (match the slide headings)
pretty_term <- function(x) {
  x <- sub("^\\(Intercept\\)$",   "Constant",          x)
  x <- sub("^GenderMale$",        "Gender",            x)
  x <- sub("^ClinClinical$",      "Clin",              x)
  x <- sub("^CertCert$",          "Cert",              x)
  x <- sub("^Log_Prate$",         "log_Prate",         x)
  x <- sub("^Log_Exper$",         "log_Exper",         x)
  x <- sub("^DeptPhysiology$",    "Physiology",        x)
  x <- sub("^DeptGenetics$",      "Genetics",          x)
  x <- sub("^DeptPediatrics$",    "Pediatrics",        x)
  x <- sub("^DeptMedicine$",      "Medicine",          x)
  x <- sub("^DeptSurgery$",       "Surgery",           x)
  x <- sub("^RankAssociate$",     "Associate",         x)
  x <- sub("^RankFull$",          "Full Professor",    x)
  x
}

# 4) build tidy tables for each model
t0 <- transform(tidy_from_lm(m0), model = "m0: Sal95 ~ Gender")
t1 <- transform(tidy_from_lm(m1), model = "m1: Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin")
t2 <- transform(tidy_from_lm(m2), model = "m2: Log_Sal94 ~ Gender + Clin + Cert + log_Prate + log_Exper + Dept + Rank")
t3 <- transform(tidy_from_lm(m3), model = "m3: Log_Sal94 ~ Gender + Clin + Cert + log_Prate + log_Exper + Dept (no Rank)")

# 5) pretty labels + keep only the 'cell' to pivot wide
t_all <- rbind(t0, t1, t2, t3)
t_all$term_pretty <- pretty_term(t_all$term)

# 6) choose column order (like the slide)
col_order <- c("Constant","Gender","Clin","Cert","log_Prate","log_Exper",
               "Physiology","Genetics","Pediatrics","Medicine","Surgery",
               "Associate","Full Professor")

# 7) pivot to WIDE (one row per model; columns = variables)
models_vec <- unique(t_all$model)
wide <- data.frame(Model = models_vec, check.names = FALSE)
for (cn in col_order) wide[[cn]] <- ""

fill_one <- function(df_model, target_row) {
  for (i in seq_len(nrow(df_model))) {
    nm  <- df_model$term_pretty[i]
    val <- df_model$cell[i]
    if (nm %in% col_order) wide[target_row, nm] <<- val
  }
}

for (i in seq_along(models_vec)) {
  dfm <- subset(t_all, model == models_vec[i])
  fill_one(dfm, i)
}

# 8) add fit stats as extra columns
wide$R2      <- c(summary(m0)$r.squared, summary(m1)$r.squared,
                  summary(m2)$r.squared, summary(m3)$r.squared)
wide$Adj_R2  <- c(summary(m0)$adj.r.squared, summary(m1)$adj.r.squared,
                  summary(m2)$adj.r.squared, summary(m3)$adj.r.squared)
wide$R2      <- round(wide$R2, 3)
wide$Adj_R2  <- round(wide$Adj_R2, 3)

# 9) save
outfile_wide <- "outputs/model_m0_m3_coefficients_wide.csv"
write.csv(wide, file = outfile_wide, row.names = FALSE)
cat("[Saved]", normalizePath(outfile_wide), "\n")

# =========================================================
# 7B. Slide visuals (Base R) — add right after model prints
# =========================================================

# ---- small palette ----
female_col <- "pink"; male_col <- "lightblue"
ink <- "#333333"; grid_col <- "#D9D9D9"; accent <- "#6FA8DC"

# ---------- (1) Gender effect: m0 (USD) vs m3 (%) ----------
png("outputs/plot_gender_effect_m0_vs_m3.png", width=1800, height=900, res=180)
layout(matrix(1:2, nrow=1))

## m0: raw $ diff (Male - Female) with 95% CI
par(mar=c(5,6,3,2), xaxs="i")
sm0 <- summary(m0); ci0 <- confint(m0)
b0  <- sm0$coefficients["GenderMale","Estimate"]
lo0 <- ci0["GenderMale",1]; hi0 <- ci0["GenderMale",2]
xlim0 <- range(c(lo0, hi0))*1.1
plot(NA, xlim=xlim0, ylim=c(0.5,1.5), yaxt="n",
     xlab="Difference in 1995 Salary (USD)", ylab="", main="m0: Raw gender gap ($)",
     col.axis=ink, col.lab=ink, col.main=ink)
abline(v=0, col=grid_col, lwd=2)
segments(lo0,1, hi0,1, col=accent, lwd=6, lend=1)
points(b0,1, pch=19, cex=2, col=accent)
axis(2, at=1, labels="Male - Female", las=1, col.axis=ink)

## m3: within-structure premium (%) with 95% CI
par(mar=c(5,6,3,2), xaxs="i")
sm3 <- summary(m3); ci3 <- confint(m3)
b3  <- sm3$coefficients["GenderMale","Estimate"]
lo3 <- ci3["GenderMale",1]; hi3 <- ci3["GenderMale",2]
pct <- (exp(b3)-1)*100; loP <- (exp(lo3)-1)*100; hiP <- (exp(hi3)-1)*100
xlim3 <- range(c(loP, hiP))*1.1
plot(NA, xlim=xlim3, ylim=c(0.5,1.5), yaxt="n",
     xlab="Estimated % difference in Salary (1994)", ylab="", main="m3: Within-structure premium (%)",
     col.axis=ink, col.lab=ink, col.main=ink)
abline(v=0, col=grid_col, lwd=2)
segments(loP,1, hiP,1, col=accent, lwd=6, lend=1)
points(pct,1, pch=19, cex=2, col=accent)
axis(2, at=1, labels="Male vs Female", las=1, col.axis=ink)
dev.off()

# ---------- helper: simplified forest ----------
simple_forest <- function(model, title, terms_keep, to_percent=FALSE) {
  sm <- summary(model); cf <- sm$coefficients; ci <- confint(model)
  keep <- terms_keep[terms_keep %in% rownames(cf)]
  est <- cf[keep,"Estimate"]; lo <- ci[keep,1]; hi <- ci[keep,2]
  xlab <- "Coefficient"
  if (to_percent) {
    ind <- grepl("^Gender|^Clin|^Cert|^Dept", keep)  # indicator terms
    est[ind] <- (exp(est[ind])-1)*100
    lo[ind]  <- (exp(lo[ind]) -1)*100
    hi[ind]  <- (exp(hi[ind]) -1)*100
    xlab <- "Effect size (%, for indicator terms); raw units otherwise"
  }
  par(mar=c(5,16,3,2), xaxs="i")
  plot(est, seq_along(est), xlim=range(c(lo,hi))*1.1, pch=19, yaxt="n",
       ylab="", xlab=xlab, main=title, col=ink)
  segments(lo, seq_along(est), hi, seq_along(est), col=accent, lwd=5, lend=1)
  points(est, seq_along(est), pch=19, cex=1.4, col=accent)
  # nicer labels
  lab <- gsub("^GenderMale$","Gender",
              gsub("^ClinClinical$","Clin",
                   gsub("^CertCert$","Cert",
                        gsub("^RankAssociate$","Associate",
                             gsub("^RankFull$","Full Professor",
                                  gsub("^DeptPhysiology$","Physiology",
                                       gsub("^DeptGenetics$","Genetics",
                                            gsub("^DeptPediatrics$","Pediatrics",
                                                 gsub("^DeptMedicine$","Medicine",
                                                      gsub("^DeptSurgery$","Surgery", keep))))))))))
  axis(2, at=seq_along(est), labels=lab, las=1)
  abline(v=0, col=grid_col, lwd=2)
}

# ---------- (2) simplified forests ----------
png("outputs/plot_forest_simple_m1.png", width=1600, height=1200, res=180)
simple_terms_m1 <- c("GenderMale","RankAssociate","RankFull",
                     "DeptPhysiology","DeptGenetics","DeptPediatrics","DeptMedicine","DeptSurgery",
                     "Exper")
simple_forest(m1, "Simplified Forest — m1 (levels, key drivers)", simple_terms_m1, to_percent=FALSE)
dev.off()

png("outputs/plot_forest_simple_m3.png", width=1600, height=1200, res=180)
simple_terms_m3 <- c("GenderMale","ClinClinical","CertCert",
                     "DeptPhysiology","DeptGenetics","DeptPediatrics","DeptMedicine","DeptSurgery",
                     "Log_Exper")
simple_forest(m3, "Simplified Forest — m3 (log; indicators in %)", simple_terms_m3, to_percent=TRUE)
dev.off()

# ---------- (3) Predicted Salary bar chart (same covariates; change only Gender) ----------
pred_base <- df
new_f <- pred_base; new_f$Gender <- factor("Female", levels=levels(df$Gender))
new_m <- pred_base; new_m$Gender <- factor("Male",   levels=levels(df$Gender))
yhat_f <- mean(predict(m1, newdata=new_f))
yhat_m <- mean(predict(m1, newdata=new_m))

png("outputs/plot_predicted_salary_gender_m1.png", width=1200, height=900, res=180)
par(mar=c(5,5,3,2))
vals <- c(yhat_f, yhat_m)
bp <- barplot(vals, names.arg=c("Female","Male"), col=c(female_col, male_col),
              ylim=c(0, max(vals)*1.15), ylab="Predicted Salary 1995 (USD)",
              main="Predicted Salary by Gender (m1, same covariates)")
text(bp, vals, labels=format(round(vals,0), big.mark=","), pos=3, cex=1)
dev.off()

# ---------- (4) Observed vs Predicted (m2 & m3) ----------
png("outputs/plot_obs_vs_pred_m2.png", width=1600, height=1000, res=180)
par(mar=c(5,5,3,2))
yhat2 <- predict(m2); y2 <- df$Log_Sal94
plot(yhat2, y2, pch=19, col=ink, xlab="Predicted Log Salary 1994 (m2)", ylab="Observed Log Salary 1994",
     main=sprintf("Observed vs Predicted — m2  (Adj R² = %.3f)", summary(m2)$adj.r.squared))
abline(0,1, col=accent, lwd=3); grid(col=grid_col)
dev.off()

png("outputs/plot_obs_vs_pred_m3.png", width=1600, height=1000, res=180)
par(mar=c(5,5,3,2))
yhat3 <- predict(m3); y3 <- df$Log_Sal94
plot(yhat3, y3, pch=19, col=ink, xlab="Predicted Log Salary 1994 (m3)", ylab="Observed Log Salary 1994",
     main=sprintf("Observed vs Predicted — m3  (Adj R² = %.3f)", summary(m3)$adj.r.squared))
abline(0,1, col=accent, lwd=3); grid(col=grid_col)
dev.off()

# ---------- Predicted Salary bar charts for m2 & m3 (Log_Sal94 models) ----------
if (!exists("female_col")) female_col <- "#E8A6A6"
if (!exists("male_col"))   male_col   <- "#9EB9D8"
if (!exists("ink"))        ink        <- "#333333"

base_dat <- df
new_f <- base_dat; new_f$Gender <- factor("Female", levels=levels(df$Gender))
new_m <- base_dat; new_m$Gender <- factor("Male",   levels=levels(df$Gender))

# ---- m2: Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept + Rank ----
smear2 <- mean(exp(residuals(m2)), na.rm=TRUE)  # Duan smearing factor
pred_f2 <- mean(smear2 * exp(predict(m2, newdata=new_f)))
pred_m2 <- mean(smear2 * exp(predict(m2, newdata=new_m)))

png("outputs/plot_predicted_salary_gender_m2.png", width=1200, height=900, res=180)
par(mar=c(5,5,3,2))
vals2 <- c(pred_f2, pred_m2)
bp2 <- barplot(vals2, names.arg=c("Female","Male"),
               col=c(female_col, male_col),
               ylim=c(0, max(vals2)*1.15),
               ylab="Predicted Salary 1994 (USD)",
               main="Predicted Salary by Gender — m2 (back-transformed)")
text(bp2, vals2, labels=format(round(vals2,0), big.mark=","), pos=3, cex=1)
dev.off()

# ---- m3: Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept ----
smear3 <- mean(exp(residuals(m3)), na.rm=TRUE)
pred_f3 <- mean(smear3 * exp(predict(m3, newdata=new_f)))
pred_m3 <- mean(smear3 * exp(predict(m3, newdata=new_m)))

png("outputs/plot_predicted_salary_gender_m3.png", width=1200, height=900, res=180)
par(mar=c(5,5,3,2))
vals3 <- c(pred_f3, pred_m3)
bp3 <- barplot(vals3, names.arg=c("Female","Male"),
               col=c(female_col, male_col),
               ylim=c(0, max(vals3)*1.15),
               ylab="Predicted Salary 1994 (USD)",
               main="Predicted Salary by Gender — m3 (back-transformed)")
text(bp3, vals3, labels=format(round(vals3,0), big.mark=","), pos=3, cex=1)
dev.off()


# ---------------------------
# 8. Diagnostic plots (Base R)
# ---------------------------
par(mfrow = c(2, 2)); plot(m0, col = "gray40",   pch = 19); par(mfrow = c(1, 1))
par(mfrow = c(2, 2)); plot(m1, col = "lightblue", pch = 19); par(mfrow = c(1, 1))
par(mfrow = c(2, 2)); plot(m2, col = "orange",    pch = 19); par(mfrow = c(1, 1))
par(mfrow = c(2, 2)); plot(m3, col = "tomato",    pch = 19); par(mfrow = c(1, 1))

cat("\nAnalysis complete. CSV files saved in 'outputs/' folder.\n")
