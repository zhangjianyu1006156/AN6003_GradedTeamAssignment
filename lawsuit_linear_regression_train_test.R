# ===============================================================
# Lawsuit - Linear Regression (Train/Test Split + CSV + Visuals)
# Scope: Base R only (plus 'car' for VIF).
# What this script does
#   - Reads Lawsuit.csv
#   - Casts factors & creates log transforms
#   - 80/20 train-test split (seed fixed for reproducibility)
#   - Fits four models on TRAIN: m0_tr, m1_tr, m2_tr, m3_tr
#   - Evaluates on TEST (RMSE / R2) incl. Duan smearing for log models
#   - Saves TRAIN coefficients per model + a wide table
#   - Exports key plots for slides (gender effect bars, forests, obs vs pred)
#   - All outputs go to ./outputs/
# ===============================================================

suppressPackageStartupMessages(library(car))

# ----------- Set working directory -----------
# Mac example:
setwd("/Users/zhangjianyu/Desktop/学习课件/NTU/AN6003 Analytics Strategy/AN6003 Course Materials/AN6003_GradedTeamAssignment")
# Windows example:
# setwd("C:/Users/Zhang/OneDrive - Nanyang Technological University/桌面/NTU学习/AN6003 Course Materials/AN6003 Course Materials/Graded Team Assignment - Gender Discrimination Lawsuit/AN6003_GradedTeamAssignment")

# 0. Setup
dir.create("outputs", showWarnings = FALSE)

# 1. Read data
df <- read.csv("Lawsuit.csv", stringsAsFactors = FALSE)

# 2. Clean and cast types
df$Dept <- factor(df$Dept,
                  levels = 1:6,
                  labels = c("Biochem/MolBio", "Physiology", "Genetics",
                             "Pediatrics", "Medicine", "Surgery"))
df$Gender <- factor(df$Gender, levels = c(0, 1), labels = c("Female", "Male"))
df$Clin   <- factor(df$Clin,   levels = c(0, 1), labels = c("Research", "Clinical"))
df$Cert   <- factor(df$Cert,   levels = c(0, 1), labels = c("NotCert", "Cert"))
df$Rank   <- factor(df$Rank,   levels = c(1, 2, 3), labels = c("Assistant", "Associate", "Full"))

# Log transforms
df$Log_Sal94  <- log(df$Sal94)
df$Log_Sal95  <- log(df$Sal95)
df$Log_Exper  <- log1p(df$Exper)
df$Log_Prate  <- log1p(df$Prate)

# 3. Basic exploration (console + CSV)
cat("\n=== Basic Summary of Dataset ===\n")
print(summary(df))

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

gender_counts <- as.data.frame(table(df$Gender))
colnames(gender_counts) <- c("Gender", "Count")
write.csv(gender_counts, file = "outputs/gender_counts.csv", row.names = FALSE)

dept_gender_tbl <- as.data.frame(table(Dept = df$Dept, Gender = df$Gender))
write.csv(dept_gender_tbl, file = "outputs/gender_by_dept_counts.csv", row.names = FALSE)

# 4. Train/Test split (80/20)
set.seed(42)
n <- nrow(df); idx <- sample.int(n); n_tr <- floor(0.6 * n)
train_idx <- idx[1:n_tr]; test_idx <- idx[(n_tr + 1):n]
train <- df[train_idx, ]
test  <- df[test_idx, ]

# 5. Models (fit on TRAIN only)
m0_tr <- lm(Sal95 ~ Gender, data = train)
m1_tr <- lm(Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin, data = train)
m2_tr <- lm(Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept + Rank, data = train)
m3_tr <- lm(Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept, data = train)

cat("\n=== TRAIN (80%) summaries ===\n")
cat("\n=== m0_tr ===\n"); print(summary(m0_tr))
cat("\n=== m1_tr ===\n"); print(summary(m1_tr)); print(vif(m1_tr))
cat("\n=== m2_tr ===\n"); print(summary(m2_tr)); print(vif(m2_tr))
cat("\n=== m3_tr ===\n"); print(summary(m3_tr)); print(vif(m3_tr))

# 6. Out-of-sample evaluation (TEST)
oos_metrics <- function(y_true, y_pred) {
  rmse <- sqrt(mean((y_true - y_pred)^2))
  r2   <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
  c(RMSE = rmse, R2 = r2)
}

# m1 (levels) on TEST
yhat_m1_test <- predict(m1_tr, newdata = test)
met_m1 <- oos_metrics(test$Sal95, yhat_m1_test)

# Duan smearing (from TRAIN residuals) for log models
smear2 <- mean(exp(residuals(m2_tr)), na.rm=TRUE)
smear3 <- mean(exp(residuals(m3_tr)), na.rm=TRUE)

# m2/m3 on TEST: log-scale and $-scale
yhat_m2_test_log <- predict(m2_tr, newdata = test)
yhat_m3_test_log <- predict(m3_tr, newdata = test)

yhat_m2_test_lvl <- smear2 * exp(yhat_m2_test_log)
yhat_m3_test_lvl <- smear3 * exp(yhat_m3_test_log)

met_m2_log <- oos_metrics(test$Log_Sal94, yhat_m2_test_log)
met_m3_log <- oos_metrics(test$Log_Sal94, yhat_m3_test_log)
met_m2_lvl <- oos_metrics(test$Sal94,    yhat_m2_test_lvl)
met_m3_lvl <- oos_metrics(test$Sal94,    yhat_m3_test_lvl)

test_metrics <- data.frame(
  model = c("m1 (level) - $",
            "m2 (log) - log", "m2 (log) - $",
            "m3 (log) - log", "m3 (log) - $"),
  RMSE  = c(met_m1["RMSE"], met_m2_log["RMSE"], met_m2_lvl["RMSE"],
            met_m3_log["RMSE"], met_m3_lvl["RMSE"]),
  R2    = c(met_m1["R2"],   met_m2_log["R2"],   met_m2_lvl["R2"],
            met_m3_log["R2"],   met_m3_lvl["R2"])
)
write.csv(test_metrics, "outputs/test_metrics_oos.csv", row.names = FALSE)
print(test_metrics)

# 7. Helpers for CSV & labels
pstars <- function(p) ifelse(p < .001, "***",
                             ifelse(p < .01,  "**",
                                    ifelse(p < .05,  "*", "")))
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
  dfc$cell <- sprintf("%.2f%s", dfc$est, dfc$star)
  dfc
}

# Save per-model TRAIN coefficients
save_model_coefs <- function(model, name) {
  sm <- summary(model)$coefficients
  out <- data.frame(term = rownames(sm),
                    est = sm[,1], se = sm[,2], t = sm[,3], p = sm[,4], row.names = NULL)
  write.csv(out, file = paste0("outputs/train_", name, "_coefficients.csv"), row.names = FALSE)
}
save_model_coefs(m0_tr, "m0")
save_model_coefs(m1_tr, "m1")
save_model_coefs(m2_tr, "m2")
save_model_coefs(m3_tr, "m3")

# Wide table (TRAIN models)
t0 <- transform(tidy_from_lm(m0_tr), model = "m0_tr: Sal95 ~ Gender")
t1 <- transform(tidy_from_lm(m1_tr), model = "m1_tr: Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin")
t2 <- transform(tidy_from_lm(m2_tr), model = "m2_tr: Log_Sal94 ~ Gender + Clin + Cert + log_Prate + log_Exper + Dept + Rank")
t3 <- transform(tidy_from_lm(m3_tr), model = "m3_tr: Log_Sal94 ~ Gender + Clin + Cert + log_Prate + log_Exper + Dept")
t_all <- rbind(t0, t1, t2, t3); t_all$term_pretty <- pretty_term(t_all$term)
col_order <- c("Constant","Gender","Clin","Cert","log_Prate","log_Exper",
               "Physiology","Genetics","Pediatrics","Medicine","Surgery","Associate","Full Professor")
models_vec <- unique(t_all$model)
wide <- data.frame(Model = models_vec, check.names = FALSE); for (cn in col_order) wide[[cn]] <- ""
fill_one <- function(df_model, target_row) {
  for (i in seq_len(nrow(df_model))) {
    nm <- df_model$term_pretty[i]; val <- df_model$cell[i]
    if (nm %in% col_order) wide[target_row, nm] <<- val
  }
}
for (i in seq_along(models_vec)) fill_one(subset(t_all, model == models_vec[i]), i)
wide$R2     <- round(c(summary(m0_tr)$r.squared, summary(m1_tr)$r.squared, summary(m2_tr)$r.squared, summary(m3_tr)$r.squared), 3)
wide$Adj_R2 <- round(c(summary(m0_tr)$adj.r.squared, summary(m1_tr)$adj.r.squared, summary(m2_tr)$adj.r.squared, summary(m3_tr)$adj.r.squared), 3)
write.csv(wide, "outputs/train_models_coefficients_wide.csv", row.names = FALSE)

## ---------- helper (overwrite): horizontal USD forest with Gender highlight ----------
make_usd_barplot_horiz <- function(fit, model_label, baseline, file,
                                   highlight_term = "Gender") {
  sm  <- summary(fit)
  cf  <- sm$coefficients
  cf  <- cf[rownames(cf) != "(Intercept)", , drop = FALSE]
  lab <- pretty_term(rownames(cf))
  b   <- cf[, 1]; p <- cf[, 4]
  
  # Δ$ ≈ baseline * (exp(β)-1)
  usd <- baseline * (exp(b) - 1)
  ord <- order(usd, decreasing = TRUE)
  usd <- usd[ord]; lab <- lab[ord]; p <- p[ord]
  
  cols <- ifelse(lab == highlight_term, "#1f77b4", "gray78")
  
  fmt_lbl <- function(u, p) {
    paste0(ifelse(u >= 0, "+$", "-$"),
           format(round(abs(u), 0), big.mark = ","),
           ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", ""))))
  }
  
  png(file, width = 1800, height = 1200, res = 180)
  par(mar = c(8, 18, 9, 6))
  xlim <- range(c(-baseline * 0.2, usd)) * 1.15
  
  plot.new(); plot.window(xlim = xlim, ylim = c(0.5, length(usd) + 0.5))
  abline(v = 0, col = "gray60", lwd = 2)
  abline(v = pretty(xlim), col = "gray90", lwd = 1)
  
  y <- barplot(usd, horiz = TRUE, names.arg = rep("", length(usd)),
               xlim = xlim, col = cols, border = "#8f8f8f", add = TRUE)
  
  axis(2, at = y, labels = lab, las = 2, tick = FALSE, cex.axis = 1.2)
  offs <- diff(par("usr")[1:2]) * 0.015
  text(ifelse(usd >= 0, usd + offs, usd - offs), y,
       labels = mapply(fmt_lbl, usd, p),
       adj = ifelse(usd >= 0, 0, 1), xpd = NA, cex = 1.1,
       col = ifelse(lab == highlight_term, "#1f77b4", "black"))
  
  mtext("Multiple Linear Regression", side = 3, line = 6.2,
        cex = 1.6, font = 2, col = "#2B50F3")
  mtext(model_label, side = 3, line = 4.6, cex = 1.25, font = 2, col = "#2B50F3")
  
  fs  <- sm$fstatistic
  mpv <- pf(fs[1], fs[2], fs[3], lower.tail = FALSE)
  mtext(sprintf("R² = %.1f%%   Adj R² = %.1f%%   F = %.1f (p = %.1e)",
                sm$r.squared*100, sm$adj.r.squared*100, fs[1], mpv),
        side = 3, line = 3.2, cex = 1.1)
  
  mtext(sprintf("x-axis: Estimated impact on salary (USD) @ baseline $%s",
                format(round(baseline, 0), big.mark = ",")),
        side = 1, line = 4.8, cex = 1.05)
  mtext(sprintf("Note: USD effects shown at baseline $%s; effects scale with salary level.",
                format(round(baseline, 0), big.mark = ",")),
        side = 1, line = 3.2, cex = 1.0)
  box()
  dev.off()
}

## ---------- call: generate the exact figure for m2 ----------
baseline <- round(mean(df$Sal94, na.rm = TRUE))  # ~153,593

make_usd_barplot_horiz(
  fit = m2,
  model_label = "Model m2 (Log_Sal94, with Rank)",
  baseline = baseline,
  file = "outputs/m2_usd_barplot.png",
  highlight_term = "Gender"
)


# 8. Visuals for slides
female_col <- "pink"; male_col <- "lightblue"
ink <- "#333333"; grid_col <- "#D9D9D9"; accent <- "#6FA8DC"

# (1) Gender effect: m0_tr ($) vs m3_tr (%)
png("outputs/plot_gender_effect_m0_vs_m3.png", width=1800, height=900, res=180)
layout(matrix(1:2, nrow=1))

# m0_tr: USD difference (Male - Female)
par(mar=c(5,6,3,2), xaxs="i")
ci0 <- suppressMessages(confint(m0_tr))
b0  <- coef(m0_tr)["GenderMale"]
lo0 <- ci0["GenderMale",1]; hi0 <- ci0["GenderMale",2]
xlim0 <- range(c(lo0, hi0))*1.1
plot(NA, xlim=xlim0, ylim=c(0.5,1.5), yaxt="n",
     xlab="Difference in 1995 Salary (USD) — TEST not used", ylab="", main="m0_tr: Raw gender gap ($)",
     col.axis=ink, col.lab=ink, col.main=ink)
abline(v=0, col=grid_col, lwd=2)
segments(lo0,1, hi0,1, col=accent, lwd=6, lend=1)
points(b0,1, pch=19, cex=2, col=accent)
axis(2, at=1, labels="Male - Female", las=1, col.axis=ink)

# m3_tr: percent effect
par(mar=c(5,6,3,2), xaxs="i")
ci3 <- suppressMessages(confint(m3_tr))
b3  <- coef(m3_tr)["GenderMale"]
lo3 <- ci3["GenderMale",1]; hi3 <- ci3["GenderMale",2]
pct <- (exp(b3)-1)*100; loP <- (exp(lo3)-1)*100; hiP <- (exp(hi3)-1)*100
xlim3 <- range(c(loP, hiP))*1.1
plot(NA, xlim=xlim3, ylim=c(0.5,1.5), yaxt="n",
     xlab="Estimated % difference in Salary (1994)", ylab="", main="m3_tr: Within-structure premium (%)",
     col.axis=ink, col.lab=ink, col.main=ink)
abline(v=0, col=grid_col, lwd=2)
segments(loP,1, hiP,1, col=accent, lwd=6, lend=1)
points(pct,1, pch=19, cex=2, col=accent)
axis(2, at=1, labels="Male vs Female", las=1, col.axis=ink)
dev.off()

# helper: simplified forest
simple_forest <- function(model, title, terms_keep, to_percent=FALSE) {
  sm <- summary(model); cf <- sm$coefficients; ci <- suppressMessages(confint(model))
  keep <- terms_keep[terms_keep %in% rownames(cf)]
  est <- cf[keep,"Estimate"]; lo <- ci[keep,1]; hi <- ci[keep,2]
  xlab <- "Coefficient"
  if (to_percent) {
    ind <- grepl("^Gender|^Clin|^Cert|^Dept", keep)
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

png("outputs/plot_forest_simple_m1.png", width=1600, height=1200, res=180)
simple_terms_m1 <- c("GenderMale","RankAssociate","RankFull",
                     "DeptPhysiology","DeptGenetics","DeptPediatrics","DeptMedicine","DeptSurgery",
                     "Exper")
simple_forest(m1_tr, "Simplified Forest — m1_tr (levels, key drivers)", simple_terms_m1, to_percent=FALSE)
dev.off()

png("outputs/plot_forest_simple_m3.png", width=1600, height=1200, res=180)
simple_terms_m3 <- c("GenderMale","ClinClinical","CertCert",
                     "DeptPhysiology","DeptGenetics","DeptPediatrics","DeptMedicine","DeptSurgery",
                     "Log_Exper")
simple_forest(m3_tr, "Simplified Forest — m3_tr (log; indicators in %)", simple_terms_m3, to_percent=TRUE)
dev.off()

# (3) Predicted Salary bars for m2_tr and m3_tr (back-transformed)
new_f <- df; new_f$Gender <- factor("Female", levels=levels(df$Gender))
new_m <- df; new_m$Gender <- factor("Male",   levels=levels(df$Gender))

pred_f2 <- mean(smear2 * exp(predict(m2_tr, newdata=new_f)))
pred_m2 <- mean(smear2 * exp(predict(m2_tr, newdata=new_m)))
pred_f3 <- mean(smear3 * exp(predict(m3_tr, newdata=new_f)))
pred_m3 <- mean(smear3 * exp(predict(m3_tr, newdata=new_m)))

png("outputs/plot_predicted_salary_gender_m2.png", width=1200, height=900, res=180)
par(mar=c(5,5,3,2))
vals2 <- c(pred_f2, pred_m2)
bp2 <- barplot(vals2, names.arg=c("Female","Male"),
               col=c(female_col, male_col),
               ylim=c(0, max(vals2)*1.15),
               ylab="Predicted Salary 1994 (USD)",
               main="Predicted Salary by Gender — m2_tr (back-transformed)")
text(bp2, vals2, labels=format(round(vals2,0), big.mark=","), pos=3, cex=1)
dev.off()

png("outputs/plot_predicted_salary_gender_m3.png", width=1200, height=900, res=180)
par(mar=c(5,5,3,2))
vals3 <- c(pred_f3, pred_m3)
bp3 <- barplot(vals3, names.arg=c("Female","Male"),
               col=c(female_col, male_col),
               ylim=c(0, max(vals3)*1.15),
               ylab="Predicted Salary 1994 (USD)",
               main="Predicted Salary by Gender — m3_tr (back-transformed)")
text(bp3, vals3, labels=format(round(vals3,0), big.mark=","), pos=3, cex=1)
dev.off()

# (4) Observed vs Predicted on TEST (log scale)
png("outputs/plot_obs_vs_pred_m2_TEST.png", width=1600, height=1000, res=180)
par(mar=c(5,5,3,2))
yhat2_test <- predict(m2_tr, newdata=test); y2_test <- test$Log_Sal94
oos_r2_m2 <- 1 - sum((y2_test - yhat2_test)^2) / sum((y2_test - mean(y2_test))^2)
plot(yhat2_test, y2_test, pch=19, col="#333333",
     xlab="Predicted Log Salary 1994 (m2_tr, TEST)", ylab="Observed Log Salary 1994 (TEST)",
     main=sprintf("Observed vs Predicted — m2_tr (Out-of-sample R² = %.3f)", oos_r2_m2))
abline(0,1, col="#6FA8DC", lwd=3); grid(col="#D9D9D9")
dev.off()

png("outputs/plot_obs_vs_pred_m3_TEST.png", width=1600, height=1000, res=180)
par(mar=c(5,5,3,2))
yhat3_test <- predict(m3_tr, newdata=test); y3_test <- test$Log_Sal94
oos_r2_m3 <- 1 - sum((y3_test - yhat3_test)^2) / sum((y3_test - mean(y3_test))^2)
plot(yhat3_test, y3_test, pch=19, col="#333333",
     xlab="Predicted Log Salary 1994 (m3_tr, TEST)", ylab="Observed Log Salary 1994 (TEST)",
     main=sprintf("Observed vs Predicted — m3_tr (Out-of-sample R² = %.3f)", oos_r2_m3))
abline(0,1, col="#6FA8DC", lwd=3); grid(col="#D9D9D9")
dev.off()

best_idx <- which.min(test_metrics$RMSE)
best_model <- test_metrics$model[best_idx]
cat("Best model (lowest RMSE):", best_model, "\n")


par(mfrow=c(1,2))

# 1) log scale
barplot(
  height=c(met_m2_log["RMSE"], met_m3_log["RMSE"]),
  names.arg=c("m2 (log)", "m3 (log)"),
  col="lightblue", ylim=c(0,0.5),
  main="RMSE on Log Scale", ylab="RMSE (log)"
)
text(x=1:2, y=c(met_m2_log["RMSE"], met_m3_log["RMSE"]),
     labels=round(c(met_m2_log["RMSE"], met_m3_log["RMSE"]),3), pos=3)

# 2) $ scale
barplot(
  height=c(met_m1["RMSE"], met_m2_lvl["RMSE"], met_m3_lvl["RMSE"]),
  names.arg=c("m1 ($)", "m2 ($)", "m3 ($)"),
  col="lightgreen", ylim=c(0,35000),
  main="RMSE on Dollar Scale", ylab="RMSE (USD)"
)
text(x=1:3, y=c(met_m1["RMSE"], met_m2_lvl["RMSE"], met_m3_lvl["RMSE"]),
     labels=round(c(met_m1["RMSE"], met_m2_lvl["RMSE"], met_m3_lvl["RMSE"]),1), pos=3)

# ---------------------------
# Save RMSE to outputs/
# ---------------------------

# 1) log RMSE
png("outputs/rmse_log_models.png", width=1200, height=900, res=150)
barplot(
  height=c(met_m2_log["RMSE"], met_m3_log["RMSE"]),
  names.arg=c("m2 (log)", "m3 (log)"),
  col="lightblue", ylim=c(0,0.5),
  main="RMSE on Log Scale", ylab="RMSE (log)"
)
text(x=1:2, y=c(met_m2_log["RMSE"], met_m3_log["RMSE"]),
     labels=round(c(met_m2_log["RMSE"], met_m3_log["RMSE"]),3), pos=3)
dev.off()

# 2) Dollar RMSE
png("outputs/rmse_dollar_models.png", width=1200, height=900, res=150)
barplot(
  height=c(met_m1["RMSE"], met_m2_lvl["RMSE"], met_m3_lvl["RMSE"]),
  names.arg=c("m1 ($)", "m2 ($)", "m3 ($)"),
  col="lightblue", ylim=c(0,35000),
  main="RMSE on Dollar Scale", ylab="RMSE (USD)"
)
text(x=1:3, y=c(met_m1["RMSE"], met_m2_lvl["RMSE"], met_m3_lvl["RMSE"]),
     labels=round(c(met_m1["RMSE"], met_m2_lvl["RMSE"], met_m3_lvl["RMSE"]),1), pos=3)
dev.off()

cat("\nAnalysis complete. CSV files and PNGs are saved in 'outputs/'.\n")
