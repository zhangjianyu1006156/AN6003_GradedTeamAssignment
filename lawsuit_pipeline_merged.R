# lawsuit_pipeline_merged.R
# Clean, single-file pipeline for the AN6003 Lawsuit analysis
# - English comments, minimal dependencies
# - Consistent train/test, models, CSV exports, and ggplot visuals
# - Safe logs (log1p where needed) and Duan smearing for back-transform
# - Outputs saved into ./outputs

# ========== 0) Libraries ==========
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(scales)
})

# ========== 1) Working directory & I/O ==========
# Edit ONE of the paths below if needed; we auto-pick the first that exists.
candidate_dirs <- c(
  "/Users/zhangjianyu/Desktop/学习课件/NTU/AN6003 Analytics Strategy/AN6003 Course Materials/AN6003_GradedTeamAssignment",
  "C:/Users/Zhang/OneDrive - Nanyang Technological University/桌面/NTU学习/AN6003 Course Materials/AN6003 Course Materials/Graded Team Assignment - Gender Discrimination Lawsuit/AN6003_GradedTeamAssignment",
  getwd()
)
setwd(candidate_dirs[dir.exists(candidate_dirs)][1])
if (!dir.exists("outputs")) dir.create("outputs")

stop_if_missing <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s (edit path in the script).", path), call. = FALSE)
}
stop_if_missing("Lawsuit.csv")

# ========== 2) Load & Clean Data ==========
df <- read.csv("Lawsuit.csv", stringsAsFactors = FALSE)

# Factors
df$Dept <- factor(df$Dept,
                  levels = 1:6,
                  labels = c("Biochem/MolBio", "Physiology", "Genetics",
                             "Pediatrics", "Medicine", "Surgery"))
df$Gender <- factor(df$Gender, levels = c(0, 1), labels = c("Female", "Male"))
df$Clin   <- factor(df$Clin,   levels = c(0, 1), labels = c("Research", "Clinical"))
df$Cert   <- factor(df$Cert,   levels = c(0, 1), labels = c("NotCert", "Cert"))
df$Rank   <- factor(df$Rank,   levels = c(1, 2, 3), labels = c("Assistant", "Associate", "Full"))

# Numerics
num_cols <- c("Sal94","Sal95","Prate","Exper")
df[num_cols] <- lapply(df[num_cols], function(x) as.numeric(x))

# Logs (use log1p where zero possible)
df <- df %>% mutate(
  Log_Sal94 = log(Sal94),      # 1994 salaries should be > 0
  Log_Sal95 = log(Sal95),      # 1995 salaries should be > 0
  Log_Exper = log1p(Exper),
  Log_Prate = log1p(Prate)
)

# ========== 3) Train/Test Split (80/20) ==========
set.seed(42)
n <- nrow(df)
test_idx <- sample.int(n, size = ceiling(0.2 * n))
train <- df[-test_idx, ]
test  <- df[test_idx, ]

# ========== 4) Models (train on 80%) ==========
m0_tr <- lm(Sal95 ~ Gender, data = train)
m1_tr <- lm(Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin, data = train)
m2_tr <- lm(Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept + Rank, data = train)
m3_tr <- lm(Log_Sal94 ~ Gender + Clin + Cert + Log_Prate + Log_Exper + Dept, data = train)

# Print summaries to console
cat("\n=== TRAIN (80%) summaries ===\n")
for (mm in list(m0_tr=m0_tr, m1_tr=m1_tr, m2_tr=m2_tr, m3_tr=m3_tr)) {
  cat("\n---", names(mm), "---\n"); print(summary(mm[[1]]))
}

# ========== 5) Out-of-sample metrics (on 20% test) ==========
rmse_vec <- function(actual, pred) sqrt(mean((actual - pred)^2, na.rm = TRUE))
r2_vec   <- function(actual, pred) {
  sse <- sum((actual - pred)^2, na.rm = TRUE)
  sst <- sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
  1 - sse/sst
}

# m1 level model => predict Sal95
yhat_m1 <- predict(m1_tr, newdata=test)
met_m1  <- c(RMSE = rmse_vec(test$Sal95, yhat_m1), R2 = r2_vec(test$Sal95, yhat_m1))

# m2, m3 log models => evaluate both in log and back-transformed $
yhat_m2_log <- predict(m2_tr, newdata=test)
yhat_m3_log <- predict(m3_tr, newdata=test)

smear2 <- mean(exp(residuals(m2_tr)), na.rm = TRUE)
smear3 <- mean(exp(residuals(m3_tr)), na.rm = TRUE)

yhat_m2_lvl <- smear2 * exp(yhat_m2_log)
yhat_m3_lvl <- smear3 * exp(yhat_m3_log)

met_m2_log <- c(RMSE = rmse_vec(test$Log_Sal94, yhat_m2_log), R2 = r2_vec(test$Log_Sal94, yhat_m2_log))
met_m3_log <- c(RMSE = rmse_vec(test$Log_Sal94, yhat_m3_log), R2 = r2_vec(test$Log_Sal94, yhat_m3_log))
met_m2_lvl <- c(RMSE = rmse_vec(test$Sal94,    yhat_m2_lvl), R2 = r2_vec(test$Sal94,    yhat_m2_lvl))
met_m3_lvl <- c(RMSE = rmse_vec(test$Sal94,    yhat_m3_lvl), R2 = r2_vec(test$Sal94,    yhat_m3_lvl))

test_metrics <- data.frame(
  model = c("m1 (level) - $", "m2 (log) - log", "m2 (log) - $", "m3 (log) - log", "m3 (log) - $"),
  RMSE  = c(met_m1["RMSE"], met_m2_log["RMSE"], met_m2_lvl["RMSE"], met_m3_log["RMSE"], met_m3_lvl["RMSE"]),
  R2    = c(met_m1["R2"],   met_m2_log["R2"],   met_m2_lvl["R2"],   met_m3_log["R2"],   met_m3_lvl["R2"])
)
write.csv(test_metrics, "outputs/test_metrics_oos.csv", row.names = FALSE)
print(test_metrics)

# ========== 6) Helper: tidy coefs with stars & pretty term labels ==========
pstars <- function(p) ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", "")))

tidy_from_lm <- function(model) {
  sm <- summary(model)$coefficients
  dfc <- data.frame(
    term = rownames(sm),
    est  = sm[, 1],
    se   = sm[, 2],
    t    = sm[, 3],
    p    = sm[, 4],
    row.names = NULL,
    check.names = FALSE
  )
  dfc$star <- pstars(dfc$p)
  dfc$cell <- sprintf("%.2f%s", dfc$est, dfc$star)  # e.g., 0.18***
  dfc
}

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

# Train-model coef tables (narrow and wide)
t0 <- transform(tidy_from_lm(m0_tr), model = "m0_tr: Sal95 ~ Gender")
t1 <- transform(tidy_from_lm(m1_tr), model = "m1_tr: Sal95 ~ Gender + Dept + Rank + Exper + Prate + Cert + Clin")
t2 <- transform(tidy_from_lm(m2_tr), model = "m2_tr: Log_Sal94 ~ Gender + Clin + Cert + log_Prate + log_Exper + Dept + Rank")
t3 <- transform(tidy_from_lm(m3_tr), model = "m3_tr: Log_Sal94 ~ Gender + Clin + Cert + log_Prate + log_Exper + Dept")

t_all <- rbind(t0, t1, t2, t3); t_all$term_pretty <- pretty_term(t_all$term)

col_order <- c("Constant","Gender","Clin","Cert","log_Prate","log_Exper",
               "Physiology","Genetics","Pediatrics","Medicine","Surgery","Associate","Full Professor")

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

wide$R2     <- round(c(summary(m0_tr)$r.squared, summary(m1_tr)$r.squared,
                       summary(m2_tr)$r.squared, summary(m3_tr)$r.squared), 3)
wide$Adj_R2 <- round(c(summary(m0_tr)$adj.r.squared, summary(m1_tr)$adj.r.squared,
                       summary(m2_tr)$adj.r.squared, summary(m3_tr)$adj.r.squared), 3)

write.csv(wide, "outputs/train_models_coefficients_wide.csv", row.names = FALSE)

# Also save individual train coefs
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

# ========== 7) Plots (ggplot) ==========

# 7.1 Gender effect: m0 (USD diff Male-Female) and m3 (% premium)
get_ci <- function(fit) {
  sm <- summary(fit)$coefficients
  ci <- confint(fit)
  data.frame(term = rownames(sm),
             est = sm[,1],
             lo = ci[,1],
             hi = ci[,2],
             row.names = NULL, check.names = FALSE)
}

# m0 in USD
ci0 <- get_ci(m0_tr) %>% filter(term == "GenderMale") %>%
  transmute(model = "m0: Raw gender gap ($)",
            label = "Male - Female",
            est = est, lo = lo, hi = hi)

# m3 in % (semi-elasticity)
ci3 <- get_ci(m3_tr) %>% filter(term == "GenderMale") %>%
  transmute(model = "m3: Within-structure premium (%)",
            label = "Male vs Female",
            est = (exp(est)-1)*100,
            lo = (exp(lo) -1)*100,
            hi = (exp(hi) -1)*100)

gg_gender <- bind_rows(ci0, ci3)

p_gender <- ggplot(gg_gender, aes(x = model, y = est)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.15) +
  geom_point(size = 3) +
  coord_flip() +
  labs(title = "Gender Effect: Raw $ (m0) vs % Premium (m3)",
       x = "", y = "Effect (USD for m0; % for m3)") +
  theme_minimal(base_size = 12)
ggsave("outputs/plot_gender_effect_m0_vs_m3.png", p_gender, width = 10, height = 5, dpi = 300)

# 7.2 Simplified coefficient "forest" plots

# Helper to build a coef/CI dataframe (optionally transform to % for indicators)
coef_forest_df <- function(fit, keep_terms = NULL, to_percent = FALSE) {
  sm <- summary(fit)$coefficients
  ci <- confint(fit)
  dfc <- data.frame(term = rownames(sm),
                    est = sm[,1], lo = ci[,1], hi = ci[,2],
                    row.names = NULL, check.names = FALSE)
  if (!is.null(keep_terms)) dfc <- dfc %>% filter(term %in% keep_terms)
  dfc <- dfc %>% filter(term != "(Intercept)")
  if (to_percent) {
    dfc <- dfc %>%
      mutate(est = 100*(exp(est)-1),
             lo  = 100*(exp(lo) -1),
             hi  = 100*(exp(hi) -1))
  }
  dfc$label <- dfc$term
  # Prettify labels
  dfc$label <- gsub("^GenderMale$","Gender",dfc$label)
  dfc$label <- gsub("^ClinClinical$","Clin",dfc$label)
  dfc$label <- gsub("^CertCert$","Cert",dfc$label)
  dfc$label <- gsub("^DeptPhysiology$","Physiology",dfc$label)
  dfc$label <- gsub("^DeptGenetics$","Genetics",dfc$label)
  dfc$label <- gsub("^DeptPediatrics$","Pediatrics",dfc$label)
  dfc$label <- gsub("^DeptMedicine$","Medicine",dfc$label)
  dfc$label <- gsub("^DeptSurgery$","Surgery",dfc$label)
  dfc$label <- gsub("^RankAssociate$","Associate",dfc$label)
  dfc$label <- gsub("^RankFull$","Full Professor",dfc$label)
  dfc$label <- gsub("^Exper$","Exper",dfc$label)
  dfc$label <- gsub("^Log_Exper$","log_Exper",dfc$label)
  dfc$label <- gsub("^Log_Prate$","log_Prate",dfc$label)
  dfc
}

simple_terms_m1 <- c("GenderMale","RankAssociate","RankFull",
                     "DeptPhysiology","DeptGenetics","DeptPediatrics","DeptMedicine","DeptSurgery",
                     "Exper")

df_m1 <- coef_forest_df(m1_tr, keep_terms = simple_terms_m1, to_percent = FALSE)

p_forest_m1 <- ggplot(df_m1, aes(x = reorder(label, est), y = est)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2) +
  geom_point(size = 2.5) +
  coord_flip() +
  labs(title = "Simplified Forest — m1 (levels, key drivers)",
       x = "", y = "Coefficient (USD units)") +
  theme_minimal(base_size = 12)
ggsave("outputs/plot_forest_simple_m1.png", p_forest_m1, width = 10, height = 8, dpi = 300)

simple_terms_m3 <- c("GenderMale","ClinClinical","CertCert",
                     "DeptPhysiology","DeptGenetics","DeptPediatrics","DeptMedicine","DeptSurgery",
                     "Log_Exper")

df_m3 <- coef_forest_df(m3_tr, keep_terms = simple_terms_m3, to_percent = TRUE)

p_forest_m3 <- ggplot(df_m3, aes(x = reorder(label, est), y = est)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2) +
  geom_point(size = 2.5) +
  coord_flip() +
  labs(title = "Simplified Forest — m3 (log; indicators in %)",
       x = "", y = "Effect size (%)") +
  theme_minimal(base_size = 12)
ggsave("outputs/plot_forest_simple_m3.png", p_forest_m3, width = 10, height = 8, dpi = 300)

# 7.3 Predicted Salary bar chart (same covariates; flip only Gender)
pred_base <- df
new_f <- pred_base; new_f$Gender <- factor("Female", levels=levels(df$Gender))
new_m <- pred_base; new_m$Gender <- factor("Male",   levels=levels(df$Gender))

yhat_f_m1 <- mean(predict(m1_tr, newdata=new_f))
yhat_m_m1 <- mean(predict(m1_tr, newdata=new_m))

p_pred_m1 <- data.frame(Gender=c("Female","Male"),
                        Pred=c(yhat_f_m1, yhat_m_m1)) %>%
  ggplot(aes(x=Gender, y=Pred, fill=Gender)) +
  geom_col() +
  geom_text(aes(label=scales::dollar(round(Pred,0))), vjust=-0.2) +
  scale_y_continuous(labels = dollar_format(), expand = expansion(mult = c(0, .15))) +
  guides(fill="none") +
  labs(title="Predicted Salary by Gender (m1, same covariates)", y="Predicted Salary 1995 (USD)", x="") +
  theme_minimal(base_size=12)
ggsave("outputs/plot_predicted_salary_gender_m1.png", p_pred_m1, width=8, height=6, dpi=300)

# For m2 & m3 (Log_Sal94 models) — back-transformed USD with Duan smearing
pred_f2 <- mean(smear2 * exp(predict(m2_tr, newdata=new_f)))
pred_m2 <- mean(smear2 * exp(predict(m2_tr, newdata=new_m)))
pred_f3 <- mean(smear3 * exp(predict(m3_tr, newdata=new_f)))
pred_m3 <- mean(smear3 * exp(predict(m3_tr, newdata=new_m)))

p_pred_m2 <- data.frame(Gender=c("Female","Male"),
                        Pred=c(pred_f2, pred_m2)) %>%
  ggplot(aes(x=Gender, y=Pred, fill=Gender)) +
  geom_col() +
  geom_text(aes(label=scales::dollar(round(Pred,0))), vjust=-0.2) +
  scale_y_continuous(labels = dollar_format(), expand = expansion(mult = c(0, .15))) +
  guides(fill="none") +
  labs(title="Predicted Salary by Gender — m2 (back-transformed)", y="Predicted Salary 1994 (USD)", x="") +
  theme_minimal(base_size=12)
ggsave("outputs/plot_predicted_salary_gender_m2.png", p_pred_m2, width=8, height=6, dpi=300)

p_pred_m3 <- data.frame(Gender=c("Female","Male"),
                        Pred=c(pred_f3, pred_m3)) %>%
  ggplot(aes(x=Gender, y=Pred, fill=Gender)) +
  geom_col() +
  geom_text(aes(label=scales::dollar(round(Pred,0))), vjust=-0.2) +
  scale_y_continuous(labels = dollar_format(), expand = expansion(mult = c(0, .15))) +
  guides(fill="none") +
  labs(title="Predicted Salary by Gender — m3 (back-transformed)", y="Predicted Salary 1994 (USD)", x="") +
  theme_minimal(base_size=12)
ggsave("outputs/plot_predicted_salary_gender_m3.png", p_pred_m3, width=8, height=6, dpi=300)

# 7.4 Observed vs Predicted (train & test) for m2 & m3 (log-scale)
mk_obs_pred_plot <- function(fit, data_x, y_log, title, file) {
  yhat <- predict(fit, newdata = data_x)
  dfp <- data.frame(pred = yhat, obs = y_log)
  p <- ggplot(dfp, aes(x=pred, y=obs)) +
    geom_point(alpha = 0.7) +
    geom_abline(slope=1, intercept=0, linewidth=1) +
    labs(title=title, x="Predicted (log)", y="Observed (log)") +
    theme_minimal(base_size=12)
  ggsave(file, p, width=10, height=6, dpi=300)
}

mk_obs_pred_plot(m2_tr, train, train$Log_Sal94,
                 sprintf("Observed vs Predicted — m2 TRAIN (Adj R² = %.3f)", summary(m2_tr)$adj.r.squared),
                 "outputs/plot_obs_vs_pred_m2_TRAIN.png")
mk_obs_pred_plot(m3_tr, train, train$Log_Sal94,
                 sprintf("Observed vs Predicted — m3 TRAIN (Adj R² = %.3f)", summary(m3_tr)$adj.r.squared),
                 "outputs/plot_obs_vs_pred_m3_TRAIN.png")

mk_obs_pred_plot(m2_tr, test, test$Log_Sal94,
                 "Observed vs Predicted — m2 TEST", "outputs/plot_obs_vs_pred_m2_TEST.png")
mk_obs_pred_plot(m3_tr, test, test$Log_Sal94,
                 "Observed vs Predicted — m3 TEST", "outputs/plot_obs_vs_pred_m3_TEST.png")

# 7.5 USD impact bars at baseline (median Sal94) for m2 & m3
baseline <- median(train$Sal94, na.rm = TRUE)
usd_impact_df <- function(fit, baseline) {
  sm <- summary(fit)$coefficients
  pv <- sm[,4]
  dfc <- data.frame(term = rownames(sm), beta = sm[,1], p = pv,
                    row.names = NULL, check.names = FALSE)
  dfc <- dfc %>% filter(term != "(Intercept)")
  dfc <- dfc %>% mutate(usd = baseline * (exp(beta) - 1),
                        label = paste0(ifelse(usd>=0, "+$", "-$"),
                                       format(round(abs(usd),0), big.mark=","),
                                       ifelse(p<.001,"***", ifelse(p<.01,"**", ifelse(p<.05,"*","")))))
  # Pretty labels
  dfc$label_left <- dfc$term
  dfc$label_left <- gsub("^GenderMale$","Gender", dfc$label_left)
  dfc$label_left <- gsub("^ClinClinical$","Clin", dfc$label_left)
  dfc$label_left <- gsub("^CertCert$","Cert", dfc$label_left)
  dfc$label_left <- gsub("^DeptPhysiology$","Physiology", dfc$label_left)
  dfc$label_left <- gsub("^DeptGenetics$","Genetics", dfc$label_left)
  dfc$label_left <- gsub("^DeptPediatrics$","Pediatrics", dfc$label_left)
  dfc$label_left <- gsub("^DeptMedicine$","Medicine", dfc$label_left)
  dfc$label_left <- gsub("^DeptSurgery$","Surgery", dfc$label_left)
  dfc$label_left <- gsub("^RankAssociate$","Associate", dfc$label_left)
  dfc$label_left <- gsub("^RankFull$","Full Professor", dfc$label_left)
  dfc$label_left <- gsub("^Log_Exper$","log_Exper", dfc$label_left)
  dfc$label_left <- gsub("^Log_Prate$","log_Prate", dfc$label_left)
  dfc
}

plot_usd_impact <- function(fit, baseline, title, file) {
  dfu <- usd_impact_df(fit, baseline) %>% arrange(usd)
  p <- ggplot(dfu, aes(x = reorder(label_left, usd), y = usd)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_col() +
    coord_flip() +
    geom_text(aes(label = label),
              hjust = ifelse(dfu$usd >= 0, -0.05, 1.05),
              size = 3.2) +
    scale_y_continuous(labels = dollar_format(),
                       expand = expansion(mult = c(0.05, 0.25))) +
    labs(title = paste0("USD Impact @ baseline ", dollar_format()(baseline), " — ", title),
         x = "", y = "Estimated $ impact") +
    theme_minimal(base_size = 12)
  ggsave(file, p, width = 10, height = 8, dpi = 300)
}

plot_usd_impact(m2_tr, baseline, "Model m2 (Log_Sal94, with Rank)",
                "outputs/m2_usd_barplot_pretty.png")
plot_usd_impact(m3_tr, baseline, "Model m3 (Log_Sal94, no Rank)",
                "outputs/m3_usd_barplot_pretty.png")

cat("\nAnalysis complete. CSVs & PNGs are in the 'outputs/' folder.\n")
