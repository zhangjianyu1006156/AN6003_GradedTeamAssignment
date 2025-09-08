### STEP 0: Setup
setwd("~/Documents/MSBA - Trimester 1/AN6003 - Analytics Strategy - Course Materials/Graded Team Assignment - Gender Discrimination Lawsuit")

library(data.table)
library(ggplot2)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

# fread without stringsAsFactors
data1 <- data.table::fread("Lawsuit.csv")
str(data1)
head(data1)

### STEP 1: Data Preparation

# Check missing values
colSums(is.na(data1))

# ---- Convert categorical variables ----

# Dept
data1$Dept <- factor(data1$Dept,
                     levels = c(1,2,3,4,5,6),
                     labels = c("Biochemistry/Molecular Biology",
                                "Physiology",
                                "Genetics",
                                "Pediatrics",
                                "Medicine",
                                "Surgery"))

# Gender
data1$Gender <- factor(data1$Gender,
                       levels = c(0, 1),
                       labels = c("Female", "Male"))

# Rank
data1$Rank <- factor(data1$Rank,
                     levels = c(1, 2, 3),
                     labels = c("Assistant", "Associate", "Full professor"))

# Clin
data1$Clin <- factor(data1$Clin,
                     levels = c(0, 1),
                     labels = c("Primarily research emphasis",
                                "Primarily clinical emphasis"))

# Cert
data1$Cert <- factor(data1$Cert,
                     levels = c(0, 1),
                     labels = c("Not certified", "Board certified"))

# ---- Global options to avoid scientific notation ----
options(scipen = 999)

# Convert any remaining character columns to factors
data1 <- data1 %>%
  mutate(across(where(is.character), as.factor))

# Check Dept distribution
table(data1$Dept)


###STEP 2: Data Exploration

# Gender vs. Income
boxplot(Sal94 ~ Gender, data = data1,
        main = "Salary Distribution by Gender (1994)",
        xlab = "Gender",
        ylab = "Salary (1994)",
        col = c("lightpink", "lightblue"),
        axes = FALSE) # Boxplot Salary vs Gender with full y-axis numbers
axis(1, at = 1:2, labels = c("Female", "Male")) # Custom x-axis
axis(2, at = pretty(data1$Sal94),
     labels = format(pretty(data1$Sal94), big.mark = ",", scientific = FALSE)) # Custom y-axis with no scientific notation
box() # Draw box around plot

# Gender - Rank vs. Income
ggplot(data1, aes(x = Gender, y = Sal94, fill = Gender)) +
  geom_boxplot() +
  facet_wrap(~ Rank) +
  labs(
    title = "Salary Distribution by Gender within Each Rank (1994)",
    x = "Gender", 
    y = "Salary (1994)"
  ) +
  scale_fill_manual(values = c("lightpink", "lightblue"))

# Gender - Dept vs. Income
ggplot(data1, aes(x = Gender, y = Sal94, fill = Gender)) +
  geom_boxplot() +
  facet_wrap(~ Dept) +
  labs(
    title = "Salary Distribution by Gender within Each Rank (1994)",
    x = "Gender", 
    y = "Salary (1994)"
  ) +
  scale_fill_manual(values = c("lightpink", "lightblue"))

# Gender - Cert vs. Income
ggplot(data1, aes(x = Gender, y = Sal94, fill = Gender)) +
  geom_boxplot() +
  facet_wrap(~ Cert) +
  labs(
    title = "Salary Distribution by Gender within Each Rank (1994)",
    x = "Gender", 
    y = "Salary (1994)"
  ) +
  scale_fill_manual(values = c("lightpink", "lightblue"))


# Distribution of YOE
ggplot(data1, aes(x = Exper, fill = Gender)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "white") +
  facet_wrap(~ Gender, ncol = 1) +
  labs(
    title = "Distribution of Experience by Gender",
    x = "Years since MD",
    y = "Count"
  ) +
  theme_minimal(base_size = 14)
# Male generally has more YOE compared to Female, making other observations biased

# Gender - Rank
ggplot(data1, aes(x = Rank, fill = Gender)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Number of People by Gender within Each Rank",
    x = "Rank",
    y = "Count"
  ) +
  scale_fill_manual(values = c("lightpink", "lightblue"))
# Male generally are more likely to be full professor compared to Female

1 Dept: 1=Biochemistry/Molecular Biology 2=Physiology 3=Genetics 4=Pediatrics 5=Medicine 6=Surgery
2 Gender: 1=Male, 0=Female
3 Clin: 1=Primarily clinical emphasis, 0=Primarily research emphasis
4 Cert: 1=Board certified, 0=not certified
5 Prate: Publication rate (# publications on cv)/(# years between CV date and MD date)
6 Exper: # years since obtaining MD
7 Rank: 1=Assistant, 2=Associate, 3=Full professor (a proxy for productivity)
8 Sal94: Salary in academic year 1994
9 Sal95: Salary after increment to 1994
  

### STEP 3: Linear Regression (Salary outcome)
# Salary 1994 regression
lm1 <- lm(Sal94 ~ Gender + Dept + Clin + Cert + Prate + Exper + Rank, data = data1)
summary(lm1)

# Salary 1995 regression
lm2 <- lm(Sal95 ~ Gender + Dept + Clin + Cert + Prate + Exper + Rank, data = data1)
summary(lm2)x

# No direct relationship between Gender and Salary if taking all factors into consideration
# As YOE is a big factor, categorizing them for further understanding
# However, this does not rule out indirect impacts on salary, such as chances of getting promotion


table(data1$Rank)

### STEP 4: Logistic Regression (Rank outcome)
# Recode Rank as binary (Full vs Not Full)
data1$FullProf <- ifelse(data1$Rank == "Full professor", 1, 0)

logit1 <- glm(FullProf ~ Gender + Dept + Clin + Cert + Prate + Exper,
              data = data1, family = binomial)
summary(logit1)

# Recode Rank as binary (Assistant vs Not Assistant)
data1$Assistant <- ifelse(data1$Rank == "Assistant", 1, 0)

logit2 <- glm(Assistant ~ Gender + Dept + Clin + Cert + Prate + Exper,
              data = data1, family = binomial)
summary(logit2)

# It is true that Assistant Prof are more likely to be Female, and Full Prof are more likely to be Male


### STEP 5: CART Decision Tree
cart_gender <- rpart(Rank ~ Gender + Exper + Dept + Clin + Cert + Prate,
                     data = data1, method = "class")
rpart.plot(cart_gender, type = 3, extra = 104)

# --- 1) Create comparable experience brackets: <=8, 8-16, >=16
data1$Exper_cat <- cut(
  data1$Exper,
  breaks = c(-Inf, 8, 16, Inf),
  right  = FALSE,                             # [lower, upper) so 8 goes to "8–16"
  labels = c("≤8 yrs", "8–16 yrs", "≥16 yrs")
)

# Make sure the order is consistent in plots/tables
data1$Exper_cat <- factor(data1$Exper_cat, levels = c("≤8 yrs", "8–16 yrs", "≥16 yrs"))

# Quick check
table(data1$Exper_cat, useNA = "ifany")


cart_f <- rpart(Rank ~ Dept + Clin + Cert + Prate + Exper_cat,
                data = subset(data1, Gender == "Female"), method = "class")
cart_m <- rpart(Rank ~ Dept + Clin + Cert + Prate + Exper_cat,
                data = subset(data1, Gender == "Male"),   method = "class")
par(mfrow = c(1, 2))
rpart.plot(cart_f, type = 3, extra = 104, main = "Females")
rpart.plot(cart_m, type = 3, extra = 104, main = "Males")
par(mfrow = c(1, 1))