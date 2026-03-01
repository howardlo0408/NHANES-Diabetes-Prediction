# 載入套件
library(tidyverse)
library(haven)
library(survey)
library(caret)
library(randomForest)

# 讀取 xpt 檔並挑選需要的欄位
diq <- read_xpt("DIQ_J.XPT") %>% select(SEQN, DIQ010, DIQ160, DIQ170, DIQ172)
glu <- read_xpt("GLU_J.XPT") %>% select(SEQN, LBXGLU)
ghb <- read_xpt("GHB_J.XPT") %>% select(SEQN, LBXGH)
diet <- read_xpt("DR1TOT_J.XPT") %>% select(SEQN, DR1TKCAL, DR1TSUGR)
paq <- read_xpt("PAQ_J.XPT") %>% select(SEQN, PAD615)
bmx <- read_xpt("BMX_J.XPT") %>% select(SEQN, BMXBMI)
demo <- read_xpt("DEMO_J.XPT") %>% select(SEQN, RIDAGEYR, RIAGENDR, RIDRETH1, WTMEC2YR, WTINT2YR, SDMVSTRA, SDMVPSU)

# 合併所有資料 (用 SEQN 當 key)
nh <- reduce(list(diq, glu, ghb, diet, paq, bmx, demo), ~left_join(.x, .y, by = "SEQN"))

# 前處理: 篩選未得糖尿病(DIQ010==2)，排除 NA
no_dx <- nh %>%
  filter(DIQ010 == 2, 
         !is.na(LBXGLU), !is.na(LBXGH), !is.na(DIQ160), !is.na(BMXBMI), 
         !is.na(DIQ172), !is.na(PAD615), !is.na(DR1TKCAL)) %>%
  mutate(
    fh_yes = (DIQ160 == 1),           
    risk_self = (DIQ172 == 1),        
    bmi = BMXBMI,
    active_150 = ifelse(PAD615 >= 150, 1, 0), # 運動量有沒有達標
    RIAGENDR = as.factor(RIAGENDR)
  )

# 設定 NHANES 抽樣權重
des1 <- svydesign(id = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR, nest = TRUE, data = no_dx)

# --- 統計檢定區 (原本的期末報告) ---

# Q1: 家族史對血糖的影響
mean_glu <- svyby(~ LBXGLU + LBXGH, ~ fh_yes, des1, svymean, vartype = "ci")
mean_glu
glu_t <- svyttest(LBXGLU ~ fh_yes, des1)
glu_t

# Q2A: 自覺風險對 BMI 的影響
bmi_t <- svyttest(bmi ~ risk_self, des1)
bmi_t

# Q2B: 加權多變項線性回歸 (預測 BMI)
model_2b <- svyglm(bmi ~ risk_self + RIDAGEYR + RIAGENDR + DR1TKCAL + PAD615, design = des1)
summary(model_2b)

# Q2C: 加權 logistic 回歸 (預測運動達標)
model_2c <- svyglm(active_150 ~ risk_self + RIDAGEYR + RIAGENDR + DR1TKCAL + bmi, 
                   design = des1, family = quasibinomial())
summary(model_2c)


# --- 機器學習區 (新增的前期糖尿病預測) ---

# 定義目標變數: 血糖>=100 或 HbA1c>=5.7 就是前期糖尿病
ml_data <- no_dx %>%
  mutate(
    Pre_Diabetes = ifelse(LBXGLU >= 100 | LBXGH >= 5.7, "HighRisk", "Normal"),
    Pre_Diabetes = as.factor(Pre_Diabetes),
    fh_yes = as.factor(fh_yes),
    risk_self = as.factor(risk_self)
  ) %>%
  select(Pre_Diabetes, fh_yes, risk_self, RIDAGEYR, RIAGENDR, bmi, PAD615, DR1TKCAL) %>%
  drop_na() 

# 切割訓練集跟測試集 (8-2 分)
set.seed(123)
train_idx <- createDataPartition(ml_data$Pre_Diabetes, p = 0.8, list = FALSE)
train_data <- ml_data[train_idx,]
test_data  <- ml_data[-train_idx,]

# 跑 Random Forest (用 5-fold CV)
ctrl <- trainControl(method = "cv", number = 5)
rf_model <- train(
  Pre_Diabetes ~ ., 
  data = train_data, 
  method = "rf",
  trControl = ctrl,
  importance = TRUE 
)

# 預測測試集並印出結果
rf_pred <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_pred, test_data$Pre_Diabetes)

# 畫特徵重要性圖
plot(varImp(rf_model), main = "Feature Importance")