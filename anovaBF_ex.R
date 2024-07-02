setwd("~/卒業論文/卒論 H201015 データファイル/分析プログラミングファイル/Pythonfiles_bachelor_thesis")
library(reticulate)
reticulate::repl_python()

quit
library(BayesFactor)

df_mu <- py$df_mu
char_culumn <- sapply(df_mu, is.character)
df_mu[char_culumn] <- lapply(df_mu[char_culumn], as.factor)

set.seed(1)
bf_result <- anovaBF(power ~ practice * cond + ID, data = df_mu, whichRandom = "ID")

# Print Bayes Factor
print(bf_result) # BF10 Alternative/Null
print(1 / bf_result) # BF01 Null/Alternative
plot(bf_result)
