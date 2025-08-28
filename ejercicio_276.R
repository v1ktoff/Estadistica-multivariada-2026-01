install.packages("datarium")
library(datarium)
df <- marketing
summary(df)
modeloM <- lm(formula = sales ~ youtube + facebook, data = df)

summary(modeloM)

par(mfrow = c(2, 2))
plot(modeloM)
bptest(modeloM)

modeloM2 <- lm(formula = sales ~ log(youtube) + facebook + newspaper, data = df)
par(mfrow = c(2, 2))
plot(modeloM)
bptest(modeloM2)
