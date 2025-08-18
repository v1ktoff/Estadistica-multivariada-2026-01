df = datasets::trees
df
m2 <- lm(Volume~Girth+Height,data=df)
m2
idv <- rep(1, nrow(df))
# Creamos matriz X
X <- matrix(c(idv,df$Girth,df$Height),nrow=31,ncol=3)
# Creamos el vector y
y <- matrix(df$Volume, nrow = 31, ncol = 1)
# El estimador de \hat{\beta} = (X'X)^{-1}X'Y
beta <- solve(t(X)%*%X) %*% t(X) %*% y
beta
SSE <- t(y)%*%y-t(beta)%*%t(X)%*%y
SSE
# Grados de libertad  np-1
varest <- SSE/(nrow(y)-nrow(beta))
varest
sum(residuals(m2)^2)/df.residual(m2)
write.csv(df, "trees.csv", row.names = FALSE)
