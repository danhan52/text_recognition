library(ggplot2)

setwd("C:/Users/danny/Repos/text_recognition/modeling/")
data <- read.csv("tf_output/online_data/modified_data.csv")

data <- data[data$cer < 1.0,]
unique(data$bunch)

bunchmean <- aggregate(cer ~ bunch, data = data, FUN = mean)
bunchmean$bunch <- bunchmean$bunch / 1000

svg("tf_output/online_data/training.svg", width=8, height=8)
ggplot(bunchmean, aes(x=bunch, y=cer)) +
  geom_smooth(method=lm, se=FALSE, size = 2.5) +
  geom_point(size=4.0) +
  labs(x = "Batch of 1000 classifications", y = "Mean Character Error Rate",
       title = "Character Error Rate During Training") +
  theme(title=element_text(size=22, face="bold"),
        axis.title=element_text(size=16),
        axis.text = element_text(size=14))
dev.off()

hist(data$cer)
abline(h)