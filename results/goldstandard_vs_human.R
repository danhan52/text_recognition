library(ggplot2)

setwd("C:/Users/danny/Repos/text_recognition/results/")

data <- read.csv("data/volunteer_vs_gs.csv", sep = "\t")

mean(data$cer)

svg("images/gs_vs_volunteer.svg", width=5, height=4)
qplot(data$cer, geom="histogram") +
  labs(x = "CER", y = "Count",
       title = "CER of Volunteers vs. Experts")
dev.off()