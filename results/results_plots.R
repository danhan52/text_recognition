library(ggplot2)

setwd("C:/Users/danny/Repos/text_recognition/results/")


plot_batch <- function(type, trpr, batch, maxbatch=-1, size=25) {
  fn = paste0("/metrics_batch", batch, ".csv")
  size_ttl = ""
  xlabel = "Epoch"
  divby = 1000
  if (type == "file_size") {
    data_batch <- read.csv(paste0("data/file_size/size", size, fn))
    size_ttl = paste(" - size:", size/100)
  } else if (type == "online") {
    data_batch <- read.csv(paste0("data/online_training", fn))
    xlabel = "Batch of 1000"
  } else if (type == "official") {
    data_batch <- read.csv(paste0("data/official_training", fn))
    divby = 1
  } else {
    return()
  }
  # data_image <- read.csv(paste0("data/file_size/size", size, "/metrics_image5000.csv"))
  
  # do graphs for batch based calculations
  data <- data_batch[data_batch$pred == trpr,]
  if (maxbatch > 0) {
    data <- data[data$tr_group <= maxbatch,]
  }
  data$tr_group <- data$tr_group/divby + 0.5
  cermean <- aggregate(cer ~ tr_group, data = data, FUN = mean)
  ttl <- ifelse(trpr == "train", "Training", "Validation")
  
  # get epoch size to add unsmoothed data in background
  epoch_size = sum(data$tr_group == 0.5)
  x_full = (1:nrow(data))/epoch_size
  
  ggplot(cermean, aes(x=tr_group, y=cer)) +
    geom_line(data=data, aes(x=x_full, y=cer), color="#CC79A7", size=0.5) +
    geom_line(size=1.0, color="#0072B2") +
    geom_point(size=3.0, color="#0072B2") +
    labs(x = xlabel, y = "Mean Character Error Rate",
         title = paste(ttl, "CER", size_ttl)) +
    theme(title=element_text(size=16, face="bold"),
          axis.title=element_text(size=12),
          axis.text = element_text(size=10)) +
    ylim(0, 1) +
    scale_x_continuous(breaks=0:(max(cermean$tr_group)+0.5), minor_breaks=NULL)
}
#

# graphs for image size experiment ####
svg("images/img_size/cer_train25.svg", width=4, height=4)
plot_batch("file_size", "train", 5000, 4000, 25)
dev.off()
svg("images/img_size/cer_train50.svg", width=4, height=4)
plot_batch("file_size", "train", 5000, 4000, 50)
dev.off()
svg("images/img_size/cer_train100.svg", width=4, height=4)
plot_batch("file_size", "train", 5000, 4000, 100)
dev.off()

svg("images/img_size/cer_pred25.svg", width=4, height=4)
plot_batch("file_size", "pred", 5000, 4000, 25)
dev.off()
svg("images/img_size/cer_pred50.svg", width=4, height=4)
plot_batch("file_size", "pred", 5000, 4000, 50)
dev.off()
svg("images/img_size/cer_pred100.svg", width=4, height=4)
plot_batch("file_size", "pred", 5000, 4000, 100)
dev.off()

# graphs for official training ####
svg("images/official_training/pred.svg", width=4, height=4)
plot_batch("official", "pred", 9)
dev.off()
svg("images/official_training/train.svg", width=4, height=4)
plot_batch("official", "train", 9)
dev.off()

# trying out an official plot for online experiment ####
batch = 41000; trpr = "pred"
fn = paste0("/metrics_batch", batch, ".csv")
size_ttl = ""
xlabel = "Epoch"
divby = 1000
data_batch <- read.csv(paste0("data/online_training", fn))
data_batch <- data_batch[data_batch$cer >0,]

data <- data_batch[data_batch$pred == trpr,]
data$tr_group <- data$tr_group/divby + 0.5
cermean <- aggregate(cer ~ tr_group, data = data, FUN = mean)
ttl <- ifelse(trpr == "train", "Training", "Validation")

# get epoch size to add unsmoothed data in background
epoch_size = sum(data$tr_group == 0.5)
x_full = (1:nrow(data))/epoch_size

ggplot(cermean, aes(x=tr_group, y=cer)) +
  geom_line(data=data, aes(x=x_full, y=cer), color="#CC79A7", size=0.5) +
  geom_line(size=1.0, color="#0072B2") +
  geom_point(size=3.0, color="#0072B2") +
  labs(x = xlabel, y = "Mean Character Error Rate",
       title = paste(ttl, "CER", size_ttl)) +
  theme(title=element_text(size=16, face="bold"),
        axis.title=element_text(size=12),
        axis.text = element_text(size=10)) +
  ylim(0, 1) +
  scale_x_continuous(breaks=0:(max(cermean$tr_group)+0.5), minor_breaks=NULL)


# temporary plot for online experiment ####
data <- read.csv("data/online_training/modified_pred.csv")
data <- read.csv("data/online_training/modified_train.csv")

data <- data[data$cer < 1.0,]
data <- data[data$cer > 0,]
unique(data$bunch)

bunchmean <- aggregate(cer ~ bunch, data = data, FUN = mean)
bunchmean$bunch <- bunchmean$bunch / 1000

svg("images/online_training/pred.svg", width=8, height=8)
ggplot(bunchmean, aes(x=bunch, y=cer)) +
  geom_smooth(method=lm, se=FALSE, size = 2.5) +
  geom_point(size=4.0) +
  labs(x = "Batch of 1000 classifications", y = "Mean Character Error Rate",
       title = "Validation CER") +
  theme(title=element_text(size=22, face="bold"),
        axis.title=element_text(size=16),
        axis.text = element_text(size=14))
dev.off()



# online experiment results ####

trpr="pred"; batch=12000; maxbatch=-1; size=25
fn = paste0("/metrics_batch", batch, ".csv")
size_ttl = ""
divby = 1000
data_batch <- read.csv(paste0("data/online_training", fn))
xlabel = "Batch of 1000"
# data_image <- read.csv(paste0("data/file_size/size", size, "/metrics_image5000.csv"))

# do graphs for batch based calculations
data <- data_batch[data_batch$pred == trpr,]
if (maxbatch > 0) {
  data <- data[data$tr_group <= maxbatch,]
}
data <- data[data$cer > 0,]
data$tr_group <- data$tr_group/divby - 0.5
cermean <- aggregate(cer ~ tr_group, data = data, FUN = mean)
ttl <- ifelse(trpr == "train", "Training", "Validation")

# get epoch size to add unsmoothed data in background
epoch_size = sum(data$tr_group == 0.5)
x_full = (1:nrow(data))/epoch_size

ggplot(cermean, aes(x=tr_group, y=cer)) +
  geom_line(data=data, aes(x=x_full, y=cer), color="#CC79A7", size=0.5) +
  geom_line(size=1.0, color="#0072B2") +
  geom_point(size=3.0, color="#0072B2") +
  labs(x = xlabel, y = "Mean Character Error Rate",
       title = paste(ttl, "CER", size_ttl)) +
  theme(title=element_text(size=16, face="bold"),
        axis.title=element_text(size=12),
        axis.text = element_text(size=10)) +
  ylim(0, 1) +
  scale_x_continuous(breaks=0:(max(cermean$tr_group)+0.5), minor_breaks=NULL)
