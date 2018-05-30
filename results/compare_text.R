# change working directory - should be folder containing three other folders:
# gold standard folder, consensus/machine folder, save folder
setwd("C:/Users/danny/Repos/text_recognition/results/data/")

# use library stringdist for Levenshtein distance
library(stringdist)
library(readtext)


# Basic CER function ####
#' Get character error rate of two vectors of text. 
#' The first vector (gs) is the "correct" text, which could be gold standard or
#' otherwise.
#' The second vector (other) is the text for which you are trying to determine 
#' the error rate.
cer <- function(gs, other) {
  require(stringdist)
  edit_dist <- stringdist(gs, other, method = "lv")
  gs_length <- max(nchar(gs), 1)
  
  edit_dist/gs_length
}


# Get CER for entire text file ####
#' Get the character error rate for each line of text in a specific file in
#' two different folders.
#' It removes the consensus score at the end of the line e.g. [1.0/1].
#' You can choose to remove metadata tags.
#' It returns a data frame with the matching lines of text and the CER
transcription_cer <- function(gs_folder, other_folder, filename, 
                              remove_tags = FALSE) {
  # read in text and convert to vectors
  text_gs <- strsplit(readtext(paste0(gs_folder, filename))$text, "\n")[[1]]
  text_other <- strsplit(readtext(paste0(other_folder, filename))$text, "\n")[[1]]
  # remove the consensus score
  text_gs <- gsub(pattern = " \\[[0-9]\\.[0-9]/[0-9]\\]", replacement = "",
                  x = text_gs)
  text_other <- gsub(pattern = " \\[[0-9]\\.[0-9]/[0-9]\\]", replacement = "",
                     x = text_other)
  # remove the metadata tags
  if (remove_tags) {
    meta_tags <- c("underline", "insertion")
    for (m in meta_tags) {
      # remove opening tag
      text_other <- gsub(pattern = paste0("\\[", m, "\\]"),
                         replacement = "",
                         x = text_other)
      text_gs <- gsub(pattern = paste0("\\[", m, "\\]"),
                      replacement = "",
                      x = text_gs)
      # remove closing tag
      text_gs <- gsub(pattern = paste0("\\[/", m, "\\]"),
                      replacement = "",
                      x = text_gs)
      text_other <- gsub(pattern = paste0("\\[/", m, "\\]"),
                      replacement = "",
                      x = text_other)
    }
  }
  # creat data frame
  text_data <- data.frame("gs" = text_gs, "other" = text_other, 
                          stringsAsFactors = F) # must be string, not factor
  
  # get CER
  text_data$cer <- cer(text_data$gs, text_data$other)
  
  return(text_data)
}


# get CER info for every file in folder ####
run_all_cer <- function(gs_folder, other_folder, save_folder) {
  filenames <- dir(gs_folder)
  filenames <- filenames[grepl(".txt", filenames)]
  
  for (fn in filenames) {
    text_data <- transcription_cer(gs_folder, other_folder, fn)
    write.csv(text_data, paste0(save_folder, gsub(".txt", ".csv", fn)), row.names = F)
  }
}


# run CER comparison ####
run_all_cer(gs_folder = "gold_standard_april_25_2018/",
            other_folder = "people_folder/",
            save_folder = "cer_output/")
