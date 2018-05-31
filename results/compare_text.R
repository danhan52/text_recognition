# change working directory - should be folder containing three other folders:
# gold standard folder, consensus/machine folder, save folder
setwd("C:/Users/danny/Repos/text_recognition/results/data/GS_I_C_comparison_files/")

# use library stringdist for Levenshtein distance
library(readtext)


#' Get the character error rate for each line of text in a specific file in
#' two different folders.
#' It removes the consensus score at the end of the line e.g. [1.0/1].
#' You can choose to remove metadata tags.
#' It returns a data frame with the matching lines of text and the CER
compare_file <- function(gs_folder, other_folder, filename,
                         remove_tags = FALSE) {
  # read in text and convert to vectors
  text_gs <- strsplit(readtext(paste0(gs_folder, filename))$text, "\n")[[1]]
  text_other <- strsplit(readtext(paste0(other_folder, filename))$text, "\n")[[1]]
  # remove the consensus score
  text_gs <- gsub(pattern = " \\[[0-9]\\.[0-9]+/[0-9]\\]", replacement = "",
                  x = text_gs)
  text_other <- gsub(pattern = " \\[[0-9]\\.[0-9]+/[0-9]\\]", replacement = "",
                     x = text_other)
  # remove the metadata tags
  if (remove_tags) {
    meta_tags <- c("underline", "insertion", "deletion", "unclear")
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
  
  # get the number of lines (to compare)
  number_lines <- c(length(text_gs), length(text_other))
  
  # get best "other" line for each gold standard
  best_ind <- numeric(length(text_gs))
  best_text <- character(length(text_gs))
  cers_ignore_case <- numeric(length(text_gs))
  cers <- numeric(length(text_gs))
  if (length(text_other) <= 0) {
    cer_data <- data.frame(gs_text = text_gs,
                           other_text = best_text,
                           other_line_number = best_ind)
    return(list(cer_data, number_lines, c(0, 0)))
  }
  for (i in 1:length(text_gs)) {
    best_ind[i] <- which.min(as.numeric(adist(text_gs[i], text_other,
                                               ignore.case = T)))
    best_text[i] <- text_other[best_ind[i]]
    cers[i] <- adist(text_gs[i], best_text[i])/max(nchar(text_gs[i]), 1)
    cers_ignore_case[i] <- adist(text_gs[i], best_text[i], ignore.case = T)/
      max(nchar(text_gs[i]), 1)
  }
  
  # combine as data frame
  cer_data <- data.frame(gs_text = text_gs,
                         other_text = best_text,
                         other_line_number = best_ind,
                         cer = cers,
                         cer_ignore_case = cers_ignore_case)
  
  cer_means <- c(mean(cer_data$cer), mean(cer_data$cer_ignore_case))
  list(cer_data, number_lines, cer_means)
}

#' go through all the files in the folder and compare them
#' Gives a few summary statistics in a text file
compare_folders <- function(gs_folder, other_folder, save_folder) {
  filenames <- dir(gs_folder)
  filenames <- filenames[grepl(".txt", filenames)]
  for (fn in filenames) {
    cat(fn, "\n")
    text_data <- compare_file(gs_folder, other_folder, fn)
    
    write.csv(text_data[[1]], paste0(save_folder, gsub(".txt", ".csv", fn)), row.names = F)
    cat("gold standard number of lines: ", text_data[[2]][1], "\n",
        gsub("/", "", other_folder), " number of lines: ", text_data[[2]][2], "\n",
        "line number ratio (other/gs): ", text_data[[2]][2]/text_data[[2]][1], "\n\n",
        "mean error rate: ", text_data[[3]][1], "\n",
        "mean error rate (ignore case): ", text_data[[3]][2], "\n",
        sep = "", 
        file = paste0(save_folder, gsub(".txt", "_other_details.txt", fn)))
  }
}


dir.create("collaborative_error", showWarnings = F)
compare_folders(gs_folder = "gold_standard/",
                other_folder = "collaborative/",
                save_folder = "collaborative_error/")


dir.create("individual_error", showWarnings = F)
compare_folders(gs_folder = "gold_standard/",
                other_folder = "individual/",
                save_folder = "individual_error/")
