#!/usr/bin/env Rscript
rm(list=ls())
library(dplyr)
library(devtools)
library(optparse)
library(data.table)
library(reshape2)
library(tidyverse)
library(stringr)
library(gridExtra)
library(grid)
library(tableone)

set.seed(123)

options <- list(
  make_option(c("-i", "--data_path"),help="Dataset for building model" ),
  make_option(c("-m", "--mutation_data"),help="File containing list of mutations"), 
  make_option(c("-o", "--outputdir"),help="File containing patient data")
)

# Get the current working directory and split at 'work'
cwd <- str_split(getwd(), "work", n = 2)[[1]][1]

# Set directory path and time_val from environment variable
time_val <- Sys.getenv("DATE_VALUE")

### Handle arguments

parser <- OptionParser(usage="%prog [options] file", option_list=options)
args <- parse_args(parser, positional_arguments = 0)
opt <- args$options
infile <- opt$data_path
mutfile <- opt$mutation_data
dir <- opt$outputdir

# Construct output paths
output_path <- file.path(cwd, dir, "DataPrep")
output_folder <- file.path(output_path, paste0("visualization_", trimws(time_val)))

# Set the current working directory and file path
mutData <- file.path(cwd, mutfile)

###### Read patient data
data <- read.table(infile, sep="\t",header=T)

# Select clinical variables to include in plot - durable clinical benefit response map to responder and non-responder
clinicalData <- data %>% 
  select(AGE, SEX, DURABLE_CLINICAL_BENEFIT, HISTOLOGY, PDL1_EXP,SMOKING_HISTORY, TMB,  HISTOLOGY) %>%
  mutate(DURABLE_CLINICAL_BENEFIT = recode(DURABLE_CLINICAL_BENEFIT, `0` = "NonResponder", `1` = "Responder")) %>%
  mutate(HISTOLOGY = ifelse(is.na(HISTOLOGY) | HISTOLOGY == "", "Unknown", HISTOLOGY))%>%
  # Order by DURABLE_CLINICAL_BENEFIT
  arrange(DURABLE_CLINICAL_BENEFIT) 

# Dynamically detect numeric and categorical columns
numeric_cols <- sapply(clinicalData, is.numeric)
categorical_cols <- !numeric_cols

# Define the categorical and continuous variables
cat_vars <- names(clinicalData)[categorical_cols]
cont_vars <- names(clinicalData)[numeric_cols]

# Test for normality using Shapiro-Wilk test (p-value < 0.05 indicates non-normal distribution)
non_normal_vars <- cont_vars[sapply(clinicalData[cont_vars], function(x) shapiro.test(x)$p.value < 0.05)]

# Create Table 1 using the tableone package
table_one <- CreateTableOne(
  vars = c(cat_vars, cont_vars),
  data = clinicalData,
  strata = "DURABLE_CLINICAL_BENEFIT", includeNA = TRUE, addOverall = TRUE)

tabPrint <- tableGrob(print(table_one, nonnormal = non_normal_vars, formatOptions = list(big.mark = ","), quote = FALSE, noSpaces = TRUE, printToggle = FALSE))

# Measure Table Dimensions
pdf(NULL)  # Open a null PDF device
table_width <- convertWidth(grobWidth(tabPrint), "in", valueOnly = TRUE)
table_height <- convertHeight(grobHeight(tabPrint), "in", valueOnly = TRUE)
dev.off()

# Save the Table 1 as a plot image
output_file <- file.path(output_folder, "table_one_plot.png")

# Save as PNG with Dynamically Adjusted Size
png(output_file, width =  10 * 150, height = table_height * 200, res = 100)
grid.draw(tabPrint)
dev.off()