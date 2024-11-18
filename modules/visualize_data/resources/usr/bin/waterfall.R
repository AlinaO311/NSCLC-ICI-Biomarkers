#!/usr/bin/env Rscript
rm(list=ls())
library(dplyr)
library(devtools)
library(optparse)
library(data.table)
library(reshape2)
library(tidyverse)
library(stringr)
library(GenVisR)


options <- list(
  make_option(c("-i", "--data_path"),help="Dataset for building model" ),
  make_option(c("-m", "--mutation_data"),help="File containing list of mutations"), 
  make_option(c("-o", "--outputdir"),help="File containing patient data")
)

# Get the current working directory and split at 'work'
cwd <- str_split(getwd(), "work", n = 2)[[1]][1]

# Set directory path and time_val from environment variable
time_val <- Sys.getenv("time_val")

### Handle arguments

outfile <- "waterfall_plot.png"

parser <- OptionParser(usage="%prog [options] file", option_list=options)
args <- parse_args(parser, positional_arguments = 0)
opt <- args$options
infile <- opt$data_path
mutfile <- opt$mutation_data
dir <- opt$outputdir

# Construct output paths
output_path <- file.path(cwd, dir, "DataPrep")
output_folder <- file.path(output_path, paste0("visualization_", time_val))

# Set the current working directory and file path
mutData <- file.path(cwd, mutfile)

set.seed(426)

###### Read patient data
meta_tbl <- read.table(infile, sep="\t",header=T)

###### Read mutation data and clean the list
mut_list <- scan(mutData, what = "", sep = ",", quiet = TRUE) # Split on commas

###### Create a data frame with mutation data
# Extract targets
targets <- mut_list[str_detect(mut_list, "=")]

# Extract only the prefix part before the "=" character
prefix_list <- gsub("=\\[.*", "", targets)

# Replace strings before '=' or "[" in each list and trim whitespace
cleaned_out_list <- gsub(".*=\\[|\\]", "", mut_list)  # Remove text before "["
cleaned_out_list <- trimws(cleaned_out_list) 

# Flatten the list of lists
flatmuts <- c(cleaned_out_list, prefix_list)

# Mask columns that start with selected mutation strings
mask <- grepl(paste(flatmuts, collapse = "|"), colnames(meta_tbl))
mdata <- meta_tbl[, mask, drop = FALSE]

# Add 'PATIENT_ID' column back to the filtered data
mdata$PATIENT_ID <- meta_tbl$PATIENT_ID

# Melt and split 'category' column in one step
## Replace col names in mdata to replace values with commas with 'multiple'
melted <- mdata %>%
  select(-contains("muts"), -contains("biomarkers")) %>%
  # melt data on PATIENT_ID, where gene and consequence are split
  pivot_longer(cols = -PATIENT_ID, names_to = "category", values_to = "value") %>%
  # split category into gene and mutation columns
  separate(category, into = c("gene", "Mutation_Type"), sep = "_",  extra = "merge")  %>%
  mutate(Mutation_Type = if_else(str_detect(Mutation_Type, "\\."), "multiple", Mutation_Type)) %>%
  distinct() %>%
  filter(value != 0)

# Reorder and rename columns for the final dataset
setnames(melted, c("sample", "gene", "variant_class", "counts"))

######################### ADD clinical data

# Load the required library
library(colorspace)

# Number of unique mutation categories (you need this to match the number of colors)
n_colors <- length(unique(melted$Mutation_Type))  # Assuming 'Mutation_Type' has the mutation categories

palette <- qualitative_hcl(n_colors, palette = "Dark3")

# Generate a palette with enough colors
mutationColours <- setNames(palette, unique(melted$Mutation_Type))

# Select clinical variables to include in plot - durable clinical benefit response map to responder and non-responder
clinicalData <- meta_tbl %>% 
  select(PATIENT_ID, DURABLE_CLINICAL_BENEFIT, HISTOLOGY) %>%
  mutate(DURABLE_CLINICAL_BENEFIT = recode(DURABLE_CLINICAL_BENEFIT, `0` = "NonResponder", `1` = "Responder")) %>%
  mutate(HISTOLOGY = ifelse(is.na(HISTOLOGY) | HISTOLOGY == "", "Unknown", HISTOLOGY))%>%
  # Order by DURABLE_CLINICAL_BENEFIT
  arrange(DURABLE_CLINICAL_BENEFIT)

# Rename columns to match melted mut data
colnames(clinicalData) <- c("sample", "Best_Response", "Histology")

# Melt clinical data to merge with mutation data
clinicalData_2 <- melt(data=clinicalData, id.vars=c("sample"))

# Create a separate data frame for mutation burden
mutationBurden <- meta_tbl %>% select(PATIENT_ID, TMB) %>%
  filter(PATIENT_ID %in% melted$sample)

colnames(mutationBurden) <- c("sample", "TMB")

# find which samples are not in the mutationBurden data frame
sampleVec <- unique(melted$sample)
sampleVec[!sampleVec %in% clinicalData_2$sample]

# Create a vector to save mutation priority order for plotting
mutation_priority <- as.character(unique(melted$variant_class))

clinVarOrder_map <- c(unique(clinicalData$Best_Response[order(clinicalData$Best_Response)]), unique(clinicalData$Histology[order(clinicalData$Histology)]))

# Create waterfall plot
#pdf(file.path(outfile,"gene_mds_plot.pdf"))
# Specify the full path for the PDF output file
output_file <- file.path(output_folder, "waterfall_plot.png")

png(output_file, height=12, width=15, units="in", res=300)
waterfall(melted, fileType = "Custom",  variant_class_order=mutation_priority , clinData=clinicalData_2,clinVarOrder=clinVarOrder_map, clinLegCol=ncol(clinicalData)-1, section_heights=c(1,5,1), mainRecurCutoff = 0.05,   mainPalette=mutationColours)
dev.off()
