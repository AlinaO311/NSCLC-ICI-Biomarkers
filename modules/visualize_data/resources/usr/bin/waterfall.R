#!/usr/bin/Rscript
rm(list=ls())
library(dplyr)
library(devtools)
library(optparse)
library(data.table)
library(reshape2)
library(stringr)
library(GenVisR)

options <- list(
  make_option(c("-i", "--data_path"),help="Dataset for building model" ),
  make_option(c("-m", "--mutation_data"),help="File containing list of mutations"),
)

### Handle arguments

outfile <- "waterfall_plot.png"

parser <- OptionParser(usage="%prog [options] file", option_list=options)
args <- parse_args(parser, positional_arguments = 0)
opt <- args$options
infile <- opt$data_path
mutfile <- opt$mutation_data

# Set the current working directory and file path
mutData <- file.path(mutfile, self$mutfile)
 dir.create(outfile)

set.seed(426)

###### Read patient data
meta_tbl <- read.table(opt$infile,sep="\t",header=T)

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
  mutate(Mutation_Type = if_else(str_detect(Mutation_Type, "\\."), "multiple", Mutation_Type))

# Reorder and rename columns for the final dataset
setnames(melted, c("sample", "gene", "variant_class", "counts"))

######################### ADD clinical data

# Select clinical variables to include in plot - durable clinical benefit response map to responder and non-responder
clinicalData <- meta_tbl %>% 
  select(PATIENT_ID, DURABLE_CLINICAL_BENEFIT, HISTOLOGY) %>%
  mutate(DURABLE_CLINICAL_BENEFIT = recode(DURABLE_CLINICAL_BENEFIT, `0` = "NonResponder", `1` = "Responder"))

# Rename columns to match melted mut data
colnames(clinicalData) <- c("sample", "Best_Response", "Histology")

# Melt clinical data to merge with mutation data
clinicalData_2 <- melt(data=clinicalData, id.vars=c("sample"))

# Create a separate data frame for mutation burden
mutationBurden <- meta_tbl %>% select(PATIENT_ID, TMB)
colnames(mutationBurden) <- c("sample", "TMB")

# find which samples are not in the mutationBurden data frame
sampleVec <- unique(melted$sample)
sampleVec[!sampleVec %in% clinicalData_2$sample]

# Create a vector to save mutation priority order for plotting
mutation_priority <- as.character(unique(melted$variant_class))

# Create waterfall plot
plot <- waterfall(melted, fileType = "Custom", 
variant_class_order=mutation_priority,
mutBurden=mutationBurden, 
clinData=clinicalData_2, clinLegCol=3,  
section_heights=c(1, 5, 1), mainRecurCutoff = 0.05, maxGenes = 20)


####### Crete waterfall plot image
#pdf(file.path(outfile,"gene_mds_plot.pdf"))
pdf("gene_mds_plot.pdf", height=10, width=15)
waterfall(melted, fileType = "Custom", variant_class_order=mutation_priority,clinData=clinicalData_2, clinLegCol=3, section_heights=c(1,5,1), mainRecurCutoff = 0.05, maxGenes = 20)
dev.off()