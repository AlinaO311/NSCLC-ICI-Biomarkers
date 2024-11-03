#!/home/xoroal/CancerKGG/nsclc_ici_biomarkers/modules/visualize_data/resources/usr/bin
rm(list=ls())
library(dplyr)
library(devtools)
library(optparse)
library(GenVisR)

options <- list(
  make_option(c("-i", "--data_path"),help="Dataset for building model" ),
  make_option(c("-m", "--mutation_data"),help="File containing list of mutations"),
)
