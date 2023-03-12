variable "project" {
  description = "interview-mlops-engineer"
  default = "interview-mlops-engineer"
}

variable "region" {
  description = "Region for GCP resources. Choose as per your location: https://cloud.google.com/about/locations"
  default = "europe-west4"
  type = string
}

variable "zone" {
  type    = string
  default = "europe-west4-a"
}