# Decision Agent Platform - Terraform Infrastructure
#
# This Terraform configuration creates all cloud infrastructure needed for
# the Decision Agent platform:
# - Redis cluster (for online feature store)
# - S3/Blob storage (for artifacts)
# - Secrets management
# - Network configuration
# - Monitoring resources

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    databricks = {
      source = "databricks/databricks"
      version = "~> 1.0"
    }
  }

  backend "s3" {
    bucket = "decision-agent-terraform-state"
    key    = "terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
}

provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

# Variables
variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "databricks_host" {
  description = "Databricks workspace URL"
  type        = string
}

variable "databricks_token" {
  description = "Databricks access token"
  type        = string
  sensitive   = true
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_nodes" {
  description = "Number of Redis nodes"
  type        = number
  default     = 2
}

# VPC Configuration
resource "aws_vpc" "decision_agent" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "decision-agent-vpc-${var.environment}"
    Environment = var.environment
    Project     = "decision-agent"
  }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.decision_agent.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name        = "decision-agent-private-${count.index + 1}-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.decision_agent.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name        = "decision-agent-public-${count.index + 1}-${var.environment}"
    Environment = var.environment
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

# Security Group for Redis
resource "aws_security_group" "redis" {
  name        = "decision-agent-redis-${var.environment}"
  description = "Security group for Redis cluster"
  vpc_id      = aws_vpc.decision_agent.id

  ingress {
    description = "Redis port"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.decision_agent.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "decision-agent-redis-sg-${var.environment}"
    Environment = var.environment
  }
}

# Redis Subnet Group
resource "aws_elasticache_subnet_group" "redis" {
  name       = "decision-agent-redis-subnet-${var.environment}"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name        = "decision-agent-redis-subnet-group-${var.environment}"
    Environment = var.environment
  }
}

# Redis Cluster
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "decision-agent-${var.environment}"
  replication_group_description = "Redis cluster for Decision Agent online feature store"

  engine               = "redis"
  engine_version       = "7.0"
  node_type            = var.redis_node_type
  number_cache_clusters = var.redis_num_nodes
  port                 = 6379

  subnet_group_name    = aws_elasticache_subnet_group.redis.name
  security_group_ids   = [aws_security_group.redis.id]

  automatic_failover_enabled = true
  multi_az_enabled           = true

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = 5
  snapshot_window          = "03:00-05:00"

  tags = {
    Name        = "decision-agent-redis-${var.environment}"
    Environment = var.environment
    Purpose     = "online-feature-store"
  }
}

# S3 Bucket for Artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket = "decision-agent-artifacts-${var.environment}"

  tags = {
    Name        = "decision-agent-artifacts-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Secrets Manager for sensitive data
resource "aws_secretsmanager_secret" "redis_password" {
  name = "decision-agent/${var.environment}/redis-password"

  tags = {
    Name        = "decision-agent-redis-password-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "redis_password" {
  secret_id     = aws_secretsmanager_secret.redis_password.id
  secret_string = random_password.redis_password.result
}

resource "random_password" "redis_password" {
  length  = 32
  special = true
}

# CloudWatch Log Group for monitoring
resource "aws_cloudwatch_log_group" "decision_agent" {
  name              = "/aws/decision-agent/${var.environment}"
  retention_in_days = 30

  tags = {
    Name        = "decision-agent-logs-${var.environment}"
    Environment = var.environment
  }
}

# Databricks Secret Scope (for storing Redis credentials)
resource "databricks_secret_scope" "decision_agent" {
  name = "decision_agent_${var.environment}"
}

resource "databricks_secret" "redis_host" {
  scope        = databricks_secret_scope.decision_agent.name
  key          = "redis_host"
  string_value = aws_elasticache_replication_group.redis.primary_endpoint_address
}

resource "databricks_secret" "redis_password" {
  scope        = databricks_secret_scope.decision_agent.name
  key          = "redis_password"
  string_value = random_password.redis_password.result
}

# Outputs
output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.redis.port
}

output "s3_bucket" {
  description = "S3 bucket for artifacts"
  value       = aws_s3_bucket.artifacts.bucket
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.decision_agent.id
}

output "databricks_secret_scope" {
  description = "Databricks secret scope name"
  value       = databricks_secret_scope.decision_agent.name
}
